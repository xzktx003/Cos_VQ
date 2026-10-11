"""Refine existing VQ metadata against the independent dense trajectory.

No assignment, extra residual, or extra per-group scale is introduced. Epochs
are selected using calibration validation MSE of FP16 serialized hard states.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import time
from pathlib import Path

import torch

from experiments.nowag_d6.hard_codebook import HardCodebookWeight


def split_finetune_tokens(tokens, train_count, valid_count, context_length):
    if min(train_count, valid_count, context_length) <= 0:
        raise ValueError('Split sizes and context length must be positive')
    if tokens.ndim != 2 or tokens.shape[0] < train_count + valid_count or tokens.shape[1] < context_length:
        raise ValueError('Insufficient calibration tokens for disjoint splits')
    return (tokens[:train_count, :context_length].to(torch.int64).contiguous(),
            tokens[train_count:train_count + valid_count, :context_length].to(torch.int64).contiguous())


def validate_baseline_ppl(value, reference, tolerance=0.1):
    if not math.isfinite(value) or value <= 0 or abs(value - reference) > tolerance:
        raise ValueError(f'Baseline environment discrepancy must be resolved first: {value}')


def remove_training_overlap(train_ids, valid_ids):
    """Keep validation content independent even when source rows repeat."""
    def digest(row):
        return hashlib.sha256(row.to('cpu', torch.int64).contiguous().numpy().tobytes()).hexdigest()
    training = {digest(row) for row in train_ids}
    seen, kept, overlap, duplicates = set(), [], [], []
    for index, row in enumerate(valid_ids):
        key = digest(row)
        if key in training:
            overlap.append(index)
        elif key in seen:
            duplicates.append(index)
        else:
            seen.add(key)
            kept.append(index)
    if not kept:
        raise ValueError('No independent validation sequences remain')
    return valid_ids[kept].contiguous(), {
        'unique_training_sequences': len(training),
        'kept_validation_rows': kept,
        'removed_training_overlap_rows': overlap,
        'removed_validation_duplicate_rows': duplicates,
        'validation_sequences_after_filtering': len(kept),
        'exact_train_validation_overlap_after_filtering': 0,
    }


def train_hard_block(states, train_inputs, train_targets, valid_inputs, valid_targets,
                     forward, *, device, epochs, batch_size, learning_rate, seed,
                     tune_codebook=True, tune_norms=True, progress=None):
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError('Epochs, batch size and learning rate must be positive')
    for inputs, targets in ((train_inputs, train_targets), (valid_inputs, valid_targets)):
        if inputs.ndim != 3 or inputs.shape != targets.shape or inputs.shape[0] == 0:
            raise ValueError('Inputs and targets must be nonempty matching 3D tensors')
    modules = {name: HardCodebookWeight(state, tune_codebook=tune_codebook,
                                       tune_norms=tune_norms).to(device)
               for name, state in states.items()}
    parameters = [p for module in modules.values() for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    def export():
        return {name: module.export_state() for name, module in modules.items()}

    @torch.no_grad()
    def validation_loss():
        # export/reload deliberately excludes optimizer-only FP32 precision.
        deployed = {name: HardCodebookWeight(state, tune_codebook=False,
                                             tune_norms=False).to(device)()
                    for name, state in export().items()}
        error, count = 0.0, 0
        for index in range(len(valid_inputs)):
            output = forward(valid_inputs[index:index+1].to(device), deployed).float()
            target = valid_targets[index:index+1].to(device).float()
            error += (output - target).square().sum().item()
            count += target.numel()
        return error / count

    best_loss = validation_loss()
    if not math.isfinite(best_loss):
        raise ValueError('Nonfinite initial validation loss')
    best = export()
    stats = {'initial_validation_mse': best_loss, 'best_validation_mse': best_loss,
             'selected_epoch': -1, 'epochs': []}
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        order = torch.randperm(len(train_inputs), generator=generator).tolist()
        loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start+batch_size]
            optimizer.zero_grad(set_to_none=True)
            for index in indices:
                weights = {name: module() for name, module in modules.items()}
                output = forward(train_inputs[index:index+1].to(device), weights).float()
                target = train_targets[index:index+1].to(device).float()
                mse = (output - target).square().mean()
                loss = mse / target.square().mean().clamp_min(1e-8)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite training loss')
                (loss / len(indices)).backward()
                loss_sum += loss.item()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
        valid_loss = validation_loss()
        if not math.isfinite(valid_loss):
            raise ValueError('Nonfinite validation loss')
        if valid_loss < best_loss:
            best_loss, best = valid_loss, export()
            stats.update(best_validation_mse=best_loss, selected_epoch=epoch)
        entry = {'epoch': epoch, 'train_relative_mse': loss_sum / len(order),
                 'validation_mse': valid_loss, 'best_validation_mse': best_loss}
        stats['epochs'].append(entry)
        if progress:
            progress(entry)
    return best, stats


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--token-cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--run-stamp', default=time.strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--train-sequences', type=int, default=320)
    parser.add_argument('--validation-sequences', type=int, default=64)
    parser.add_argument('--context-length', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-layers', type=int, default=0)
    parser.add_argument('--norms-only', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'\d{8}_\d{6}', args.run_stamp):
        raise ValueError('run-stamp must be YYYYMMDD_HHMMSS')
    from transformers import AutoModelForCausalLM
    from experiments.nowag_d6.diagnose_block_vector_gsq import LINEAR_NAMES, get_linear_modules
    from experiments.nowag_d6.staged_vector_gsq import run_hidden_stage_output

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda:0')
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    progress_path = root / f'{args.run_stamp}_progress.json'
    checkpoint_path = root / f'{args.run_stamp}_checkpoint.pt'
    if progress_path.exists():
        raise FileExistsError('Use a new timestamped output directory; resume is not implemented')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cache = torch.load(args.token_cache, map_location='cpu', weights_only=False)
    train_ids, valid_ids = split_finetune_tokens(cache['finetune_tokens'], args.train_sequences,
                                               args.validation_sequences, args.context_length)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                local_files_only=True, attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    if cache['metadata']['tokenizer_vocab_size'] != model.config.vocab_size:
        raise ValueError('Calibration tokenizer vocabulary does not match model')
    if cache['metadata']['tokenizer_model_type'] != model.config.model_type:
        raise ValueError('Calibration model type mismatch')
    layer_count = len(model.model.layers)
    if not 0 <= args.max_layers <= layer_count:
        raise ValueError('max-layers must be zero or within model layer count')
    if set(checkpoint['layers']) != {str(i) for i in range(layer_count)}:
        raise ValueError('Source checkpoint must cover the full model')
    tokenizer_fingerprints = {}
    for filename in ('tokenizer.json', 'tokenizer_config.json'):
        base_file = Path(args.model) / filename
        cache_file = Path(cache['metadata']['tokenizer_model']) / filename
        base_sha = hashlib.sha256(base_file.read_bytes()).hexdigest()
        if base_sha != hashlib.sha256(cache_file.read_bytes()).hexdigest():
            raise ValueError(f'Calibration tokenizer identity mismatch: {filename}')
        tokenizer_fingerprints[filename] = base_sha
    ids = torch.cat((train_ids, valid_ids))
    if int(ids.min()) < 0 or int(ids.max()) >= model.config.vocab_size:
        raise ValueError('Calibration token IDs out of range')
    with torch.no_grad():
        dense_hidden = model.model.embed_tokens(ids)
    student_hidden = dense_hidden.clone()
    rotary = getattr(model.model, 'rotary_emb', None)
    if rotary is not None:
        rotary.to(device)
    report = {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'arguments': vars(args),
              'method': 'hard_fixed_assignment_codebook_norm_dense_trajectory',
              'selection': 'disjoint_calibration_validation_serialized_hard_block_mse',
              'completed': False, 'layers': [], 'calibration_metadata': cache['metadata']}
    report['source_metadata'] = checkpoint.get('metadata', {})
    report['tokenizer_sha256'] = tokenizer_fingerprints
    report['model_config_sha256'] = hashlib.sha256((Path(args.model) / 'config.json').read_bytes()).hexdigest()
    write_json(progress_path, report)

    @torch.no_grad()
    def propagate(hidden, callback, weights):
        result = torch.empty_like(hidden)
        for index in range(len(hidden)):
            result[index:index+1].copy_(callback(hidden[index:index+1].to(device), weights).cpu())
        return result

    for index in range(args.max_layers or layer_count):
        started = time.time()
        block = model.model.layers[index].to(device)
        modules = get_linear_modules(model, index)
        states = checkpoint['layers'][str(index)]
        if set(states) != set(LINEAR_NAMES):
            raise ValueError(f'Incomplete Linear coverage at layer {index}')
        for name, module in modules.items():
            if tuple(states[name]['shape'].tolist()) != tuple(module.weight.shape):
                raise ValueError(f'Checkpoint shape mismatch at layer {index} {name}')
        def forward(inputs, weights, layer_index=index, linears=modules):
            return run_hidden_stage_output(model, layer_index, linears, inputs, weights, 'block')
        dense_targets = propagate(dense_hidden, forward, None)
        def progress(entry):
            print(json.dumps({'layer': index, **entry}), flush=True)
        selected, stats = train_hard_block(
            states, student_hidden[:args.train_sequences], dense_targets[:args.train_sequences],
            student_hidden[args.train_sequences:], dense_targets[args.train_sequences:], forward,
            device=device, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, seed=args.seed + index,
            tune_codebook=not args.norms_only, progress=progress)
        for name, state in selected.items():
            source = states[name]
            if set(source) != set(state) or not torch.equal(source['assignments'], state['assignments']):
                raise ValueError('Refinement changed logical format or assignments')
            for key in ('codebook', 'row_norm', 'col_norm'):
                if source[key].shape != state[key].shape or state[key].dtype != torch.float16:
                    raise ValueError('Refinement changed metadata storage budget')
        with torch.no_grad():
            weights = {name: HardCodebookWeight(state, tune_codebook=False, tune_norms=False).to(device)()
                       for name, state in selected.items()}
            student_hidden = propagate(student_hidden, forward, weights)
        dense_hidden = dense_targets
        checkpoint['layers'][str(index)] = selected
        checkpoint.setdefault('metadata', {})['hard_codebook_finetune'] = vars(args)
        temp = checkpoint_path.with_suffix('.pt.tmp')
        torch.save(checkpoint, temp)
        temp.replace(checkpoint_path)
        report['layers'].append({'layer': index, 'seconds': time.time() - started, **stats})
        write_json(progress_path, report)
        print(json.dumps({'event': 'layer_complete', **report['layers'][-1]}), flush=True)
        del weights, selected, states, modules, forward
        block.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
    report['completed'] = True
    report['refined_full_model'] = len(report['layers']) == layer_count
    write_json(progress_path, report)


if __name__ == '__main__':
    main()
