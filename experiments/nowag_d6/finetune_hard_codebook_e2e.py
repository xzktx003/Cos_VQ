"""Optimize VQ metadata using text cross entropy or frozen-teacher distillation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from experiments.nowag_d6.hard_codebook import HardCodebookWeight
from experiments.nowag_d6.finetune_hard_codebook import remove_training_overlap, split_finetune_tokens, write_json


def distillation_kl(student_logits, teacher_logits, chunk_size=128):
    """Mean next-token KL(teacher || student), temperature 1, in FP32.

    Recompute each chunk on backward instead of retaining FP32 vocabulary-sized
    intermediates for every token. The final position has no next-token target.
    """
    if student_logits.ndim != 3 or student_logits.shape != teacher_logits.shape:
        raise ValueError('Teacher and student logits must have matching [batch, sequence, vocabulary] shapes')
    batch, length, vocab = student_logits.shape
    if batch < 1 or length < 2 or vocab < 1 or chunk_size <= 0:
        raise ValueError('KL requires nonempty next-token positions and a positive chunk size')

    def chunk_kl(student, teacher):
        log_student = student.float().log_softmax(-1)
        log_teacher = teacher.detach().float().log_softmax(-1)
        return F.kl_div(log_student, log_teacher, log_target=True, reduction='sum')

    total = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, length - 1, chunk_size):
        end = min(start + chunk_size, length - 1)
        student = student_logits[:, start:end]
        teacher = teacher_logits[:, start:end].detach()
        if torch.is_grad_enabled() and student.requires_grad:
            value = checkpoint(chunk_kl, student, teacher, use_reentrant=False)
        else:
            value = chunk_kl(student, teacher)
        total = total + value
    return total / (batch * (length - 1))


class HardCodebookLinear(nn.Module):
    """Permanent module replacement also used during checkpoint recomputation."""

    def __init__(self, linear, state, alternative_assignments=None, residual_rank=0, *, tune_group_scales=False):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        if tune_group_scales and (alternative_assignments is not None or residual_rank):
            raise ValueError('Group-scale training requires fixed assignments and no residual factors')
        if tuple(int(v) for v in state['shape']) != tuple(linear.weight.shape):
            raise ValueError('Logical checkpoint Linear shape mismatch')
        if residual_rank:
            if alternative_assignments is not None:
                raise ValueError('Residual compensation with binary assignments is unsupported')
            from experiments.nowag_d6.residual_codebook import ResidualCodebookWeight
            self.quantizer = ResidualCodebookWeight(state, rank=residual_rank)
        elif {'residual_A', 'residual_B'}.intersection(state):
            raise ValueError('Source has residual factors; explicitly set matching residual-rank')
        elif alternative_assignments is None:
            self.quantizer = HardCodebookWeight(state, tune_group_scales=tune_group_scales)
        else:
            from experiments.nowag_d6.binary_assignment import BinaryAssignmentWeight
            self.quantizer = BinaryAssignmentWeight(state, alternative_assignments)
        self.register_buffer('bias', linear.bias.detach().clone() if linear.bias is not None else None)

    def forward(self, inputs):
        return F.linear(inputs, self.quantizer().to(inputs.dtype), self.bias)


def validate_output_paths(source_path, output_dir, run_stamp):
    """Reject reused or source-aliasing artifacts before creating a run directory."""
    source_path = Path(source_path).resolve(strict=True)
    root = Path(output_dir).resolve()
    checkpoint_path = root / f'{run_stamp}_checkpoint.pt'
    progress_path = root / f'{run_stamp}_progress.json'
    for path in (checkpoint_path, checkpoint_path.with_suffix('.pt.tmp'),
                 progress_path, progress_path.with_suffix('.json.tmp')):
        if path.resolve() == source_path or (path.exists() and path.samefile(source_path)):
            raise ValueError(f'Output would alias the source checkpoint: {path}')
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Use a new timestamped run directory; artifact already exists: {path}')
    return checkpoint_path, progress_path


def validate_exported_state(state, original, allow_assignment_changes=False, residual_rank=0, *, tune_group_scales=False):
    trainable_keys = {'assignments', 'codebook', 'row_norm', 'col_norm'}
    if tune_group_scales:
        if allow_assignment_changes or residual_rank or 'group_scale' not in original:
            raise ValueError('Group-scale training requires grouped fixed-assignment state')
        for value in (original['group_scale'], state.get('group_scale')):
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float16 or not torch.isfinite(value).all() or not (value > 0).all():
                raise ValueError('Group scales must remain positive finite FP16')
        trainable_keys.add('group_scale')
    residual_keys = {'residual_A', 'residual_B'} if residual_rank else set()
    if residual_rank:
        if allow_assignment_changes:
            raise ValueError('Residual compensation requires fixed assignments')
        rows, columns = (int(v) for v in original['shape'])
        if residual_rank < 1 or residual_rank > min(rows, columns):
            raise ValueError('Invalid residual rank')
        for key, shape in [('residual_A', (residual_rank, columns)), ('residual_B', (rows, residual_rank))]:
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float16 or not torch.isfinite(value).all():
                raise ValueError('Invalid serialized residual factor')
    if set(state) != set(original) | residual_keys:
        raise ValueError('E2E changed logical format')
    for key, before in original.items():
        after = state[key]
        if isinstance(before, torch.Tensor):
            if after.shape != before.shape or after.dtype != before.dtype:
                raise ValueError('E2E changed storage budget')
            if key not in trainable_keys | residual_keys and not torch.equal(before, after):
                raise ValueError('E2E changed nontrainable metadata')
        elif after != before:
            raise ValueError('E2E changed nontrainable metadata')
    ids = state['assignments'].long()
    if int(ids.min()) < 0 or int(ids.max()) >= len(state['codebook']):
        raise ValueError('Exported assignment out of range')
    if not allow_assignment_changes and not torch.equal(state['assignments'], original['assignments']):
        raise ValueError('E2E changed fixed assignments')
    for key in ('codebook', 'row_norm', 'col_norm'):
        if state[key].dtype != torch.float16 or not torch.isfinite(state[key]).all():
            raise ValueError('Invalid serialized parameters')


def load_e2e_base_model(model_path):
    from transformers import AutoConfig, AutoModelForCausalLM, modeling_utils
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if getattr(modeling_utils, 'ALL_PARALLEL_STYLES', ()) is None:
        from experiments.qwen_qtip.qtip_runtime import install_tp_plan_compat
        install_tp_plan_compat(config)
    return AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa')


def training_data_description(dataset):
    return {'dataset': dataset, 'objective': 'next_token_cross_entropy',
            'selection': 'disjoint_validation_nll'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--token-cache', required=True)
    parser.add_argument('--dataset-label', default='RedPajama-1T')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--run-stamp', default=time.strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--train-sequences', type=int, default=320)
    parser.add_argument('--validation-sequences', type=int, default=64)
    parser.add_argument('--context-length', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--learning-rate-selection', help='Reuse a completed same-source development-selected rate for seed replication')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--binary-assignments', action='store_true')
    parser.add_argument('--distillation', action='store_true')
    parser.add_argument('--tune-group-scales', action='store_true')
    parser.add_argument('--residual-rank', type=int, default=0)
    parser.add_argument('--residual-learning-rate', type=float, default=0.001)
    parser.add_argument('--assignment-learning-rate', type=float, default=0.01)
    parser.add_argument('--validation-interval', type=int, default=0)
    args = parser.parse_args()
    if args.learning_rate_selection:
        from experiments.nowag_d6.v19_e2e_selection import selected_learning_rate
        args.learning_rate = selected_learning_rate(args.learning_rate_selection, args.checkpoint)
    if args.tune_group_scales and (args.binary_assignments or args.residual_rank):
        raise ValueError('Group-scale training requires fixed assignments and no residual factors')
    if args.residual_rank < 0 or not math.isfinite(args.residual_learning_rate) or args.residual_learning_rate <= 0:
        raise ValueError('Invalid residual training configuration')
    if args.residual_rank and (args.binary_assignments or args.distillation):
        raise ValueError('Residual compensation with binary assignments or distillation is unsupported')
    if args.distillation and args.binary_assignments:
        raise ValueError('Distillation with binary assignments is not supported in this experiment')
    if not re.fullmatch(r'\d{8}_\d{6}', args.run_stamp):
        raise ValueError('run-stamp must be YYYYMMDD_HHMMSS')
    if min(args.epochs, args.batch_size, args.learning_rate) <= 0:
        raise ValueError('Training parameters must be positive')
    if args.validation_interval < 0 or not math.isfinite(args.assignment_learning_rate) or args.assignment_learning_rate <= 0:
        raise ValueError('Invalid assignment training configuration')
    checkpoint_path, progress_path = validate_output_paths(args.checkpoint, args.output_dir, args.run_stamp)
    from experiments.nowag_d6.logical_state import LINEAR_NAMES, reconstruct_logical_state

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = not bool(args.residual_rank)
    device = torch.device('cuda:0')
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cache = torch.load(args.token_cache, map_location='cpu', weights_only=False)
    train_ids, valid_ids = split_finetune_tokens(cache['finetune_tokens'], args.train_sequences,
                                               args.validation_sequences, args.context_length)
    valid_ids, split_audit = remove_training_overlap(train_ids, valid_ids)
    model = load_e2e_base_model(args.model)
    model.requires_grad_(False)
    model.config.use_cache = False
    if cache['metadata']['tokenizer_vocab_size'] != model.config.vocab_size:
        raise ValueError('Calibration vocabulary mismatch')
    if int(train_ids.min()) < 0 or int(valid_ids.min()) < 0 or max(int(train_ids.max()), int(valid_ids.max())) >= model.config.vocab_size:
        raise ValueError('Calibration tokens out of range')
    tokenizer_sha = {}
    for filename in ('tokenizer.json', 'tokenizer_config.json'):
        sha = hashlib.sha256((Path(args.model) / filename).read_bytes()).hexdigest()
        if sha != hashlib.sha256((Path(cache['metadata']['tokenizer_model']) / filename).read_bytes()).hexdigest():
            raise ValueError('Calibration tokenizer identity mismatch')
        tokenizer_sha[filename] = sha
    if set(source['layers']) != {str(i) for i in range(len(model.model.layers))}:
        raise ValueError('Source checkpoint must cover the full model')
    replacements = {}
    for index, block in enumerate(model.model.layers):
        states = source['layers'][str(index)]
        if set(states) != set(LINEAR_NAMES):
            raise ValueError('Source checkpoint must cover seven Linears per block')
        for name in LINEAR_NAMES:
            parent_name, attribute = name.rsplit('.', 1)
            parent = block.get_submodule(parent_name)
            linear = getattr(parent, attribute)
            alternatives = None
            if args.binary_assignments:
                from experiments.nowag_d6.binary_assignment import build_alternative_assignments
                alternatives = build_alternative_assignments(states[name], linear.weight.detach(), device=device)
            replacement = HardCodebookLinear(linear, states[name], alternatives, args.residual_rank,
                                             tune_group_scales=args.tune_group_scales)
            setattr(parent, attribute, replacement)
            replacements[(str(index), name)] = replacement
        if args.binary_assignments:
            print(json.dumps({'event': 'assignment_candidates_ready', 'layer': index}), flush=True)
    model.to(device)
    teacher = None
    if args.distillation:
        teacher = load_e2e_base_model(args.model).requires_grad_(False).eval().to(device)
        teacher.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    # Dropout would confound validation and checkpoint recomputation for this PTQ run.
    if getattr(model.config, 'attention_dropout', 0) != 0 or any(isinstance(m, nn.Dropout) and m.p != 0 for m in model.modules()):
        raise ValueError('This protocol requires a dropout-free base model')
    parameters = [p for p in model.parameters() if p.requires_grad]
    if args.residual_rank:
        residual_parameters = [p for m in replacements.values()
                               for p in (m.quantizer.residual_A, m.quantizer.residual_B)]
        residual_ids = {id(p) for p in residual_parameters}
        optimizer = torch.optim.Adam([
            {'params': [p for p in parameters if id(p) not in residual_ids], 'lr': args.learning_rate},
            {'params': residual_parameters, 'lr': args.residual_learning_rate},
        ], foreach=False)
    elif args.binary_assignments:
        assignment_parameters = [m.quantizer.assignment_logits for m in replacements.values()]
        assignment_parameter_ids = {id(p) for p in assignment_parameters}
        ordinary_parameters = [p for p in parameters if id(p) not in assignment_parameter_ids]
        optimizer = torch.optim.Adam([
            {'params': ordinary_parameters, 'lr': args.learning_rate},
            {'params': assignment_parameters, 'lr': args.assignment_learning_rate},
        ], foreach=False)
    else:
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    report = {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'arguments': vars(args),
              **training_data_description(args.dataset_label),
              'trainable_parameters': sum(p.numel() for p in parameters),
              'tokenizer_sha256': tokenizer_sha, 'epochs': [], 'completed': False,
              'split_audit': split_audit,
              'source_metadata': source.get('metadata', {}), 'step_validations': []}
    report['tune_group_scales'] = args.tune_group_scales
    if args.tune_group_scales:
        report['group_scale_training'] = {
            'parameters': sum(m.quantizer.group_scale_delta.numel() for m in replacements.values()),
            'storage_dtype': 'float16', 'additional_bits': 0, 'assignments_fixed': True,
            'parameterization': 'multiplicative_delta_positive_fp16_projection_ste'}
    if args.residual_rank:
        residual_bits = sum(m.quantizer.residual_bits for m in replacements.values())
        weight_count = sum(m.in_features * m.out_features for m in replacements.values())
        report['objective'] = 'rp1t_next_token_cross_entropy_with_low_rank_compensation'
        report['residual_compensation'] = {'rank': args.residual_rank, 'additional_bits': residual_bits,
                                           'additional_bits_per_quantized_weight': residual_bits / weight_count,
                                           'storage_dtype': 'float16',
                                           'initialization': 'source' if all('residual_A' in source['layers'][layer][name]
                                                                          for layer, name in replacements) else 'A_normal_std_0.01_B_zero'}
    if teacher is not None:
        report.update(objective='rp1t_next_token_kl_teacher_to_student', distillation={
            'teacher_model': args.model, 'teacher_dtype': 'bfloat16',
            'teacher_parameters': sum(p.numel() for p in teacher.parameters()),
            'teacher_trainable_parameters': 0, 'temperature': 1.0,
            'kl_dtype': 'float32', 'sequence_chunk_size': 128,
            'reduction': 'mean_next_token', 'teacher_saved_in_checkpoint': False})

    @torch.no_grad()
    def export_and_validate():
        model.eval()
        changed_assignments = 0
        exported = {'metadata': dict(source.get('metadata', {})), 'layers': {key: {} for key in source['layers']}}
        for (layer, name), module in replacements.items():
            state = module.quantizer.export_state()
            original = source['layers'][layer][name]
            validate_exported_state(state, original, args.binary_assignments, args.residual_rank,
                                    tune_group_scales=args.tune_group_scales)
            if args.binary_assignments:
                changed_assignments += int((state['assignments'].long() != original['assignments'].long()).sum())
            actual = module.quantizer().to(torch.bfloat16)
            expected = reconstruct_logical_state(state, device).to(torch.bfloat16)
            if not torch.equal(actual, expected):
                raise ValueError('Training weight differs from serialized deployment weight')
            exported['layers'][layer][name] = state
        nll = 0.0
        for ids in valid_ids:
            batch = ids[None].to(device)
            loss = model(input_ids=batch, labels=batch, use_cache=False).loss
            nll += float(loss)
        nll /= len(valid_ids)
        if not math.isfinite(nll):
            raise ValueError('Nonfinite validation NLL')
        report['last_assignment_changes'] = changed_assignments
        return exported, nll

    def save_best(exported):
        exported['metadata']['hard_codebook_e2e'] = vars(args)
        temp = checkpoint_path.with_suffix('.pt.tmp')
        torch.save(exported, temp)
        temp.replace(checkpoint_path)

    initial, best_nll = export_and_validate()
    save_best(initial)
    del initial
    report.update(initial_validation_nll=best_nll, best_validation_nll=best_nll, selected_epoch=-1)
    write_json(progress_path, report)
    print(json.dumps({'event': 'initial_validation', 'nll': best_nll}), flush=True)
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        model.train()
        started = time.time()
        order = torch.randperm(len(train_ids), generator=generator).tolist()
        train_nll = 0.0
        for step, start in enumerate(range(0, len(order), args.batch_size)):
            indices = order[start:start+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            for index in indices:
                batch = train_ids[index:index+1].to(device)
                if teacher is None:
                    loss = model(input_ids=batch, labels=batch, use_cache=False).loss
                else:
                    with torch.no_grad():
                        teacher_logits = teacher(input_ids=batch, use_cache=False).logits
                    student_logits = model(input_ids=batch, use_cache=False).logits
                    loss = distillation_kl(student_logits, teacher_logits)
                    del student_logits, teacher_logits
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite training loss')
                (loss / len(indices)).backward()
                train_nll += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if args.validation_interval and (step + 1) % args.validation_interval == 0 and start + args.batch_size < len(order):
                optimizer.zero_grad(set_to_none=True)
                exported, validation_nll = export_and_validate()
                if validation_nll < best_nll:
                    save_best(exported)
                    best_nll = validation_nll
                    report.update(best_validation_nll=best_nll, selected_epoch=epoch, selected_step=step + 1,
                                  selected_assignment_changes=report['last_assignment_changes'])
                del exported
                entry = {'epoch': epoch, 'step': step + 1, 'validation_nll': validation_nll}
                report['step_validations'].append(entry)
                write_json(progress_path, report)
                print(json.dumps({'event': 'step_validation', **entry}), flush=True)
                model.train()
            if step % 20 == 0:
                print(json.dumps({'event': 'train_step', 'epoch': epoch, 'step': step,
                                  ('mean_kl' if args.distillation else 'mean_nll'): train_nll / min(start + args.batch_size, len(order))}), flush=True)
        optimizer.zero_grad(set_to_none=True)
        exported, validation_nll = export_and_validate()
        if validation_nll < best_nll:
            save_best(exported)
            best_nll = validation_nll
            report.update(best_validation_nll=best_nll, selected_epoch=epoch,
                          selected_step=math.ceil(len(order) / args.batch_size),
                          selected_assignment_changes=report['last_assignment_changes'])
        del exported
        entry = {'epoch': epoch, ('train_kl' if args.distillation else 'train_nll'): train_nll / len(order),
                 'validation_nll': validation_nll, 'seconds': time.time() - started}
        report['epochs'].append(entry)
        write_json(progress_path, report)
        print(json.dumps({'event': 'epoch_complete', **entry}), flush=True)
    report['completed'] = True
    write_json(progress_path, report)


if __name__ == '__main__':
    main()
