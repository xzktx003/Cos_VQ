"""Select logical checkpoints on independent local text, before final tests."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.nowag_d6.finetune_hard_codebook import remove_training_overlap
from experiments.nowag_d6.logical_state import LINEAR_NAMES, get_linear_modules
from experiments.nowag_d6.v19_model_driver import atomic_json, file_sha256, validate_model_identity
from experiments.nowag_d6.v19_model_placement import place_evaluation_model, placement_summary, reconstruct_on_module


def heldout_tokens(cache, *, start, count, context_length, exclude_count):
    tokens = cache.get('finetune_tokens', cache.get('input_ids'))
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
        raise ValueError('Expected two-dimensional cached tokens')
    if min(count, exclude_count) < 1 or context_length < 2 or start < exclude_count:
        raise ValueError('Validation must follow a nonempty excluded training region')
    if start + count > len(tokens) or tokens.shape[1] < context_length:
        raise ValueError('Cache is too small for requested validation')
    train = tokens[:exclude_count, :context_length].long()
    valid = tokens[start:start + count, :context_length].long()
    valid, audit = remove_training_overlap(train, valid)
    audit.update(requested_start=start, requested_count=count, excluded_training_rows=exclude_count,
                 context_length=context_length,
                 token_sha256=hashlib.sha256(valid.cpu().contiguous().numpy().tobytes()).hexdigest())
    return valid, audit


@torch.no_grad()
def next_token_nll(model, tokens, device, *, chunk_size=128):
    if tokens.ndim != 2 or len(tokens) == 0 or tokens.shape[1] < 2 or chunk_size < 1:
        raise ValueError('NLL requires nonempty sequences with next-token targets')
    loss_sum, count = 0., 0
    for row in tokens:
        inputs = row[None].to(device)
        logits = model(input_ids=inputs, use_cache=False).logits
        for start in range(0, inputs.shape[1] - 1, chunk_size):
            end = min(start + chunk_size, inputs.shape[1] - 1)
            prediction = logits[:, start:end].float().reshape(-1, logits.shape[-1])
            targets = inputs[:, start + 1:end + 1].to(logits.device).reshape(-1)
            loss_sum += float(F.cross_entropy(prediction, targets, reduction='sum'))
            count += targets.numel()
        del logits
    nll = loss_sum / count
    if not math.isfinite(nll):
        raise ValueError('Nonfinite validation NLL')
    return {'nll': nll, 'predicted_tokens': count, 'sequences': len(tokens),
            'context_length': tokens.shape[1]}


def select_candidate(scores):
    if not scores or any(not math.isfinite(item['nll']) for item in scores):
        raise ValueError('Selection requires nonempty finite validation scores')
    # Input order is preregistered; put the unchanged baseline first.
    return min(scores, key=lambda item: item['nll'])


def select_family(scores, labels):
    if not labels or not set(labels).issubset({item['label'] for item in scores}):
        raise ValueError('Selection family must name existing candidates')
    return select_candidate([item for item in scores if item['label'] in labels])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--select-labels', nargs='+')
    parser.add_argument('--identity-manifest')
    parser.add_argument('--token-cache', required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--sequences', type=int, default=64)
    parser.add_argument('--exclude-sequences', type=int, required=True)
    parser.add_argument('--context-length', type=int, default=2048)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split-gpus', action='store_true', help='Keep complete decoder layers resident across exactly two visible GPUs')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh timestamped validation output')
    if len(args.checkpoints) != len(args.labels) or len(set(args.labels)) != len(args.labels):
        raise ValueError('Every checkpoint needs a unique matching label')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    cache = torch.load(args.token_cache, map_location='cpu', weights_only=False, mmap=True)
    valid, audit = heldout_tokens(cache, start=args.start, count=args.sequences,
                                 context_length=args.context_length, exclude_count=args.exclude_sequences)
    from transformers import AutoConfig, AutoModelForCausalLM, modeling_utils
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if getattr(modeling_utils, 'ALL_PARALLEL_STYLES', ()) is None:
        from experiments.qwen_qtip.qtip_runtime import install_tp_plan_compat
        install_tp_plan_compat(config)
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, torch_dtype=torch.bfloat16,
                                                local_files_only=True, attn_implementation='sdpa')
    model = place_evaluation_model(model, device, split_gpus=args.split_gpus)
    model.requires_grad_(False).eval()
    if int(valid.min()) < 0 or int(valid.max()) >= config.vocab_size:
        raise ValueError('Validation tokens outside model vocabulary')
    result = {'completed': False, 'arguments': vars(args), 'split_audit': audit,
              'model_placement': placement_summary(model),
              'selection': 'lowest independent validation NLL; baseline first resolves ties',
              'scores': [], 'uses_test_benchmarks_for_selection': False}
    for label, checkpoint_path in zip(args.labels, args.checkpoints):
        started = time.time()
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
        identity_manifest = args.identity_manifest
        if identity_manifest:
            identity_record = json.loads(Path(identity_manifest).read_text())
            if Path(checkpoint_path).resolve() != Path(identity_record['source_checkpoint']).resolve():
                # Derived outputs use a canonical model path and the original
                # manifest must never be used as a blanket model-name alias.
                identity_manifest = None
        validate_model_identity(checkpoint, cache, args.model,
                                identity_manifest=identity_manifest, source_path=checkpoint_path)
        if set(checkpoint['layers']) != {str(i) for i in range(len(model.model.layers))}:
            raise ValueError('Checkpoint must cover complete model')
        for index in range(len(model.model.layers)):
            modules = get_linear_modules(model, index)
            states = checkpoint['layers'][str(index)]
            if set(states) != set(LINEAR_NAMES):
                raise ValueError('Checkpoint must cover all seven Linears')
            for name in LINEAR_NAMES:
                reconstruct_on_module(modules[name], states[name])
        score = next_token_nll(model, valid, device)
        score.update(label=label, checkpoint=str(Path(checkpoint_path).resolve()),
                     checkpoint_sha256=file_sha256(checkpoint_path), seconds=time.time() - started)
        result['scores'].append(score)
        atomic_json(output, result)
        print(score, flush=True)
        del checkpoint
    result.update(completed=True, overall_selected=select_candidate(result['scores']),
                  selected=select_family(result['scores'], args.select_labels or args.labels))
    atomic_json(output, result)


if __name__ == '__main__':
    main()
