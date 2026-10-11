"""Validate V19 on frozen-codebook logical checkpoints, one block at a time.

Inputs for the seven local objectives are captured on the original quantized
block and current optimized prefix. Each block's inputs remain fixed during its
local optimization; updated block outputs supply the next block's inputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from torch import nn

from experiments.nowag_d6.logical_state import (
    LINEAR_NAMES, get_linear_modules, logical_state_bits, reconstruct_logical_state,
)


def move_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: move_tree(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_tree(item, device) for item in value)
    if isinstance(value, list):
        return [move_tree(item, device) for item in value]
    return value


class _CapturedInput(Exception):
    pass


@torch.no_grad()
def capture_first_layer_inputs(model, tokens, device):
    batches = []

    class Catcher(nn.Module):
        def forward(self, hidden_states, **kwargs):
            batches.append((hidden_states.detach().cpu(), move_tree(kwargs, 'cpu')))
            raise _CapturedInput

    original = model.model.layers[0]
    moved = []
    for name in ('embed_tokens', 'rotary_emb'):
        module = getattr(model.model, name, None)
        if module is not None:
            tensors = list(module.parameters()) + list(module.buffers())
            moved.append((module, tensors[0].device if tensors else torch.device('cpu')))
            module.to(device)
    model.model.layers[0] = Catcher()
    try:
        for row in tokens:
            try:
                model(input_ids=row[None].to(device), use_cache=False)
            except _CapturedInput:
                continue
            raise RuntimeError('First block did not capture model input')
    finally:
        model.model.layers[0] = original
        for module, original_device in moved:
            module.to(original_device)
    return batches


def _row_hash(row):
    return hashlib.sha256(row.to('cpu', torch.int64).contiguous().numpy().tobytes()).hexdigest()


def select_calibration_tokens(cache, *, count, context_length, start, development_start, development_count):
    tokens = cache.get('finetune_tokens', cache.get('input_ids'))
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
        raise ValueError('Cache needs a two-dimensional finetune_tokens or input_ids tensor')
    if min(count, context_length, development_count) <= 0 or min(start, development_start) < 0:
        raise ValueError('Invalid calibration split configuration')
    if tokens.shape[1] < context_length or development_start + development_count > len(tokens):
        raise ValueError('Cache does not cover the requested context or development split')
    development = {_row_hash(row[:context_length])
                   for row in tokens[development_start:development_start + development_count]}
    selected, hashes, rows = [], set(), []
    for index in range(start, len(tokens)):
        if development_start <= index < development_start + development_count:
            continue
        row = tokens[index, :context_length].to(torch.int64)
        digest = _row_hash(row)
        if digest in development or digest in hashes:
            continue
        hashes.add(digest)
        selected.append(row)
        rows.append(index)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError('Not enough unique calibration rows independent of development data')
    return torch.stack(selected), {'selected_rows': rows, 'sequences': count,
                                   'context_length': context_length, 'tokens': count * context_length,
                                   'development_start': development_start,
                                   'development_count': development_count,
                                   'exact_development_overlap': len(hashes & development),
                                   'tokens_sha256': _row_hash(torch.stack(selected))}


@torch.no_grad()
def collect_block_hessians(block, modules, batches, device):
    hessians = {name: torch.zeros(module.in_features, module.in_features,
                                 dtype=torch.float64, device=device)
                for name, module in modules.items()}
    counts = {name: 0 for name in modules}
    handles = []

    def hook_for(name):
        def capture(_module, inputs):
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).double()
            hessians[name].addmm_(values.T, values)
            counts[name] += len(values)
        return capture

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(hook_for(name)))
        for hidden, kwargs in batches:
            block(hidden.to(device), **move_tree(kwargs, device))
    finally:
        for handle in handles:
            handle.remove()
    for name, value in hessians.items():
        if counts[name] == 0:
            raise ValueError(f'No calibration activations for {name}')
        value.div_(counts[name])
        if not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite calibration Hessian for {name}')
    return hessians, counts


@torch.no_grad()
def propagate_block(block, batches, device):
    result = []
    for hidden, kwargs in batches:
        output = block(hidden.to(device), **move_tree(kwargs, device))
        output = output[0] if isinstance(output, (tuple, list)) else output
        result.append((output.detach().cpu(), kwargs))
    return result


def validate_unchanged_format(original, updated, *, grouped):
    allowed = {'assignments', 'group_scale' if grouped else 'row_norm'}
    if set(original) != set(updated):
        raise ValueError('Changed checkpoint format')
    for name, before in original.items():
        after = updated[name]
        if isinstance(before, torch.Tensor):
            if before.shape != after.shape or before.dtype != after.dtype:
                raise ValueError(f'Changed storage shape/dtype: {name}')
            if name not in allowed and not torch.equal(before.cpu(), after.cpu()):
                raise ValueError(f'Changed fixed metadata: {name}')
            if torch.is_floating_point(after) and not torch.isfinite(after).all():
                raise ValueError(f'Nonfinite exported metadata: {name}')
        elif before != after:
            raise ValueError(f'Changed fixed metadata: {name}')
    if logical_state_bits(original) != logical_state_bits(updated):
        raise ValueError('Changed logical bit budget')
    ids = updated['assignments']
    if ids.min() < 0 or ids.max() >= len(updated['codebook']):
        raise ValueError('Exported code index outside codebook')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_identity_manifest(manifest_path, source_path, checkpoint_model, target):
    """Bind one migrated checkpoint to an explicitly recorded local model."""
    if source_path is None:
        raise ValueError('Identity manifest requires source_path')
    try:
        path = Path(manifest_path).resolve(strict=True)
        checkpoint_path = Path(source_path).resolve(strict=True)
        manifest = json.loads(path.read_text())
    except (OSError, ValueError, TypeError) as error:
        raise ValueError('Identity manifest or source path cannot be read') from error
    if not isinstance(manifest, dict) or manifest.get('format') != 'v19-model-identity-v1':
        raise ValueError('Identity manifest has unsupported format')

    def require_path(value, expected, label):
        if not isinstance(value, str) or Path(value).resolve() != expected:
            raise ValueError(f'Identity manifest {label} mismatch')

    def require_hash(file, expected, label):
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
            raise ValueError(f'Identity manifest {label} SHA256 is required')
        try:
            actual = file_sha256(file)
        except OSError as error:
            raise ValueError(f'Identity manifest {label} file cannot be read') from error
        if actual != expected:
            raise ValueError(f'Identity manifest {label} SHA256 mismatch')
        return actual

    require_path(manifest.get('source_checkpoint'), checkpoint_path, 'source checkpoint path')
    require_path(manifest.get('requested_model'), target, 'requested model path')
    if manifest.get('checkpoint_metadata_model') != checkpoint_model:
        raise ValueError('Identity manifest checkpoint metadata model mismatch')
    source_sha = require_hash(checkpoint_path, manifest.get('source_sha256'), 'source checkpoint')
    files = manifest.get('model_files_sha256')
    if not isinstance(files, dict):
        raise ValueError('Identity manifest model file hashes are required')
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        require_hash(target / name, files.get(name), name)
    history = manifest.get('historical_e2e_progress')
    if not isinstance(history, dict) or not isinstance(history.get('path'), str):
        raise ValueError('Identity manifest historical E2E progress is required')
    history_path = Path(history['path']).resolve()
    history_sha = require_hash(history_path, history.get('sha256'), 'historical E2E progress')
    try:
        progress = json.loads(history_path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError('Identity manifest historical E2E progress is invalid') from error
    if not isinstance(progress, dict) or progress.get('completed') is not True:
        raise ValueError('Identity manifest historical E2E progress must be complete')
    metadata, arguments = progress.get('source_metadata'), progress.get('arguments')
    if not isinstance(metadata, dict) or metadata.get('model') != checkpoint_model:
        raise ValueError('Identity manifest historical source model mismatch')
    if not isinstance(arguments, dict):
        raise ValueError('Identity manifest historical E2E arguments are required')
    require_path(arguments.get('checkpoint'), checkpoint_path, 'historical source checkpoint')
    require_path(arguments.get('model'), target, 'historical dense model')
    return {'path': str(path), 'sha256': file_sha256(path), 'source_sha256': source_sha,
            'historical_e2e_progress': str(history_path), 'historical_progress_sha256': history_sha,
            'evidence_limit': 'Explicit migration provenance and local file hashes; original remote dense '
                              'weight shards were not verified bit-for-bit.'}


def validate_model_identity(source, cache, model_path, *, identity_manifest=None, source_path=None):
    """Keep strict paths by default; an optional JSON manifest binds one source file."""
    target = Path(model_path).resolve(strict=True)
    checkpoint_model = source.get('metadata', {}).get('model')
    metadata = cache.get('metadata', {})
    cache_model = cache.get('target_model', metadata.get('tokenizer_model', metadata.get('model_path')))
    if not checkpoint_model or not cache_model:
        raise ValueError('Source and calibration model identity evidence is required')
    manifest_audit = None
    if identity_manifest is not None:
        # A supplied manifest always binds the actual file, even when model
        # paths already match. It cannot authorize arbitrary derived sources.
        manifest_audit = _validate_identity_manifest(identity_manifest, source_path, checkpoint_model, target)
    elif Path(checkpoint_model).resolve() != target:
        raise ValueError('Source checkpoint model does not match requested dense model')
    # Token caches may name an alias directory. In that case both tokenizer
    # files must agree, rather than accepting equal vocabulary sizes alone.
    if Path(cache_model).resolve() != target:
        for name in ('tokenizer.json', 'tokenizer_config.json'):
            cached, requested = Path(cache_model) / name, target / name
            if not cached.is_file() or not requested.is_file() or file_sha256(cached) != file_sha256(requested):
                raise ValueError('Calibration cache tokenizer identity mismatch')
    audit = {'checkpoint_model': str(checkpoint_model), 'cache_tokenizer': str(cache_model),
             'model': str(target), 'cache_identity': cache.get('identity')}
    if manifest_audit is not None:
        audit.update(identity_manifest=manifest_audit, source_sha256=manifest_audit['source_sha256'])
    return audit


def validated_output_metadata(metadata, identity_audit, source_sha256):
    """Canonicalize only new output metadata and retain its original provenance."""
    updated = dict(metadata)
    updated.setdefault('original_model', metadata['model'])
    updated.update(model=identity_audit['model'], identity_audit=copy.deepcopy(identity_audit),
                   source_sha256=source_sha256)
    return updated


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


@torch.no_grad()
def optimize_passes(optimize, source, reference, hessian, *, passes=1, grouped=False, **kwargs):
    """Warm-start repeated local passes on identical inputs, with net reporting."""
    if not isinstance(passes, int) or passes <= 0:
        raise ValueError('passes must be a positive integer')
    state = copy.deepcopy(source)
    history = []
    for _ in range(passes):
        previous = state
        state, stats = optimize(state, reference, hessian, **kwargs)
        history.append(stats)
        scale_key = 'group_scale' if grouped else 'row_norm'
        if torch.equal(state['assignments'], previous['assignments']) and torch.equal(state[scale_key], previous[scale_key]):
            break
    summary = copy.deepcopy(history[-1])
    summary.update(objective_before=history[0]['objective_before'],
                   objective_after=history[-1]['objective_after'],
                   changed_assignments=int((state['assignments'].long() != source['assignments'].long()).sum()),
                   changed_scales=int((state[scale_key] != source[scale_key]).sum()),
                   passes_requested=passes, passes_completed=len(history), pass_stats=history)
    summary['true_decrease'] = summary['objective_before'] - summary['objective_after']
    if 'updated_rows' in summary:
        rows = int(source['shape'][0])
        row_changes = (state['assignments'].long() != source['assignments'].long()).reshape(rows, -1).any(-1)
        row_changes |= (state[scale_key] != source[scale_key]).reshape(rows, -1).any(-1)
        summary['updated_rows'] = int(row_changes.sum())
        summary['unchanged_rows'] = rows - summary['updated_rows']
    for prefix in ('deployed', 'ideal'):
        before, after = f'{prefix}_objective_before', f'{prefix}_objective_after'
        if before in summary:
            summary[before] = history[0][before]
            summary[after] = history[-1][after]
            summary[f'{prefix}_objective_decrease'] = summary[before] - summary[after]
    for alias, key in [('assignments_changed', 'changed_assignments'), ('scales_changed', 'changed_scales')]:
        if alias in summary:
            summary[alias] = summary[key]
    for key in ('seconds', 'elapsed_seconds', 'interval_count', 'intervals_evaluated', 'groups_selected',
                'groups_accepted', 'groups_no_update', 'candidate_positions_scanned', 'candidate_codes_scanned',
                'pool_candidates', 'proposal_evaluations', 'nonfinite_proposals', 'accepted_objective_increases'):
        if key in summary:
            summary[key] = sum(item[key] for item in history)
    for key in ('proposal_wins', 'winner_counts'):
        if key in summary:
            summary[key] = {name: sum(item[key].get(name, 0) for item in history) for name in summary[key]}
    return state, summary


def parse_diagnostic_layers(value):
    if value == '':
        return set()
    parts = value.split(',')
    if any(not re.fullmatch(r'\d+', item.strip()) for item in parts):
        raise ValueError('diagnostic-layers must be comma-separated nonnegative layer IDs')
    return {int(item.strip()) for item in parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--identity-manifest', help='Optional JSON binding a migrated source to this local model')
    parser.add_argument('--token-cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--run-stamp', default=time.strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--mode', choices=['interval', 'scale_only', 'fixed_scale'], required=True)
    parser.add_argument('--passes', type=int, default=1)
    parser.add_argument('--majorizer', choices=['block_frobenius', 'diagonal_heuristic'], default='block_frobenius')
    parser.add_argument('--diagnostic-layers', default='')
    parser.add_argument('--diagnostic-rows', type=int, default=4)
    parser.add_argument('--scale-layout', choices=['group', 'row'], default='group')
    parser.add_argument('--calibration-sequences', type=int, default=32)
    parser.add_argument('--context-length', type=int, default=2048)
    parser.add_argument('--calibration-start', type=int, default=0)
    parser.add_argument('--development-start', type=int, default=4096)
    parser.add_argument('--development-sequences', type=int, default=16)
    parser.add_argument('--active-positions', type=int, default=8)
    parser.add_argument('--candidates-per-ranking', type=int, default=4)
    parser.add_argument('--groups-per-row', type=int, default=1)
    parser.add_argument('--row-batch-size', type=int, default=32)
    parser.add_argument('--eta-relative', type=float, default=1e-6)
    parser.add_argument('--max-layers', type=int, default=0, help='Development-only prefix budget; zero means all layers')
    args = parser.parse_args()
    diagnostic_layers = parse_diagnostic_layers(args.diagnostic_layers)
    if args.diagnostic_rows <= 0:
        raise ValueError('diagnostic-rows must be positive')
    if args.passes <= 0:
        raise ValueError('passes must be a positive integer')
    if not re.fullmatch(r'\d{8}_\d{6}', args.run_stamp):
        raise ValueError('run-stamp must use YYYYMMDD_HHMMSS')
    source_path = Path(args.checkpoint).resolve(strict=True)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError('Use a fresh timestamped output directory')
    if args.max_layers < 0:
        raise ValueError('max-layers must be nonnegative')
    output_dir.mkdir(parents=True, exist_ok=False)
    progress_path = output_dir / f'{args.run_stamp}_progress.json'
    checkpoint_path = output_dir / f'{args.run_stamp}_checkpoint.pt'
    if checkpoint_path.resolve() == source_path:
        raise ValueError('Output aliases source checkpoint')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    started = time.time()
    source = torch.load(source_path, map_location='cpu', weights_only=False, mmap=True)
    cache = torch.load(args.token_cache, map_location='cpu', weights_only=False, mmap=True)
    tokenizer_audit = validate_model_identity(source, cache, args.model,
                                             identity_manifest=args.identity_manifest, source_path=source_path)
    source_sha = tokenizer_audit.get('source_sha256') or file_sha256(source_path)
    tokens, split_audit = select_calibration_tokens(
        cache, count=args.calibration_sequences, context_length=args.context_length,
        start=args.calibration_start, development_start=args.development_start,
        development_count=args.development_sequences)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, modeling_utils
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if getattr(modeling_utils, 'ALL_PARALLEL_STYLES', ()) is None:
        from experiments.qwen_qtip.qtip_runtime import install_tp_plan_compat
        install_tp_plan_compat(config)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                config=config, local_files_only=True,
                                                attn_implementation='sdpa')
    model.requires_grad_(False).eval()
    model.config.use_cache = False
    if int(tokens.min()) < 0 or int(tokens.max()) >= model.config.vocab_size:
        raise ValueError('Calibration token outside model vocabulary')
    tokenizer_audit['vocab_size'] = len(tokenizer)
    layer_count = len(model.model.layers)
    if any(layer >= layer_count for layer in diagnostic_layers):
        raise ValueError('diagnostic layer lies outside the model')
    if set(source['layers']) != {str(i) for i in range(layer_count)}:
        raise ValueError('Source checkpoint must cover every model layer')
    grouped = args.scale_layout == 'group'
    for states in source['layers'].values():
        if set(states) != set(LINEAR_NAMES):
            raise ValueError('Source must cover all seven Linears')
        if any(('group_scale' in state) != grouped for state in states.values()):
            raise ValueError('Requested scale layout does not match checkpoint')
        if any('residual_A' in state or 'residual_B' in state for state in states.values()):
            raise ValueError('V19 validation excludes residual-augmented checkpoints')
    batches = capture_first_layer_inputs(model, tokens, device)
    updated_layers = {index: dict(states) for index, states in source['layers'].items()}
    report = {'completed': False, 'method': 'v19_' + args.mode, 'configuration': vars(args),
              'source_checkpoint': str(source_path), 'source_sha256_before': source_sha,
              'calibration': split_audit, 'tokenizer': tokenizer_audit,
              'deployment_dtype': 'bfloat16', 'hessian_dtype': 'float64',
              'input_protocol': 'fixed local inputs from original quantized block on optimized prefix',
              'scale_semantics': 'existing group_scale; row_norm/col_norm/codebook fixed' if grouped
              else 'existing row_norm as whole-row scale; col_norm/codebook fixed (V19 specialization)',
              'uses_gumbel_softmax': False, 'uses_e2e': False, 'linears': [], 'mechanism_diagnostics': [],
              'surrogate_valid_majorizer': args.majorizer == 'block_frobenius'}
    atomic_json(progress_path, report)
    if grouped:
        from experiments.nowag_d6.v19_group_optimizer import optimize_groups as optimize
    else:
        from experiments.nowag_d6.v19_row_optimizer import optimize_rows as optimize
    for index in range(min(args.max_layers or layer_count, layer_count)):
        block = model.model.layers[index].to(device)
        modules = get_linear_modules(model, index)
        references = {name: module.weight.detach().cpu().clone() for name, module in modules.items()}
        for name, module in modules.items():
            module.weight.copy_(reconstruct_logical_state(source['layers'][str(index)][name], device).to(module.weight.dtype))
        hessians, counts = collect_block_hessians(block, modules, batches, device)
        for name in LINEAR_NAMES:
            if index in diagnostic_layers:
                from experiments.nowag_d6.v19_mechanism_probe import analyze_linear
                diagnostic = analyze_linear(source['layers'][str(index)][name], references[name].to(device),
                                            hessians[name], max_rows=args.diagnostic_rows,
                                            active_positions=3, candidates_per_ranking=2, seed=0)
                diagnostic_path = output_dir / f'{args.run_stamp}_layer{index}_{name.replace(".", "_")}_mechanism.json'
                atomic_json(diagnostic_path, diagnostic)
                report['mechanism_diagnostics'].append({'layer': index, 'name': name, 'path': str(diagnostic_path)})
            kwargs = {'active_positions': args.active_positions,
                      'candidates_per_ranking': args.candidates_per_ranking,
                      'row_batch_size': args.row_batch_size, 'eta_relative': args.eta_relative,
                      'deployment_dtype': torch.bfloat16, 'mode': args.mode, 'majorizer': args.majorizer}
            if grouped:
                kwargs['groups_per_row'] = args.groups_per_row
            state, stats = optimize_passes(optimize, source['layers'][str(index)][name], references[name].to(device),
                                           hessians[name], passes=args.passes, grouped=grouped, **kwargs)
            validate_unchanged_format(source['layers'][str(index)][name], state, grouped=grouped)
            updated_layers[str(index)][name] = move_tree(state, 'cpu')
            modules[name].weight.copy_(reconstruct_logical_state(state, device).to(modules[name].weight.dtype))
            report['linears'].append({'layer': index, 'name': name, 'calibration_tokens': counts[name], **stats})
            report['elapsed_seconds'] = time.time() - started
            report['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated(device)
            atomic_json(progress_path, report)
            print(json.dumps({'event': 'linear_complete', 'layer': index, 'name': name, 'stats': stats}), flush=True)
            del hessians[name], references[name]
        batches = propagate_block(block, batches, device)
        block.to('cpu')
        torch.cuda.empty_cache()
    result = dict(source)
    result['metadata'] = validated_output_metadata(source['metadata'], tokenizer_audit, source_sha)
    result['layers'] = updated_layers
    after_sha = file_sha256(source_path)
    if source_sha != after_sha:
        raise RuntimeError('Source checksum changed during optimization')
    report.update(completed=True, checkpoint=str(checkpoint_path), source_sha256_after=after_sha,
                  elapsed_seconds=time.time() - started, total_layers=layer_count,
                  optimized_layers=min(args.max_layers or layer_count, layer_count),
                  logical_bits=sum(logical_state_bits(state) for states in updated_layers.values() for state in states.values()))
    result['v19_validation'] = {key: value for key, value in report.items() if key != 'linears'}
    temporary = checkpoint_path.with_suffix('.pt.tmp')
    torch.save(result, temporary)
    temporary.replace(checkpoint_path)
    report.update(checkpoint_sha256=file_sha256(checkpoint_path), elapsed_seconds=time.time() - started)
    atomic_json(progress_path, report)
    print(json.dumps({'event': 'completed', 'checkpoint': str(checkpoint_path)}), flush=True)


if __name__ == '__main__':
    main()
