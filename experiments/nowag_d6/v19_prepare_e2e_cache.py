"""Adapt an identified local calibration cache without retokenizing its text."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.nowag_d6.v19_model_driver import file_sha256, validate_model_identity


def prepare_cache(cache, model_path, *, train_sequences=None,
                  validation_start=None, validation_sequences=None):
    model_path = Path(model_path).resolve(strict=True)
    identity = validate_model_identity({'metadata': {'model': str(model_path)}}, cache, model_path)
    tokens = cache.get('finetune_tokens', cache.get('input_ids'))
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or not tokens.numel():
        raise ValueError('Cache requires nonempty two-dimensional token IDs')
    if tokens.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError('Token IDs must use an integer dtype')
    vocab = int(json.loads((model_path / 'config.json').read_text())['vocab_size'])
    if int(tokens.min()) < 0 or int(tokens.max()) >= vocab:
        raise ValueError('Token IDs outside model vocabulary')
    region = None
    selection = (train_sequences, validation_start, validation_sequences)
    if any(value is not None for value in selection):
        if any(type(value) is not int for value in selection):
            raise ValueError('Region selection requires all three integer arguments')
        if (train_sequences <= 0 or validation_sequences <= 0
                or validation_start < train_sequences
                or validation_start + validation_sequences > len(tokens)):
            raise ValueError('Region selection must be nonempty, disjoint and within the source cache')
        indices = list(range(train_sequences)) + list(range(validation_start, validation_start + validation_sequences))
        region = {'train_sequences': train_sequences, 'validation_start': validation_start,
                  'validation_sequences': validation_sequences}
    else:
        indices = list(range(len(tokens)))
    source_tokens = tokens.to('cpu', torch.int64).contiguous()
    source_hash = hashlib.sha256(source_tokens.numpy().tobytes()).hexdigest()
    index_tensor = torch.tensor(indices, dtype=torch.int64)
    tokens = source_tokens.index_select(0, index_tensor)
    return {'finetune_tokens': tokens, 'metadata': {
        'tokenizer_model': str(model_path), 'tokenizer_vocab_size': vocab,
        'source_identity_audit': identity, 'source_metadata': cache.get('metadata', {}),
        'source_shape': list(source_tokens.shape), 'source_tokens_sha256': source_hash,
        'source_indices': indices,
        'source_indices_sha256': hashlib.sha256(index_tensor.numpy().tobytes()).hexdigest(),
        'region_selection': region, 'hash_encoding': 'contiguous CPU int64 bytes',
        'tokens_sha256': hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
        'retokenized': False, 'shape': list(tokens.shape)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--source-cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-sequences', type=int, help='Select this many rows from the source start')
    parser.add_argument('--validation-start', type=int, help='Original source row where validation begins')
    parser.add_argument('--validation-sequences', type=int, help='Append this many source validation rows')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Use a new timestamped cache path')
    source = torch.load(args.source_cache, map_location='cpu', weights_only=False, mmap=True)
    result = prepare_cache(source, args.model, train_sequences=args.train_sequences,
                           validation_start=args.validation_start,
                           validation_sequences=args.validation_sequences)
    result['metadata'].update(source_cache=str(Path(args.source_cache).resolve()),
                              source_sha256=file_sha256(args.source_cache))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.pt.tmp')
    torch.save(result, temporary)
    temporary.replace(output)
    print(json.dumps({'output': str(output), 'metadata': result['metadata']}, indent=2))


if __name__ == '__main__':
    main()
