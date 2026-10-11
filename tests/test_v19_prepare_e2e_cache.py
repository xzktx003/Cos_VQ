import json
import hashlib

import pytest
import torch

from experiments.nowag_d6.v19_prepare_e2e_cache import prepare_cache


def test_cache_adapter_preserves_token_identity_and_e2e_schema(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'vocab_size': 10}))
    tokens = torch.arange(8).reshape(2, 4)
    source = {'target_model': str(tmp_path), 'input_ids': tokens, 'identity': 'fixture'}
    result = prepare_cache(source, tmp_path)
    assert torch.equal(result['finetune_tokens'], tokens)
    assert result['metadata']['tokenizer_vocab_size'] == 10
    assert result['metadata']['tokenizer_model'] == str(tmp_path)
    assert result['metadata']['source_indices'] == [0, 1]
    assert result['metadata']['region_selection'] is None
    result['finetune_tokens'][0, 0] = 9
    assert source['input_ids'][0, 0] == 0


def test_cache_adapter_rejects_wrong_identity_and_out_of_range_tokens(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'vocab_size': 2}))
    source = {'target_model': str(tmp_path), 'input_ids': torch.tensor([[0, 2]])}
    with pytest.raises(ValueError, match='vocabulary'):
        prepare_cache(source, tmp_path)
    source['input_ids'] = torch.tensor([[0, 1]])
    source['target_model'] = str(tmp_path / 'other')
    with pytest.raises(ValueError, match='identity'):
        prepare_cache(source, tmp_path)


def test_cache_adapter_concatenates_explicit_regions_and_records_provenance(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'vocab_size': 100}))
    tokens = torch.arange(32, dtype=torch.int32).reshape(8, 4)
    source = {'target_model': str(tmp_path), 'input_ids': tokens, 'identity': 'fixture'}
    result = prepare_cache(source, tmp_path, train_sequences=2,
                           validation_start=5, validation_sequences=2)
    expected = tokens[[0, 1, 5, 6]].long()
    assert torch.equal(result['finetune_tokens'], expected)
    metadata = result['metadata']
    assert metadata['source_indices'] == [0, 1, 5, 6]
    assert metadata['source_shape'] == [8, 4]
    assert metadata['region_selection'] == {
        'train_sequences': 2, 'validation_start': 5, 'validation_sequences': 2}
    assert metadata['tokens_sha256'] == hashlib.sha256(expected.numpy().tobytes()).hexdigest()
    assert metadata['source_tokens_sha256'] == hashlib.sha256(tokens.long().numpy().tobytes()).hexdigest()
    assert metadata['source_indices_sha256'] == hashlib.sha256(
        torch.tensor([0, 1, 5, 6], dtype=torch.int64).numpy().tobytes()).hexdigest()
    result['finetune_tokens'][0, 0] = 99
    assert source['input_ids'][0, 0] == 0


@pytest.mark.parametrize('arguments', [
    {'train_sequences': 2},
    {'validation_start': 5, 'validation_sequences': 2},
    {'train_sequences': 0, 'validation_start': 5, 'validation_sequences': 2},
    {'train_sequences': 2, 'validation_start': 1, 'validation_sequences': 2},
    {'train_sequences': 2, 'validation_start': 7, 'validation_sequences': 2},
    {'train_sequences': 2, 'validation_start': 5, 'validation_sequences': 0},
    {'train_sequences': 2.5, 'validation_start': 5, 'validation_sequences': 2},
])
def test_cache_adapter_rejects_incomplete_invalid_or_overlapping_regions(tmp_path, arguments):
    (tmp_path / 'config.json').write_text(json.dumps({'vocab_size': 100}))
    source = {'target_model': str(tmp_path), 'input_ids': torch.arange(32).reshape(8, 4)}
    with pytest.raises(ValueError, match='[Rr]egion'):
        prepare_cache(source, tmp_path, **arguments)


def test_cache_adapter_does_not_silently_remove_duplicate_region_rows(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'vocab_size': 10}))
    tokens = torch.tensor([[1, 2], [3, 4], [5, 6], [1, 2], [1, 2]])
    source = {'target_model': str(tmp_path), 'input_ids': tokens}
    result = prepare_cache(source, tmp_path, train_sequences=1,
                           validation_start=3, validation_sequences=2)
    assert result['metadata']['source_indices'] == [0, 3, 4]
    assert result['finetune_tokens'].tolist() == [[1, 2], [1, 2], [1, 2]]
