import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.nowag_d6.v19_model_driver import (
    capture_first_layer_inputs,
    collect_block_hessians,
    file_sha256,
    select_calibration_tokens,
    validate_model_identity,
    validate_unchanged_format,
)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, hidden, position_ids=None, position_embeddings=None):
        return (self.proj(hidden),)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(20, 4)
        self.model.layers = nn.ModuleList([Block(), Block()])
        self.config = SimpleNamespace(use_cache=False)

    def forward(self, input_ids, use_cache=False):
        hidden = self.model.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
        for block in self.model.layers:
            hidden = block(hidden, position_ids=positions)[0]
        return hidden


def test_capture_restores_model_and_matches_real_first_layer_inputs():
    model = Model().eval()
    original = model.model.layers[0]
    tokens = torch.tensor([[1, 2, 3], [5, 6, 7]])
    batches = capture_first_layer_inputs(model, tokens, torch.device('cpu'))
    assert model.model.layers[0] is original
    assert len(batches) == 2
    for row, (hidden, kwargs) in zip(tokens, batches):
        torch.testing.assert_close(hidden, model.model.embed_tokens(row[None]))
        assert torch.equal(kwargs['position_ids'], torch.tensor([[0, 1, 2]]))


def test_hessian_is_full_covariance_and_hooks_are_removed():
    block = Block().eval()
    x = torch.tensor([[[1., 2., 0., 0.], [2., 4., 1., 0.]]])
    batches = [(x, {'position_ids': torch.tensor([[0, 1]])})]
    hessians, counts = collect_block_hessians(block, {'proj': block.proj}, batches, torch.device('cpu'))
    expected = x.reshape(-1, 4).double().T @ x.reshape(-1, 4).double() / 2
    torch.testing.assert_close(hessians['proj'], expected)
    assert counts == {'proj': 2}
    assert not block.proj._forward_pre_hooks


def test_capture_restores_original_layer_after_unexpected_error():
    model = Model()
    original = model.model.layers[0]
    with pytest.raises(IndexError):
        capture_first_layer_inputs(model, torch.tensor([[999]]), torch.device('cpu'))
    assert model.model.layers[0] is original


def test_token_selection_deduplicates_and_excludes_development_content():
    cache = {'input_ids': torch.tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]])}
    tokens, audit = select_calibration_tokens(cache, count=2, context_length=3, start=0, development_start=4, development_count=1)
    assert tokens.tolist() == [[4, 5, 6], [7, 8, 9]]
    assert audit['exact_development_overlap'] == 0


def test_token_selection_rejects_insufficient_unique_data():
    cache = {'finetune_tokens': torch.ones((3, 4), dtype=torch.int64)}
    with pytest.raises(ValueError, match='unique'):
        select_calibration_tokens(cache, count=2, context_length=4, start=0, development_start=2, development_count=1)


def test_identity_accepts_top_level_cache_target_and_rejects_wrong_dense_model(tmp_path):
    model_path = tmp_path / 'correct'
    model_path.mkdir()
    other = tmp_path / 'wrong'
    other.mkdir()
    source = {'metadata': {'model': str(model_path)}}
    cache = {'target_model': str(model_path)}
    assert validate_model_identity(source, cache, model_path)['checkpoint_model'] == str(model_path)
    with pytest.raises(ValueError, match='checkpoint model'):
        validate_model_identity(source, cache, other)
    with pytest.raises(ValueError, match='cache tokenizer'):
        validate_model_identity(source, {'target_model': str(other)}, model_path)


def test_identity_requires_evidence_for_source_and_cache(tmp_path):
    with pytest.raises(ValueError, match='identity'):
        validate_model_identity({}, {}, tmp_path)


@pytest.fixture
def identity_manifest_case(tmp_path):
    model = tmp_path / 'local_model'
    model.mkdir()
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        (model / name).write_text('{}\n')
    source = {'metadata': {'model': '/old_remote/models/model'}, 'layers': {}}
    checkpoint = tmp_path / 'original.pt'
    torch.save(source, checkpoint)
    progress = tmp_path / 'completed_progress.json'
    progress.write_text(json.dumps({
        'completed': True, 'source_metadata': source['metadata'],
        'arguments': {'checkpoint': str(checkpoint), 'model': str(model)},
    }))
    manifest = {
        'format': 'v19-model-identity-v1', 'source_checkpoint': str(checkpoint),
        'source_sha256': file_sha256(checkpoint),
        'checkpoint_metadata_model': source['metadata']['model'],
        'requested_model': str(model.resolve()),
        'model_files_sha256': {name: file_sha256(model / name) for name in
                               ('config.json', 'tokenizer.json', 'tokenizer_config.json')},
        'historical_e2e_progress': {'path': str(progress), 'sha256': file_sha256(progress)},
    }
    path = tmp_path / 'identity.json'
    path.write_text(json.dumps(manifest))
    return source, {'target_model': str(model)}, model, checkpoint, manifest, path


def test_identity_manifest_accepts_audited_remote_alias(identity_manifest_case):
    source, cache, model, checkpoint, manifest, path = identity_manifest_case
    before = copy.deepcopy(source)
    audit = validate_model_identity(source, cache, model, identity_manifest=path, source_path=checkpoint)
    assert audit['checkpoint_model'] == manifest['checkpoint_metadata_model']
    assert audit['source_sha256'] == manifest['source_sha256']
    assert audit['identity_manifest']['sha256'] == file_sha256(path)
    assert source == before
    with pytest.raises(ValueError, match='checkpoint model'):
        validate_model_identity(source, cache, model)


@pytest.mark.parametrize('field', [
    'source_sha256', 'checkpoint_metadata_model', 'requested_model', 'source_checkpoint',
    'config.json', 'tokenizer.json', 'tokenizer_config.json', 'historical_sha256',
])
def test_identity_manifest_rejects_each_wrong_binding(identity_manifest_case, field):
    source, cache, model, checkpoint, manifest, path = identity_manifest_case
    if field.endswith('.json'):
        manifest['model_files_sha256'][field] = '0' * 64
    elif field == 'historical_sha256':
        manifest['historical_e2e_progress']['sha256'] = '0' * 64
    elif field == 'source_sha256':
        manifest[field] = '0' * 64
    else:
        manifest[field] += '_wrong'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='[Ii]dentity manifest'):
        validate_model_identity(source, cache, model, identity_manifest=path, source_path=checkpoint)


@pytest.mark.parametrize('wrong_field', ['completed', 'checkpoint', 'model', 'source_model'])
def test_identity_manifest_requires_matching_completed_history(identity_manifest_case, wrong_field):
    source, cache, model, checkpoint, manifest, path = identity_manifest_case
    from pathlib import Path
    progress_path = Path(manifest['historical_e2e_progress']['path'])
    progress = json.loads(progress_path.read_text())
    if wrong_field == 'completed':
        progress['completed'] = False
    elif wrong_field == 'source_model':
        progress['source_metadata']['model'] += '_wrong'
    else:
        progress['arguments'][wrong_field] += '_wrong'
    progress_path.write_text(json.dumps(progress))
    manifest['historical_e2e_progress']['sha256'] = file_sha256(progress_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='[Ii]dentity manifest'):
        validate_model_identity(source, cache, model, identity_manifest=path, source_path=checkpoint)


def test_identity_manifest_requires_source_path_and_still_checks_cache(identity_manifest_case):
    source, cache, model, checkpoint, _manifest, path = identity_manifest_case
    with pytest.raises(ValueError, match='source_path'):
        validate_model_identity(source, cache, model, identity_manifest=path)
    cache['target_model'] = str(model.parent / 'different_tokenizer')
    with pytest.raises(ValueError, match='cache tokenizer'):
        validate_model_identity(source, cache, model, identity_manifest=path, source_path=checkpoint)


def test_identity_manifest_cannot_be_bypassed_by_matching_metadata(identity_manifest_case):
    source, cache, model, checkpoint, manifest, path = identity_manifest_case
    source['metadata']['model'] = str(model)
    torch.save(source, checkpoint)
    manifest['checkpoint_metadata_model'] = str(model)
    # Keep the old source SHA: matching metadata must not authorize a new file.
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='[Ii]dentity manifest'):
        validate_model_identity(source, cache, model, identity_manifest=path, source_path=checkpoint)


def test_identity_manifest_rejects_changed_model_files(identity_manifest_case):
    source, cache, model, checkpoint, _manifest, path = identity_manifest_case
    (model / 'config.json').write_text('{"hidden_size": 999}\n')
    with pytest.raises(ValueError, match='[Ii]dentity manifest'):
        validate_model_identity(source, cache, model, identity_manifest=path, source_path=checkpoint)


def test_output_metadata_records_validated_identity_without_mutating_source():
    from experiments.nowag_d6.v19_model_driver import validated_output_metadata
    metadata = {'model': '/old/model', 'stage': 'gsq'}
    audit = {'model': '/local/model', 'checkpoint_model': '/old/model'}
    result = validated_output_metadata(metadata, audit, 'a' * 64)
    assert result['model'] == '/local/model'
    assert result['original_model'] == '/old/model'
    assert result['source_sha256'] == 'a' * 64
    assert result['identity_audit'] == audit
    result['identity_audit']['model'] = 'changed'
    assert audit['model'] == '/local/model'
    assert metadata == {'model': '/old/model', 'stage': 'gsq'}


@pytest.mark.parametrize('key', ['codebook', 'col_norm', 'row_norm'])
def test_grouped_export_rejects_changing_fixed_metadata(key):
    source = {'assignments': torch.tensor([0], dtype=torch.int32),
              'codebook': torch.ones(1, 2, dtype=torch.float16),
              'row_norm': torch.ones(1, dtype=torch.float16),
              'col_norm': torch.ones(2, dtype=torch.float16),
              'shape': torch.tensor([1, 2]), 'vector_dim': torch.tensor(2),
              'assignment_bits': torch.tensor(1),
              'group_scale': torch.ones(1, 1, dtype=torch.float16), 'group_size': torch.tensor(2)}
    changed = copy.deepcopy(source)
    changed[key] *= 2
    with pytest.raises(ValueError, match='fixed'):
        validate_unchanged_format(source, changed, grouped=True)
