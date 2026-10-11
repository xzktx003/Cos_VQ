import copy

import pytest
import torch
from torch import nn

from experiments.nowag_d6.hard_codebook import HardCodebookWeight
from experiments.nowag_d6.logical_state import logical_state_bits, reconstruct_logical_state
from experiments.nowag_d6.finetune_hard_codebook_e2e import HardCodebookLinear, validate_exported_state, validate_output_paths


def state():
    return {'shape': torch.tensor([2, 5]), 'vector_dim': torch.tensor(3),
            'assignment_bits': torch.tensor(1), 'assignments': torch.tensor([0, 1, 1, 0]),
            'codebook': torch.tensor([[.2, .3, .4], [-.5, .6, .7]]).half(),
            'row_norm': torch.ones(2).half(), 'col_norm': torch.ones(5).half(),
            'group_size': torch.tensor(2), 'group_scale': torch.tensor([[.5, 1., 2.], [1., .25, .5]]).half()}


def test_group_training_gradients_export_and_budget():
    original = state()
    layer = HardCodebookLinear(nn.Linear(5, 2, bias=False), original, tune_group_scales=True)
    before = layer.quantizer.export_state()
    layer(torch.ones(2, 5)).square().sum().backward()
    assert layer.quantizer.group_scale_delta.grad.abs().sum() > 0
    assert layer.quantizer.codebook.grad.abs().sum() > 0
    torch.optim.Adam(layer.parameters(), lr=.01).step()
    exported = layer.quantizer.export_state()
    assert not torch.equal(exported['group_scale'], before['group_scale'])
    validate_exported_state(exported, original, tune_group_scales=True)
    assert logical_state_bits(exported) == logical_state_bits(original)
    assert set(exported) == set(original)
    assert torch.equal(exported['assignments'], original['assignments'])
    assert torch.equal(layer.quantizer(), reconstruct_logical_state(exported, torch.device('cpu')))
    with pytest.raises(ValueError):
        validate_exported_state(exported, original)


def test_default_group_is_frozen_and_old_export_unchanged():
    original = state()
    module = HardCodebookWeight(original)
    assert 'group_scale_delta' not in dict(module.named_parameters())
    module().sum().backward()
    assert module.group_scale.grad is None
    assert torch.equal(module.export_state()['group_scale'], original['group_scale'])
    validate_exported_state(module.export_state(), original)


@pytest.mark.parametrize('bad', ['missing', 'zero', 'negative', 'nan', 'dtype'])
def test_group_training_rejects_invalid_source(bad):
    source = state()
    if bad == 'missing':
        del source['group_scale']
    elif bad == 'dtype':
        source['group_scale'] = source['group_scale'].float()
    else:
        source['group_scale'][0, 0] = {'zero': 0., 'negative': -1., 'nan': float('nan')}[bad]
    with pytest.raises(ValueError):
        HardCodebookWeight(source, tune_group_scales=True)


def test_positive_finite_fp16_projection_and_export_guard():
    original = state()
    module = HardCodebookWeight(original, tune_group_scales=True)
    with torch.no_grad():
        module.group_scale_delta[0].copy_(torch.tensor([-1., 1e8, 1.]))
    exported = module.export_state()
    assert exported['group_scale'].dtype == torch.float16
    assert exported['group_scale'][0, 0] == 2**-24
    assert exported['group_scale'][0, 1] == 65504
    validate_exported_state(exported, original, tune_group_scales=True)
    for key in ('group_scale', 'assignments', 'group_size'):
        bad = copy.deepcopy(exported)
        bad[key].reshape(-1)[0] = 0 if key == 'group_scale' else bad[key].reshape(-1)[0] + 1
        with pytest.raises(ValueError):
            validate_exported_state(bad, original, tune_group_scales=True)


def test_group_export_path_keeps_source(tmp_path):
    source = tmp_path / 'source_checkpoint.pt'
    source.write_bytes(b'checkpoint')
    with pytest.raises(ValueError):
        validate_output_paths(source, tmp_path, 'source')
    assert source.read_bytes() == b'checkpoint'


def test_checkpoint_recomputation_keeps_group_gradients():
    from torch.utils.checkpoint import checkpoint
    layer = HardCodebookLinear(nn.Linear(5, 2, bias=False), state(), tune_group_scales=True)
    cloned = copy.deepcopy(layer)
    inputs = torch.ones(2, 5)
    layer(inputs).square().sum().backward()
    checkpoint(cloned, inputs, use_reentrant=False).square().sum().backward()
    for (name, parameter), (other_name, other) in zip(layer.named_parameters(), cloned.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)


@pytest.mark.parametrize('extra', [['--binary-assignments'], ['--residual-rank', '2']])
def test_group_cli_rejects_index_or_budget_changing_modes(monkeypatch, extra):
    from experiments.nowag_d6.finetune_hard_codebook_e2e import main
    monkeypatch.setattr('sys.argv', ['train', '--model', 'unused', '--checkpoint', 'unused',
                                   '--token-cache', 'unused', '--output-dir', 'unused',
                                   '--tune-group-scales', *extra])
    with pytest.raises(ValueError, match='fixed assignments'):
        main()
