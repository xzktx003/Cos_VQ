import copy

import pytest
import torch

from experiments.nowag_d6.v19_row_optimizer import optimize_rows
from experiments.nowag_d6.v19_group_optimizer import optimize_groups
from experiments.nowag_d6.v19_model_driver import optimize_passes


def source(grouped):
    result = {'shape': (2, 6), 'vector_dim': 2, 'assignment_bits': 1,
              'assignments': torch.zeros(2, 3, dtype=torch.int16),
              'codebook': torch.tensor([[1., 1.], [-1., -1.]]).half(),
              'row_norm': torch.ones(2).half(), 'col_norm': torch.ones(6).half()}
    if grouped:
        result.update(group_size=4, group_scale=torch.ones(2, 2).half())
    return result


@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('majorizer', ['block_frobenius', 'diagonal_heuristic'])
def test_fixed_scale_and_majorizer_labels(grouped, majorizer):
    original = source(grouped)
    optimize = optimize_groups if grouped else optimize_rows
    kwargs = {'groups_per_row': 2} if grouped else {}
    result, stats = optimize(original, -torch.ones(2, 6), torch.eye(6).double(),
                             mode='fixed_scale', majorizer=majorizer, **kwargs)
    assert torch.equal(result['row_norm'], original['row_norm'])
    if grouped:
        assert torch.equal(result['group_scale'], original['group_scale'])
    assert stats['changed_assignments'] > 0
    assert stats['objective_after'] <= stats['objective_before']
    assert stats['surrogate_valid_majorizer'] == (majorizer == 'block_frobenius')
    assert stats['proposal_wins']['scale_only'] == 0
    assert stats['proposal_wins']['interval_refit'] == 0


def test_passes_summary_is_net_change_not_sum_and_source_is_preserved():
    original = source(False)
    saved = copy.deepcopy(original)
    seen = []

    def fake(state, reference, hessian, **kwargs):
        seen.append(int(state['assignments'][0, 0]))
        out = copy.deepcopy(state)
        out['assignments'][0, 0] = 1 - out['assignments'][0, 0]
        return out, {'objective_before': 10.-len(seen), 'objective_after': 9.-len(seen),
                     'changed_assignments': 1, 'changed_scales': 0, 'seconds': 1.,
                     'interval_count': 1, 'proposal_wins': {'interval': 1}}

    result, stats = optimize_passes(fake, original, None, None, passes=2, grouped=False)
    assert seen == [0, 1]
    assert torch.equal(original['assignments'], saved['assignments'])
    assert torch.equal(result['assignments'], original['assignments'])
    assert stats['changed_assignments'] == 0
    assert stats['objective_before'] == 9.
    assert stats['objective_after'] == 7.
    assert len(stats['pass_stats']) == 2


def test_passes_default_and_early_stop():
    original = source(False)
    kwargs = {'mode': 'scale_only'}
    reference, h = torch.ones(2, 6), torch.eye(6).double()
    first, direct = optimize_rows(original, reference, h, **kwargs)
    second, stats = optimize_passes(optimize_rows, original, reference, h, grouped=False, **kwargs)
    assert torch.equal(first['assignments'], second['assignments'])
    assert direct['objective_after'] == stats['objective_after']
    _, stopped = optimize_passes(optimize_rows, original, reference, h, grouped=False, passes=5, **kwargs)
    assert len(stopped['pass_stats']) == 1
    with pytest.raises(ValueError):
        optimize_passes(optimize_rows, original, reference, h, grouped=False, passes=0)


@pytest.mark.parametrize('grouped', [False, True])
def test_empirical_diagonal_correlated_objective_gate_and_repeated_passes(grouped):
    from experiments.nowag_d6.logical_state import reconstruct_logical_state
    original = source(grouped)
    torch.manual_seed(132)
    x = torch.randn(6, 8).double()
    h = x @ x.T
    reference = torch.randn(2, 6)
    optimize = optimize_groups if grouped else optimize_rows
    kwargs = {'groups_per_row': 2} if grouped else {}
    result, stats = optimize_passes(optimize, original, reference, h, passes=3, grouped=grouped,
                                    majorizer='diagonal_heuristic', **kwargs)
    def actual(state):
        error = reconstruct_logical_state(state, torch.device('cpu')).bfloat16().double()-reference.double()
        return float(((error@h)*error).sum())
    assert actual(result) <= actual(original) + 1e-10
    assert stats['objective_before'] == pytest.approx(actual(original))
    assert stats['objective_after'] == pytest.approx(actual(result))
    assert stats['changed_assignments'] == int((result['assignments'] != original['assignments']).sum())
    assert all(not item['surrogate_valid_majorizer'] for item in stats['pass_stats'])


@pytest.mark.parametrize('passes', ['0', '-1'])
def test_driver_invalid_passes_rejected_before_file_access(monkeypatch, passes):
    from experiments.nowag_d6.v19_model_driver import main
    monkeypatch.setattr('sys.argv', ['driver', '--model', 'missing', '--checkpoint', 'missing',
                                    '--token-cache', 'missing', '--output-dir', 'missing',
                                    '--mode', 'fixed_scale', '--passes', passes])
    with pytest.raises(ValueError, match='passes'):
        main()


@pytest.mark.parametrize('value', ['-1', 'a', '1,,2', ',1', '1,'])
def test_diagnostic_layer_parser_rejects_invalid_values(value):
    from experiments.nowag_d6.v19_model_driver import parse_diagnostic_layers
    with pytest.raises(ValueError):
        parse_diagnostic_layers(value)


def test_diagnostic_layer_parser_defaults_and_nonnegative_ids():
    from experiments.nowag_d6.v19_model_driver import parse_diagnostic_layers
    assert parse_diagnostic_layers('') == set()
    assert parse_diagnostic_layers('0, 5,10,5') == {0, 5, 10}


def test_diagnostic_rows_rejected_before_file_access(monkeypatch):
    from experiments.nowag_d6.v19_model_driver import main
    monkeypatch.setattr('sys.argv', ['driver', '--model', 'missing', '--checkpoint', 'missing',
                                    '--token-cache', 'missing', '--output-dir', 'missing',
                                    '--mode', 'interval', '--diagnostic-rows', '0'])
    with pytest.raises(ValueError, match='diagnostic-rows'):
        main()
