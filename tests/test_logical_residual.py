import pytest
import torch

from experiments.nowag_d6.logical_state import reconstruct_logical_state


def residual_state():
    return {
        'shape': torch.tensor([2, 3]), 'vector_dim': torch.tensor(1),
        'assignment_bits': torch.tensor(1),
        'assignments': torch.zeros(6, dtype=torch.int32),
        'codebook': torch.tensor([[1.], [2.]], dtype=torch.float16),
        'row_norm': torch.tensor([2., 3.], dtype=torch.float16),
        'col_norm': torch.ones(3, dtype=torch.float16),
        'group_size': torch.tensor(3),
        'group_scale': torch.full((2, 1), 2., dtype=torch.float16),
        'residual_A': torch.tensor([[1., 2., 3.]], dtype=torch.float16),
        'residual_B': torch.tensor([[.5], [2.]], dtype=torch.float16),
    }


def test_residual_is_added_after_all_quantized_scales_in_fp32():
    state = residual_state()
    actual = reconstruct_logical_state(state, torch.device('cpu'))
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.tensor([[4.5, 5., 5.5], [8., 10., 12.]]))


@pytest.mark.parametrize('mutation', ['missing_a', 'missing_b', 'zero_rank', 'rank_mismatch',
                                     'columns', 'rows', 'ndim', 'dtype', 'nonfinite', 'not_tensor'])
def test_malformed_residual_is_rejected(mutation):
    state = residual_state()
    if mutation == 'missing_a':
        del state['residual_A']
    elif mutation == 'missing_b':
        del state['residual_B']
    elif mutation == 'zero_rank':
        state['residual_A'] = torch.empty(0, 3, dtype=torch.float16)
        state['residual_B'] = torch.empty(2, 0, dtype=torch.float16)
    elif mutation == 'rank_mismatch':
        state['residual_B'] = torch.ones(2, 2, dtype=torch.float16)
    elif mutation == 'columns':
        state['residual_A'] = torch.ones(1, 4, dtype=torch.float16)
    elif mutation == 'rows':
        state['residual_B'] = torch.ones(3, 1, dtype=torch.float16)
    elif mutation == 'ndim':
        state['residual_A'] = state['residual_A'].flatten()
    elif mutation == 'dtype':
        state['residual_B'] = state['residual_B'].float()
    elif mutation == 'nonfinite':
        state['residual_A'][0, 0] = float('nan')
    else:
        state['residual_A'] = [[1., 2., 3.]]
    with pytest.raises(ValueError, match='residual'):
        reconstruct_logical_state(state, torch.device('cpu'))


def test_residual_budget_counts_both_fp16_factors():
    from experiments.nowag_d6.logical_state import logical_state_bits, residual_metadata
    state = residual_state()
    # Six 1-bit indices + two codewords, five norms and two group scales.
    original_bits = 6 + (2 + 5 + 2) * 16
    assert logical_state_bits(state) == original_bits + 5 * 16
    assert residual_metadata(state) == {'residual_rank': 1, 'residual_parameters': 5, 'residual_bits': 80}
    del state['residual_A'], state['residual_B']
    assert logical_state_bits(state) == original_bits
    assert residual_metadata(state)['residual_bits'] == 0
