import pytest
import torch

from experiments.nowag_d6.finetune_hard_codebook import split_finetune_tokens, train_hard_block
from experiments.nowag_d6.finetune_hard_codebook import validate_baseline_ppl
from experiments.nowag_d6.finetune_hard_codebook import remove_training_overlap


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 0.0, 27.16])
def test_baseline_ppl_gate_rejects_invalid_or_changed_values(value):
    with pytest.raises(ValueError, match='Baseline environment discrepancy'):
        validate_baseline_ppl(value, 22.68)


def test_baseline_ppl_gate_accepts_matched_measurement():
    validate_baseline_ppl(22.681, 22.68)


def test_disjoint_calibration_split_rejects_short_cache():
    ids = torch.arange(24).reshape(6, 4)
    train, valid = split_finetune_tokens(ids, 4, 2, 4)
    assert torch.equal(train, ids[:4])
    assert torch.equal(valid, ids[4:])
    with pytest.raises(ValueError):
        split_finetune_tokens(ids, 5, 2, 4)


def test_int32_cache_is_promoted_for_cross_entropy_labels():
    ids = torch.arange(24, dtype=torch.int32).reshape(6, 4)
    train, valid = split_finetune_tokens(ids, 4, 2, 4)
    assert train.dtype == valid.dtype == torch.int64
    assert torch.equal(torch.cat((train, valid)), ids.long())


def test_validation_content_overlap_removed_despite_different_row_indices():
    train = torch.tensor([[1, 2, 3], [4, 5, 6]])
    valid = torch.tensor([[4, 5, 6], [7, 8, 9], [7, 8, 9]])
    filtered, audit = remove_training_overlap(train, valid)
    assert torch.equal(filtered, valid[1:2])
    assert audit['removed_training_overlap_rows'] == [0]
    assert audit['removed_validation_duplicate_rows'] == [2]
    assert audit['kept_validation_rows'] == [1]


def test_all_overlapping_validation_is_rejected():
    ids = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match='No independent validation'):
        remove_training_overlap(ids, ids)


def test_empty_block_training_is_rejected():
    x = torch.empty(0, 1, 1)
    with pytest.raises(ValueError, match='nonempty'):
        train_hard_block({}, x, x, x, x, lambda x, w: x,
                         device=torch.device('cpu'), epochs=1, batch_size=1,
                         learning_rate=0.01, seed=0)


def test_block_training_selects_serialized_hard_state_and_keeps_assignments():
    state = {
        'shape': torch.tensor([2, 2]), 'vector_dim': torch.tensor(2),
        'assignment_bits': torch.tensor(1),
        'assignments': torch.tensor([0, 1]),
        'codebook': torch.eye(2).half(),
        'row_norm': torch.ones(2).half(), 'col_norm': torch.ones(2).half(),
    }
    x = torch.eye(2).unsqueeze(0).repeat(4, 1, 1)
    y = x * 1.2
    def forward(inputs, weights):
        return torch.nn.functional.linear(inputs, weights['linear'])
    selected, stats = train_hard_block(
        {'linear': state}, x, y, x, y, forward,
        device=torch.device('cpu'), epochs=3, batch_size=2,
        learning_rate=0.03, seed=0,
    )
    assert stats['best_validation_mse'] < stats['initial_validation_mse']
    assert torch.equal(selected['linear']['assignments'], state['assignments'])
    assert selected['linear']['codebook'].dtype == torch.float16
    assert len(stats['epochs']) == 3


def test_validation_rejects_training_gain_that_hurts_heldout():
    state = {
        'shape': torch.tensor([1, 1]), 'vector_dim': torch.tensor(1),
        'assignment_bits': torch.tensor(1), 'assignments': torch.tensor([0]),
        'codebook': torch.ones(2, 1).half(),
        'row_norm': torch.ones(1).half(), 'col_norm': torch.ones(1).half(),
    }
    x = torch.ones(2, 1, 1)
    selected, stats = train_hard_block(
        {'linear': state}, x, x * 2, x, x,
        lambda inputs, weights: inputs * weights['linear'],
        device=torch.device('cpu'), epochs=2, batch_size=1,
        learning_rate=0.03, seed=0,
    )
    assert stats['selected_epoch'] == -1
    assert stats['best_validation_mse'] == 0
    assert torch.equal(selected['linear']['codebook'], state['codebook'])
