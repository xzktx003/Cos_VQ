from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from experiments.nowag_d6.v19_validation_score import heldout_tokens, next_token_nll, select_candidate


def test_family_selection_never_relabels_baseline_as_v19():
    from experiments.nowag_d6.v19_validation_score import select_family
    scores = [{'label': 'baseline', 'nll': 1.}, {'label': 'v1', 'nll': 2.},
              {'label': 'v2', 'nll': 1.5}]
    assert select_family(scores, ['v1', 'v2'])['label'] == 'v2'


def test_heldout_content_is_unique_and_excludes_training():
    cache = {'input_ids': torch.tensor([[1, 2, 3], [2, 3, 4], [1, 2, 3], [5, 6, 7], [5, 6, 7], [8, 9, 1]])}
    ids, audit = heldout_tokens(cache, start=2, count=4, context_length=3, exclude_count=2)
    assert ids.tolist() == [[5, 6, 7], [8, 9, 1]]
    assert audit['validation_sequences_after_filtering'] == 2
    assert audit['exact_train_validation_overlap_after_filtering'] == 0


def test_heldout_rejects_empty_independent_split():
    with pytest.raises(ValueError, match='independent'):
        heldout_tokens({'finetune_tokens': torch.ones((4, 3), dtype=torch.int64)},
                       start=2, count=2, context_length=3, exclude_count=2)


def test_nll_uses_shifted_full_tokens_with_chunk_independent_result():
    logits = torch.tensor([[[2., 1., 0.], [0., 3., 1.], [1., 1., 1.], [8., 0., 0.]]])
    tokens = torch.tensor([[0, 1, 2, 0]])

    class Model:
        def __call__(self, **kwargs):
            assert kwargs['use_cache'] is False
            return SimpleNamespace(logits=logits)

    expected = F.cross_entropy(logits[:, :-1].reshape(-1, 3), tokens[:, 1:].reshape(-1)).item()
    for chunk in (1, 2, 8):
        result = next_token_nll(Model(), tokens, torch.device('cpu'), chunk_size=chunk)
        assert result['nll'] == pytest.approx(expected, abs=1e-7)
        assert result['predicted_tokens'] == 3


def test_selection_uses_validation_and_preserves_baseline_on_ties():
    scores = [{'label': 'baseline', 'nll': 2.}, {'label': 'v19', 'nll': 2.}, {'label': 'bad', 'nll': 3.}]
    assert select_candidate(scores)['label'] == 'baseline'
    scores[1]['nll'] = 1.9
    assert select_candidate(scores)['label'] == 'v19'
    scores[1]['nll'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        select_candidate(scores)
