import pytest
import torch

from experiments.nowag_d6.binary_assignment import (
    BinaryAssignmentWeight,
    build_alternative_assignments,
)
from experiments.nowag_d6.hard_codebook import HardCodebookWeight


def state():
    return dict(
        shape=torch.tensor([1, 1]),
        vector_dim=torch.tensor(1),
        assignment_bits=torch.tensor(2),
        assignments=torch.tensor([0]),
        codebook=torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float16),
        row_norm=torch.ones(1, dtype=torch.float16),
        col_norm=torch.ones(1, dtype=torch.float16),
    )


def test_gate_switch_reduces_loss_and_codebook_gradient_is_selected_only():
    module = BinaryAssignmentWeight(state(), torch.tensor([1]))
    loss = (module() - 1).square().sum()
    loss.backward()
    assert module.assignment_logits.grad.item() < 0
    probability = torch.tensor(-0.1).sigmoid()
    torch.testing.assert_close(
        module.assignment_logits.grad, (-2 * probability * (1 - probability)).reshape(1)
    )
    torch.testing.assert_close(
        module.codebook.grad, torch.tensor([[-2.0], [0.0], [0.0]])
    )
    with torch.no_grad():
        module.assignment_logits -= module.assignment_logits.grad
    assert (module() - 1).square().sum().item() == 0
    assert module.export_state()["assignments"].item() == 1


def test_padding_group_uint16_export_has_identical_storage_and_forward():
    s = state()
    s.update(
        shape=torch.tensor([1, 3]),
        vector_dim=torch.tensor(2),
        assignments=torch.tensor([[0, 1]], dtype=torch.uint16),
        codebook=torch.tensor(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=torch.float16
        ),
        row_norm=torch.tensor([2.0], dtype=torch.float16),
        col_norm=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16),
        group_size=torch.tensor(2),
        group_scale=torch.tensor([[2.0, 4.0]], dtype=torch.float16),
    )
    module = BinaryAssignmentWeight(s, torch.tensor([[1, 2]], dtype=torch.uint16))
    with torch.no_grad():
        module.assignment_logits[0, 1] = 1
        module.codebook.add_(0.001)
    exported = module.export_state()
    assert exported.keys() == s.keys()
    assert exported["assignments"].dtype == torch.uint16
    assert exported["assignments"].shape == s["assignments"].shape
    assert sum(v.numel() * v.element_size() for v in exported.values()) == sum(
        v.numel() * v.element_size() for v in s.values()
    )
    assert torch.equal(module(), HardCodebookWeight(exported)())
    for dtype in (torch.float16, torch.bfloat16):
        assert torch.equal(module().to(dtype), HardCodebookWeight(exported)().to(dtype))
    assert torch.equal(s["assignments"].long(), torch.tensor([[0, 1]]))


@pytest.mark.parametrize(
    "alternative",
    [
        torch.tensor([0]),
        torch.tensor([3]),
        torch.tensor([-1]),
        torch.tensor([float("nan")]),
    ],
)
def test_invalid_alternatives_rejected(alternative):
    with pytest.raises(ValueError):
        BinaryAssignmentWeight(state(), alternative)


def test_nonfinite_logits_rejected_on_forward_and_export():
    module = BinaryAssignmentWeight(state(), torch.tensor([1]))
    with torch.no_grad():
        module.assignment_logits.fill_(float("nan"))
    with pytest.raises(ValueError):
        module()
    with pytest.raises(ValueError):
        module.export_state()


def test_candidate_search_unscales_dense_and_ignores_padding():
    s = state()
    s.update(
        shape=torch.tensor([1, 3]),
        vector_dim=torch.tensor(2),
        assignments=torch.tensor([[0, 0]], dtype=torch.uint16),
        codebook=torch.tensor(
            [[0.0, 0.0], [1.0, 100.0], [2.0, 0.0]], dtype=torch.float16
        ),
        row_norm=torch.tensor([2.0], dtype=torch.float16),
        col_norm=torch.tensor([3.0, 4.0, 5.0], dtype=torch.float16),
        group_size=torch.tensor(2),
        group_scale=torch.tensor([[2.0, 3.0]], dtype=torch.float16),
    )
    # Normalized vectors [2, 0] and [1, padding]: padding must not penalize code 1.
    result = build_alternative_assignments(
        s, torch.tensor([[24.0, 0.0, 30.0]]), neighbors=2, chunk_size=1
    )
    assert result.dtype == torch.uint16
    assert result.device.type == "cpu"
    assert torch.equal(result.long(), torch.tensor([[2, 1]]))


def test_candidate_search_respects_nearest_code_shortlist():
    s = state()
    result = build_alternative_assignments(s, torch.tensor([[2.0]]), neighbors=1)
    assert result.item() == 1


def test_selected_alternative_alone_receives_codebook_gradient():
    module = BinaryAssignmentWeight(state(), torch.tensor([1]))
    with torch.no_grad():
        module.assignment_logits.fill_(0.2)
    module().sum().backward()
    torch.testing.assert_close(
        module.codebook.grad, torch.tensor([[0.0], [1.0], [0.0]])
    )
    p = torch.tensor(0.2).sigmoid()
    torch.testing.assert_close(module.assignment_logits.grad, (p * (1 - p)).reshape(1))


def test_zero_gate_uses_original_and_zero_scale_is_supported():
    module = BinaryAssignmentWeight(state(), torch.tensor([1]))
    with torch.no_grad():
        module.assignment_logits.zero_()
    assert module.export_state()["assignments"].item() == 0
    s = state()
    s["row_norm"].zero_()
    result = build_alternative_assignments(s, torch.ones(1, 1))
    assert result.item() in (1, 2)


@pytest.mark.parametrize("small_scale", [0.0, 1e-7, 0.01])
def test_candidate_search_uses_actual_dense_weighted_error(small_scale):
    s = state()
    s.update(
        shape=torch.tensor([1, 2]),
        vector_dim=torch.tensor(2),
        codebook=torch.tensor(
            [[0.0, 0.0], [1.0, 3.0], [2.0, 1.0]], dtype=torch.float16
        ),
        row_norm=torch.tensor([2.0], dtype=torch.float16),
        col_norm=torch.tensor([1.0, small_scale], dtype=torch.float16),
        group_size=torch.tensor(1),
        group_scale=torch.tensor([[5.0, 2.0]], dtype=torch.float16),
    )
    # Unweighted normalized error prefers code 2 (5 vs 10). Actual dense
    # error prefers code 1 because the first coordinate has much greater scale.
    result = build_alternative_assignments(s, torch.zeros(1, 2), neighbors=2)
    assert result.item() == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_candidate_search_rejects_nonfinite_dense(value):
    with pytest.raises(ValueError):
        build_alternative_assignments(state(), torch.tensor([[value]]))
