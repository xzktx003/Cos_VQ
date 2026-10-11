import pytest
import torch

from experiments.nowag_d6.hard_codebook import HardCodebookWeight
from experiments.nowag_d6.run_prefix_vector_gsq import (
    logical_state_usage,
    reconstruct_logical_state,
)


def _logical_state() -> dict[str, torch.Tensor]:
    return {
        "shape": torch.tensor([2, 5], dtype=torch.int32),
        "vector_dim": torch.tensor(2, dtype=torch.int32),
        "assignment_bits": torch.tensor(2, dtype=torch.int32),
        "assignments": torch.tensor([0, 1, 2, 2, 3, 1], dtype=torch.int32),
        "codebook": torch.tensor(
            [
                [1.0, -1.0],
                [0.5, 2.0],
                [-3.0, 4.0],
                [1.5, -0.25],
            ],
            dtype=torch.float16,
        ),
        "row_norm": torch.tensor([2.0, 0.5], dtype=torch.float16),
        "col_norm": torch.tensor([1.0, 2.0, 0.5, 4.0, 3.0], dtype=torch.float16),
        "group_size": torch.tensor(3, dtype=torch.int32),
        "group_scale": torch.tensor(
            [[1.0, 0.25], [2.0, 0.5]],
            dtype=torch.float16,
        ),
    }


def test_forward_matches_official_reconstruction_with_padding_and_group_scale() -> None:
    state = _logical_state()
    module = HardCodebookWeight(state)

    expected = reconstruct_logical_state(state, torch.device("cpu"))
    torch.testing.assert_close(module(), expected)


def test_trainable_params_receive_finite_nonzero_gradients() -> None:
    module = HardCodebookWeight(_logical_state())
    loss = module().square().sum()

    loss.backward()

    grads = [module.codebook.grad, module.row_norm_delta.grad, module.col_norm_delta.grad]
    assert all(grad is not None for grad in grads)
    for grad in grads:
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0
    assert module.group_scale.grad is None


def test_export_state_keeps_frozen_assignments_metadata_fp16_and_bpp() -> None:
    state = _logical_state()
    module = HardCodebookWeight(state)
    before_bits, before_weights = logical_state_usage(state)

    with torch.no_grad():
        module.codebook.add_(0.001)
        module.row_norm_delta.mul_(1.01)
        module.col_norm_delta.mul_(0.99)
    exported = module.export_state()

    assert set(exported) == set(state)
    assert torch.equal(exported["assignments"], state["assignments"])
    assert exported["assignments"].dtype == torch.int32
    assert torch.equal(exported["shape"], state["shape"])
    assert torch.equal(exported["vector_dim"], state["vector_dim"])
    assert torch.equal(exported["assignment_bits"], state["assignment_bits"])
    assert torch.equal(exported["group_size"], state["group_size"])
    assert torch.equal(exported["group_scale"], state["group_scale"])
    assert exported["codebook"].dtype == torch.float16
    assert exported["row_norm"].dtype == torch.float16
    assert exported["col_norm"].dtype == torch.float16
    assert logical_state_usage(exported) == (before_bits, before_weights)
    torch.testing.assert_close(
        module(),
        reconstruct_logical_state(exported, torch.device("cpu")),
    )


def test_tuning_flags_freeze_selected_parameter_groups() -> None:
    module = HardCodebookWeight(
        _logical_state(),
        tune_codebook=False,
        tune_norms=False,
    )

    assert module.codebook.requires_grad is False
    assert module.row_norm_delta.requires_grad is False
    assert module.col_norm_delta.requires_grad is False


def test_uint16_assignments_are_supported_and_preserved() -> None:
    state = _logical_state()
    state["assignments"] = state["assignments"].to(torch.uint16)

    module = HardCodebookWeight(state)
    exported = module.export_state()

    torch.testing.assert_close(
        module(),
        reconstruct_logical_state(state, torch.device("cpu")),
    )
    assert exported["assignments"].dtype == torch.uint16
    assert torch.equal(exported["assignments"], state["assignments"])


def test_rejects_nonfinite_serialized_values() -> None:
    state = _logical_state()
    state["codebook"] = state["codebook"].clone()
    state["codebook"][0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        HardCodebookWeight(state)
