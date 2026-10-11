from __future__ import annotations

import pytest
import torch

from experiments.nowag_d6.interpolate_group_scale_checkpoints import (
    interpolate_checkpoint_group_scales,
    interpolate_group_scale,
)


def _checkpoint(scale: float) -> dict:
    return {
        "layers": {
            "0": {
                "self_attn.q_proj": {
                    "assignments": torch.tensor([1, 2], dtype=torch.int16),
                    "codebook": torch.tensor([[1.0], [2.0], [3.0]]),
                    "row_norm": torch.ones(1),
                    "col_norm": torch.ones(2),
                    "group_scale": torch.full((1, 2), scale, dtype=torch.float16),
                    "group_size": 1,
                    "shape": [1, 2],
                    "vector_dim": 1,
                    "assignment_bits": 2,
                }
            }
        }
    }


def test_group_scale_interpolation_is_geometric_and_endpoint_exact() -> None:
    base = torch.tensor([[1.0, 4.0]], dtype=torch.float16)
    tuned = torch.tensor([[4.0, 1.0]], dtype=torch.float16)
    assert torch.equal(interpolate_group_scale(base, tuned, 0.0), base)
    assert torch.equal(interpolate_group_scale(base, tuned, 1.0), tuned)
    midpoint = interpolate_group_scale(base, tuned, 0.5)
    assert torch.allclose(midpoint.float(), torch.tensor([[2.0, 2.0]]))


def test_checkpoint_interpolation_changes_only_group_scale() -> None:
    base = _checkpoint(1.0)
    tuned = _checkpoint(4.0)
    output, audit = interpolate_checkpoint_group_scales(base, tuned, alpha=0.5)
    state = output["layers"]["0"]["self_attn.q_proj"]
    assert torch.allclose(state["group_scale"].float(), torch.full((1, 2), 2.0))
    assert torch.equal(state["assignments"], base["layers"]["0"]["self_attn.q_proj"]["assignments"])
    assert audit["changed_group_scale_tensors"] == 1
    assert audit["fixed_tensor_checks"] == 4


def test_checkpoint_interpolation_rejects_fixed_state_mismatch() -> None:
    base = _checkpoint(1.0)
    tuned = _checkpoint(4.0)
    tuned["layers"]["0"]["self_attn.q_proj"]["assignments"][0] = 0
    with pytest.raises(ValueError, match="fixed tensor mismatch"):
        interpolate_checkpoint_group_scales(base, tuned, alpha=0.5)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
def test_checkpoint_interpolation_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        interpolate_checkpoint_group_scales(_checkpoint(1.0), _checkpoint(2.0), alpha=alpha)
