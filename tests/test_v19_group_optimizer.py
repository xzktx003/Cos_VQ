"""CPU contracts for V19 optimization in an existing grouped format."""

import copy
import math

import pytest
import torch

from experiments.nowag_d6.logical_state import logical_state_bits, reconstruct_logical_state
from experiments.nowag_d6.v19_group_optimizer import _build_geometry, _scale_neighbours, optimize_groups


def make_state(columns=14, rows=2, vector_dim=6, group_size=8):
    return {
        "shape": (rows, columns),
        "vector_dim": vector_dim,
        "assignment_bits": 2,
        "group_size": group_size,
        "assignments": torch.zeros((rows, (columns + vector_dim - 1) // vector_dim), dtype=torch.int16),
        "codebook": torch.stack([
            torch.ones(vector_dim), -torch.ones(vector_dim),
            torch.linspace(0.5, 1.5, vector_dim), torch.zeros(vector_dim),
        ]).half(),
        "row_norm": torch.ones(rows, dtype=torch.float16),
        "col_norm": torch.ones(columns, dtype=torch.float16),
        "group_scale": torch.ones((rows, (columns + group_size - 1) // group_size), dtype=torch.float16),
        "metadata": {"unchanged": [1, 2, 3]},
    }


def decoded(state, dtype=torch.bfloat16):
    return reconstruct_logical_state(state, torch.device("cpu")).to(dtype).double()


def objective(state, reference, hessian, dtype=torch.bfloat16):
    error = decoded(state, dtype) - reference.double()
    return float(((error @ hessian.double()) * error).sum())


def test_cross_boundary_indices_frozen_in_both_groups_and_source_untouched():
    state = make_state()
    before = copy.deepcopy(state)
    target = copy.deepcopy(state)
    target["assignments"].fill_(1)
    reference = decoded(target)
    result, stats = optimize_groups(
        state, reference, torch.eye(14), groups_per_row=2,
        row_batch_size=1, active_positions=8,
    )
    # d6 code at columns 6:12 crosses the g8 boundary and must never move.
    assert torch.equal(result["assignments"][:, 1], before["assignments"][:, 1])
    assert bool((result["assignments"][:, (0, 2)] != 0).any())
    assert objective(result, reference, torch.eye(14)) < objective(state, reference, torch.eye(14))
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(state[key], value)
            assert result[key].dtype == value.dtype
            assert result[key].shape == value.shape
            if key not in {"assignments", "group_scale"}:
                assert torch.equal(result[key], value)
        else:
            assert state[key] == result[key] == value
    assert set(result) == set(state)
    assert logical_state_bits(result) == logical_state_bits(state)
    assert stats["assignments_changed"] > 0
    assert stats["groups_selected"] == 4
    assert stats["deployed_objective_after"] <= stats["deployed_objective_before"]


def coupling_problem():
    state = make_state(columns=4, rows=1, vector_dim=2, group_size=2)
    hessian = torch.eye(4, dtype=torch.float64)
    hessian[0, 2] = hessian[2, 0] = 0.8
    hessian[1, 3] = hessian[3, 1] = 0.8
    return state, torch.tensor([[3.0, 3.0, 0.0, 0.0]]), hessian


def test_scale_only_uses_full_outside_residual():
    state, reference, hessian = coupling_problem()
    result, stats = optimize_groups(
        state, reference, hessian, mode="scale_only", groups_per_row=1,
        deployment_dtype=torch.float32,
    )
    # The local target alone suggests 3. Full outside residual gives 3 - 0.8 = 2.2.
    assert abs(float(result["group_scale"][0, 0]) - 2.2) < 0.002
    assert float(result["group_scale"][0, 1]) == 1.0
    assert torch.equal(result["assignments"], state["assignments"])
    assert stats["groups_selected"] == 1


def test_second_group_uses_residual_after_first_acceptance():
    state, reference, hessian = coupling_problem()
    result, _ = optimize_groups(
        state, reference, hessian, mode="scale_only", groups_per_row=2,
        deployment_dtype=torch.float32,
    )
    first = float(result["group_scale"][0, 0])
    expected_second = 0.8 * (3.0 - first)
    assert abs(float(result["group_scale"][0, 1]) - expected_second) < 0.002
    assert objective(result, reference, hessian, torch.float32) < objective(state, reference, hessian, torch.float32)


def test_partial_tail_and_crossing_parts_participate_in_scale_fit():
    state = make_state(columns=5, rows=1, vector_dim=3, group_size=4)
    # The last code contains a padded coordinate that must not enter H or fitting.
    state["codebook"][0, 2] = 100.0
    target = copy.deepcopy(state)
    target["group_scale"][0, 1] = 2.0
    reference = decoded(target, torch.float32)
    result, _ = optimize_groups(
        state, reference, torch.eye(5), mode="scale_only", groups_per_row=1,
        deployment_dtype=torch.float32,
    )
    assert torch.equal(result["assignments"], state["assignments"])
    assert float(result["group_scale"][0, 1]) == 2.0
    assert objective(result, reference, torch.eye(5), torch.float32) == 0.0


def test_full_hessian_upper_bound_with_partial_support_blocks():
    generator = torch.Generator().manual_seed(93)
    x = torch.randn(14, 23, generator=generator, dtype=torch.float64)
    hessian = x @ x.T / x.shape[1]
    geometry = _build_geometry(hessian, 1, 8, 6, eta_relative=1e-6)
    gap = geometry.d_matrix - hessian[8:14, 8:14]
    assert float(torch.linalg.eigvalsh(gap).min()) >= -1e-10
    assert geometry.mutable == (False, True)


def test_bf16_deployment_objective_nonincrease_under_correlated_hessian():
    generator = torch.Generator().manual_seed(31)
    state = make_state(columns=17, rows=3, vector_dim=6, group_size=8)
    state["row_norm"] = torch.tensor([0.75, 1.2, 2.0], dtype=torch.float16)
    state["col_norm"] = torch.linspace(0.7, 1.3, 17).half()
    reference = torch.randn(3, 17, generator=generator)
    x = torch.randn(17, 29, generator=generator, dtype=torch.float64)
    hessian = x @ x.T / 29 + torch.eye(17) * 0.1
    result, stats = optimize_groups(
        state, reference, hessian, groups_per_row=3,
        active_positions=1, candidates_per_ranking=2, row_batch_size=2,
    )
    before = objective(state, reference, hessian)
    after = objective(result, reference, hessian)
    assert after <= before + 1e-10
    assert stats["deployed_objective_before"] == pytest.approx(before)
    assert stats["deployed_objective_after"] == pytest.approx(after)
    assert stats["accepted_objective_increases"] == 0
    assert stats["groups_selected"] == 9


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("inf")])
def test_invalid_anchor_scale_rejected_before_mutation(bad_scale):
    state = make_state()
    state["group_scale"][0, 0] = bad_scale
    with pytest.raises(ValueError, match="positive|finite"):
        optimize_groups(state, torch.ones(2, 14), torch.eye(14))


def test_ungrouped_initialization_rejected():
    state = make_state()
    del state["group_scale"]
    with pytest.raises(ValueError, match="group_scale"):
        optimize_groups(state, torch.ones(2, 14), torch.eye(14))


def test_fp16_neighbours_at_subnormal_and_maximum_boundaries():
    targets = torch.tensor([0.0, 2.0 ** -24, 0.0010002, 1.0, 65504.0, 1e8], dtype=torch.float64)
    nearest, below, above = _scale_neighbours(torch.ones_like(targets), targets, torch.ones_like(targets))
    expected = targets.clamp(2.0 ** -24, 65504).half()
    # CPU nextafter is the independent reference; the implementation must also
    # run on torch 2.4 CUDA, which does not implement nextafter for float16.
    expected_below = torch.nextafter(expected, torch.zeros_like(expected)).double().clamp_min(2.0 ** -24)
    expected_above = torch.nextafter(expected, torch.full_like(expected, math.inf)).double().clamp_max(65504)
    assert torch.equal(nearest, expected.double())
    assert torch.equal(below, expected_below)
    assert torch.equal(above, expected_above)
    assert all(bool(torch.isfinite(value).all() and (value > 0).all()) for value in (nearest, below, above))


def test_actual_g160_d6_crossings_and_final_short_group():
    state = make_state(columns=322, rows=1, vector_dim=6, group_size=160)
    target = copy.deepcopy(state)
    target["assignments"].fill_(1)
    result, stats = optimize_groups(
        state, decoded(target), torch.eye(322), groups_per_row=3,
        active_positions=8, candidates_per_ranking=4,
    )
    assert torch.equal(result["assignments"][0, [26, 53]], state["assignments"][0, [26, 53]])
    assert stats["globally_frozen_crossing_indices_per_row"] == 2
    assert stats["groups_selected"] == stats["cached_groups"] == 3
    assert stats["changed_assignments"] > 0
    assert stats["objective_after"] < stats["objective_before"]


def test_row_batch_size_does_not_change_independent_row_solution():
    state = make_state(rows=3)
    target = copy.deepcopy(state)
    target["assignments"][:, 0] = 1
    target["group_scale"] = torch.tensor([[0.5, 1.2], [0.75, 1.5], [1.25, 0.8]], dtype=torch.float16)
    reference = decoded(target)
    one, _ = optimize_groups(state, reference, torch.eye(14), groups_per_row=2, row_batch_size=1)
    batched, _ = optimize_groups(state, reference, torch.eye(14), groups_per_row=2, row_batch_size=3)
    assert torch.equal(one["assignments"], batched["assignments"])
    assert torch.equal(one["group_scale"], batched["group_scale"])


def test_insufficient_assignment_bits_rejected():
    state = make_state()
    state["assignment_bits"] = 1
    with pytest.raises(ValueError, match="bit budget"):
        optimize_groups(state, torch.ones(2, 14), torch.eye(14))
