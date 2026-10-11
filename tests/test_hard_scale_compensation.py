import copy
from pathlib import Path

import pytest
import torch

from experiments.nowag_d6.hard_scale_compensation import (
    CompensationInfeasibleError,
    HardScaleCompensatedApplicator,
    apply_compensation_to_checkpoint,
    audit_fresh_compensated_checkpoint,
    compute_local_group_scale_compensation,
)
from experiments.nowag_d6.probe_text_constrained_assignment_gain import (
    choose_text_constrained_bundle,
)
from experiments.nowag_d6.probe_scale_conditioned_assignment_curvature import (
    RankedCandidates,
)
from experiments.nowag_d6.run_prefix_vector_gsq import reconstruct_logical_state


ROOT = Path(__file__).resolve().parents[1]


def _selection(index: int = 0, alternative: int = 2) -> RankedCandidates:
    scalar = torch.tensor([0.0])
    return RankedCandidates(
        vector_indices=torch.tensor([index]),
        alternative_ids=torch.tensor([alternative]),
        rank_scores=scalar,
        mean_first_order=scalar,
        worst_view_first_order=scalar,
        curvature_cost=torch.tensor([1.0]),
        eligible_count=1,
    )


def _state() -> dict[str, object]:
    return {
        "shape": [1, 8],
        "vector_dim": 2,
        "assignments": torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        "codebook": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float16
        ),
        "row_norm": torch.ones(1, dtype=torch.float16),
        "col_norm": torch.arange(1, 9, dtype=torch.float16),
        "group_size": 4,
        "group_scale": torch.tensor([[2.0, 3.0]], dtype=torch.float16),
    }


def test_closed_form_compensation_changes_only_touched_group_and_reduces_anchor_error() -> None:
    state = _state()
    compensation = compute_local_group_scale_compensation(state, _selection())

    assert compensation.group_rows.tolist() == [0]
    assert compensation.group_ids.tolist() == [0]
    assert compensation.source_scales.tolist() == [2.0]
    assert compensation.compensated_scales.item() == pytest.approx(
        torch.tensor(2.0 * 17.0 / 21.0, dtype=torch.float16).item()
    )
    assert compensation.weighted_error_after < compensation.weighted_error_before
    assert compensation.groups_changed == 1

    checkpoint = {"layers": {"0": {"linear": state}}}
    apply_compensation_to_checkpoint(
        checkpoint,
        {(0, "linear"): _selection()},
        {(0, "linear"): compensation},
    )
    saved = checkpoint["layers"]["0"]["linear"]
    assert saved["assignments"].tolist() == [2, 1, 0, 1]
    assert saved["group_scale"][0, 0] == compensation.compensated_scales[0]
    assert saved["group_scale"][0, 1].item() == 3.0
    assert saved["group_scale"].dtype == torch.float16


def test_compensation_rejects_state_without_deployable_group_scale() -> None:
    state = _state()
    del state["group_scale"]
    with pytest.raises(ValueError, match="group_scale"):
        compute_local_group_scale_compensation(state, _selection())


def test_non_positive_projection_is_candidate_infeasibility_not_corrupt_geometry() -> None:
    state = _state()
    state["codebook"] = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=torch.float16
    )
    with pytest.raises(CompensationInfeasibleError, match="no positive"):
        compute_local_group_scale_compensation(state, _selection(index=1))

    decision = choose_text_constrained_bundle(
        incumbent_task_loss=1.0,
        incumbent_text_loss=2.0,
        candidates={
            "bad_projection": {
                "task_loss": 1.0,
                "text_loss": None,
                "infeasible_reason": "scale_compensation_infeasible",
            },
            "valid": {
                "task_loss": 0.9,
                "text_loss": 1.99,
                "infeasible_reason": None,
            },
        },
        task_tolerance=0.0,
        text_tolerance=0.0,
    )
    assert decision["accepted"] is True
    assert decision["key"] == "valid"
    assert decision["rejections"]["bad_projection"] == "scale_compensation_infeasible"


def test_sparse_paired_action_matches_fresh_reconstruction_and_restores_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.nowag_d6.hard_scale_compensation as hard_scale

    state = _state()
    checkpoint = {"layers": {"0": {"linear": state}}}
    model = torch.nn.Module()
    model.linear = torch.nn.Linear(8, 1, bias=False)
    source_weight = reconstruct_logical_state(state, torch.device("cpu"))
    with torch.no_grad():
        model.linear.weight.copy_(source_weight)
    monkeypatch.setattr(
        hard_scale,
        "get_linear_modules",
        lambda _model, _layer: {"linear": _model.linear},
    )
    applicator = HardScaleCompensatedApplicator(
        model,
        checkpoint,
        layers=[0],
        device=torch.device("cpu"),
    )
    selection = _selection()
    applicator.apply({(0, "linear"): selection})

    deployed = copy.deepcopy(checkpoint)
    compensation = applicator.compensation_for((0, "linear"), selection)
    apply_compensation_to_checkpoint(
        deployed,
        {(0, "linear"): selection},
        {(0, "linear"): compensation},
    )
    expected = reconstruct_logical_state(
        deployed["layers"]["0"]["linear"], torch.device("cpu")
    )
    torch.testing.assert_close(model.linear.weight.float(), expected)

    contract = audit_fresh_compensated_checkpoint(
        checkpoint,
        deployed,
        model,
        {(0, "linear"): selection},
        device=torch.device("cpu"),
    )
    assert contract == {
        "assignments_changed": True,
        "assignment_changes": 1,
        "assignment_dtype": "int32",
        "group_scales_changed": True,
        "group_scale_changes": 1,
        "group_scale_dtype": "float16",
        "codebooks_changed": False,
        "normalizers_changed": False,
        "fixed_metadata_exact": True,
        "untouched_states_exact": True,
        "fresh_reconstruction_max_abs": 0.0,
        "fresh_reconstruction_exact": True,
        "logical_bitrate_changed": False,
    }

    applicator.restore()
    torch.testing.assert_close(model.linear.weight.float(), source_weight)

