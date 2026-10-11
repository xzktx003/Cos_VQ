from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.nowag_d6.calibrate_task_local_assignments import (
    BinaryLocalAssignmentParametrization,
    build_confidence_projection_counts,
    build_codebook_neighbor_table,
    choose_loss_dominant_snapshot,
    choose_projection_source_snapshot,
    choose_text_loss_dominant_snapshot,
    fold_hard_assignments,
    prefer_sparse_projection,
    prefer_task_loss_projection,
    partition_proposal_examples,
    project_snapshot_by_confidence,
    select_consensus_gradient_directed_alternative,
    select_gradient_directed_alternative,
    should_evaluate_projection_path,
)
from experiments.nowag_d6.probe_scale_conditioned_assignment_curvature import (
    RankedCandidates,
    SparseAssignmentApplicator,
    curvature_predictor_gate,
    rank_scale_conditioned_candidates,
)
from experiments.nowag_d6.probe_set_conditional_assignment_gain import (
    accept_conditional_bundle,
    score_batched_task_examples,
    select_cache_parity_indices,
    summarize_cached_task_scores,
    spearman_rank_correlation,
)


def _state() -> dict[str, torch.Tensor]:
    return {
        "assignments": torch.tensor([0, 2], dtype=torch.int32),
        "codebook": torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=torch.float16,
        ),
        "row_norm": torch.tensor([2.0], dtype=torch.float16),
        "col_norm": torch.tensor([3.0, 5.0, 7.0, 11.0], dtype=torch.float16),
        "shape": torch.tensor([1, 4], dtype=torch.int32),
        "vector_dim": torch.tensor(2, dtype=torch.int32),
        "assignment_bits": torch.tensor(2, dtype=torch.int32),
        "group_scale": torch.tensor([[0.5, 0.25]], dtype=torch.float16),
        "group_size": torch.tensor(2, dtype=torch.int32),
    }


def test_codebook_neighbor_table_keeps_each_anchor_first() -> None:
    codebook = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [4.0, 4.0]],
        dtype=torch.float32,
    )
    table = build_codebook_neighbor_table(codebook, candidates=3)
    assert table.shape == (4, 3)
    assert torch.equal(table[:, 0], torch.arange(4))
    assert len(set(table[0].tolist())) == 3
    assert table[0, 1].item() == 1


def test_gradient_directed_proposal_chooses_best_local_alternative() -> None:
    codebook = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    neighbor_table = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 2, 3], [2, 0, 1, 3], [3, 0, 1, 2]],
        dtype=torch.long,
    )
    assignments = torch.tensor([0, 2], dtype=torch.long)
    gradient_vectors = torch.tensor([[-2.0, 0.1], [0.0, 3.0]])
    vector_scales = torch.ones_like(gradient_vectors)
    alternatives, stats = select_gradient_directed_alternative(
        assignments=assignments,
        codebook=codebook,
        neighbor_table=neighbor_table,
        gradient_vectors=gradient_vectors,
        vector_scales=vector_scales,
        chunk_size=1,
    )
    assert alternatives.tolist() == [1, 0]
    assert stats["negative_first_order_rate"] == pytest.approx(1.0)


def test_proposal_examples_are_stratified_into_disjoint_gradient_views() -> None:
    task_examples = [
        {"task": task, "document_index": index}
        for task in ("task_a", "task_b")
        for index in range(4)
    ]
    text_examples = [{"row": index} for index in range(8)]
    views = partition_proposal_examples(
        task_examples,
        text_examples,
        shards=2,
        optimization_objective="task_adapted",
    )
    assert len(views) == 2
    for task_view, text_view in views:
        assert [item["task"] for item in task_view].count("task_a") == 2
        assert [item["task"] for item in task_view].count("task_b") == 2
        assert len(text_view) == 4
    assert {
        (item["task"], item["document_index"])
        for task_view, _text_view in views
        for item in task_view
    } == {(item["task"], item["document_index"]) for item in task_examples}
    assert {
        item["row"] for _task_view, text_view in views for item in text_view
    } == set(range(8))


def test_consensus_proposal_freezes_vectors_without_one_shared_descent_neighbor() -> (
    None
):
    codebook = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    neighbor_table = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 2, 3], [2, 0, 1, 3], [3, 0, 1, 2]],
        dtype=torch.long,
    )
    assignments = torch.tensor([0, 0], dtype=torch.long)
    gradient_views = [
        torch.tensor([[-2.0, 0.5], [-2.0, -2.0]]),
        torch.tensor([[-1.0, 0.2], [-2.0, 1.0]]),
        torch.tensor([[-3.0, 0.1], [1.0, -2.0]]),
        torch.tensor([[-0.5, 0.4], [1.0, 1.0]]),
    ]
    alternatives, stats = select_consensus_gradient_directed_alternative(
        assignments=assignments,
        codebook=codebook,
        neighbor_table=neighbor_table,
        gradient_vector_views=gradient_views,
        vector_scales=torch.ones(2, 2),
        chunk_size=2,
    )
    assert alternatives.tolist() == [1, 0]
    assert stats["gradient_views"] == 4
    assert stats["aggregate_negative_first_order_rate"] == pytest.approx(1.0)
    assert stats["strict_consensus_eligible_rate"] == pytest.approx(0.5)
    assert stats["frozen_no_consensus"] == 1


def test_scale_conditioned_curvature_changes_ranking_at_matched_candidates() -> None:
    codebook = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    neighbors = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 2, 3], [2, 0, 1, 3], [3, 0, 1, 2]],
        dtype=torch.long,
    )
    rankings = rank_scale_conditioned_candidates(
        assignments=torch.tensor([0, 0]),
        codebook=codebook,
        neighbor_table=neighbors,
        gradient_vector_views=[
            torch.tensor([[-2.0, 0.2], [-1.0, 0.2]]),
            torch.tensor([[-2.0, 0.1], [-1.0, 0.1]]),
        ],
        vector_scales=torch.ones(2, 2),
        activation_second_moment=torch.tensor([100.0, 100.0, 1.0, 1.0]),
        rows=1,
        columns=4,
        chunk_size=1,
        retain=1,
    )
    assert rankings["aggregate_first_order"].vector_indices.tolist() == [0]
    assert rankings["strict_consensus"].vector_indices.tolist() == [0]
    assert rankings["curvature_normalized"].vector_indices.tolist() == [1]
    assert rankings["curvature_normalized"].alternative_ids.tolist() == [1]
    assert rankings["curvature_normalized"].eligible_count == 2


def test_curvature_predictor_gate_requires_matched_wins_and_mean_gain() -> None:
    budgets = [1, 2, 3, 4]
    losses = {
        "aggregate_first_order": [1.00, 1.02, 1.04, 1.06],
        "strict_consensus": [0.99, 1.01, 1.03, 1.05],
        "curvature_normalized": [0.98, 1.00, 1.02, 1.06],
    }
    evaluations = [
        {
            "method": method,
            "budget_per_linear": budget,
            "validation": {"balanced_loss": loss},
        }
        for method, method_losses in losses.items()
        for budget, loss in zip(budgets, method_losses)
    ]
    gate = curvature_predictor_gate(
        evaluations, budgets=budgets, min_wins=3, tolerance=1e-7
    )
    assert gate["passed"]
    assert gate["controls"]["aggregate_first_order"]["wins"] == 3
    assert gate["controls"]["strict_consensus"]["wins"] == 3

    evaluations[-1]["validation"]["balanced_loss"] = 1.20
    assert not curvature_predictor_gate(
        evaluations, budgets=budgets, min_wins=3, tolerance=1e-7
    )["passed"]


def test_sparse_assignment_applicator_restores_exact_source_weight(monkeypatch) -> None:
    import experiments.nowag_d6.probe_scale_conditioned_assignment_curvature as probe

    linear = torch.nn.Linear(4, 1, bias=False)
    state = _state()
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[0.0, 0.0, 0.0, 5.5]]))
    source = linear.weight.detach().clone()
    model = torch.nn.Module()
    model.linear = linear
    monkeypatch.setattr(
        probe, "get_linear_modules", lambda _model, _layer: {"q_proj": linear}
    )
    applicator = SparseAssignmentApplicator(
        model,
        {"layers": {"0": {"q_proj": state}}},
        layers=[0],
        device=torch.device("cpu"),
    )
    selection = RankedCandidates(
        vector_indices=torch.tensor([0]),
        alternative_ids=torch.tensor([1], dtype=torch.int32),
        rank_scores=torch.tensor([-1.0]),
        mean_first_order=torch.tensor([-1.0]),
        worst_view_first_order=torch.tensor([-0.5]),
        curvature_cost=torch.tensor([2.0]),
        eligible_count=1,
    )
    applicator.apply({(0, "q_proj"): selection})
    assert torch.equal(linear.weight, torch.tensor([[3.0, 0.0, 0.0, 5.5]]))
    applicator.restore()
    assert torch.equal(linear.weight, source)


def test_proposal_gradient_collector_reuses_backward_inputs_for_curvature(
    monkeypatch,
) -> None:
    import experiments.nowag_d6.calibrate_task_local_assignments as calibration

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 1, bias=False)

    model = TinyModel()
    monkeypatch.setattr(calibration, "LINEAR_NAMES", ("q_proj",))
    monkeypatch.setattr(
        calibration,
        "get_linear_modules",
        lambda _model, _layer: {"q_proj": model.linear},
    )

    def train_one(
        local_model,
        example,
        *,
        device,
        teacher_kl_weight,
        teacher_temperature,
        loss_scale,
    ):
        del device, teacher_kl_weight, teacher_temperature
        (local_model.linear(example["x"]).sum() * loss_scale).backward()
        return {"combined_loss": 0.0}

    def text_loss(local_model, example, *, device):
        del device
        return local_model.linear(example["x"]).sum()

    monkeypatch.setattr(calibration, "_train_one_example", train_one)
    monkeypatch.setattr(calibration, "_text_guard_loss", text_loss)
    task_examples = [
        {"task": "task_a", "x": torch.tensor([[1.0, 2.0]])},
        {"task": "task_a", "x": torch.tensor([[3.0, 4.0]])},
    ]
    text_examples = [
        {"row": 0, "x": torch.tensor([[5.0, 6.0]])},
        {"row": 1, "x": torch.tensor([[7.0, 8.0]])},
    ]
    gradients, moments = calibration._proposal_gradients(
        model,
        task_examples,
        text_examples,
        layers=[0],
        device=torch.device("cpu"),
        teacher_kl_weight=0.25,
        teacher_temperature=1.0,
        text_guard_weight=1.0,
        optimization_objective="task_adapted",
        gradient_shards=2,
        capture_activation_second_moment=True,
    )
    assert len(gradients[(0, "q_proj")]) == 2
    assert moments is not None
    assert torch.allclose(moments[(0, "q_proj")], torch.tensor([21.0, 30.0]))


def test_binary_local_assignment_hard_anchor_and_switch_are_exact() -> None:
    state = _state()
    alternatives = torch.tensor([1, 3], dtype=torch.int32)
    parametrization = BinaryLocalAssignmentParametrization(
        state=state,
        alternative_assignments=alternatives,
        initial_logit_margin=0.25,
        expectation_chunk=1,
    )
    base = torch.zeros(1, 4, dtype=torch.float32)
    parametrization.set_hard_mode(True)
    anchor_weight = parametrization(base)
    expected_anchor = torch.tensor([[0.0, 0.0, 0.0, 5.5]])
    assert torch.equal(anchor_weight, expected_anchor)
    assert torch.equal(parametrization.hard_assignments().cpu(), state["assignments"])

    with torch.no_grad():
        parametrization.switch_logits[0] = 1.0
    switched = parametrization(base)
    expected_switched = torch.tensor([[3.0, 0.0, 0.0, 5.5]])
    assert torch.equal(switched, expected_switched)
    assert parametrization.hard_assignments().cpu().tolist() == [1, 2]


def test_binary_local_assignment_soft_path_backpropagates_to_logits() -> None:
    parametrization = BinaryLocalAssignmentParametrization(
        state=_state(),
        alternative_assignments=torch.tensor([1, 3]),
        initial_logit_margin=0.25,
        expectation_chunk=1,
    )
    parametrization.set_soft_schedule(temperature=0.7, kappa=2.0)
    loss = parametrization(torch.zeros(1, 4)).square().sum()
    loss.backward()
    assert parametrization.switch_logits.grad is not None
    assert torch.isfinite(parametrization.switch_logits.grad).all()
    assert torch.count_nonzero(parametrization.switch_logits.grad) > 0


def test_joint_parametrization_trains_scale_and_switch_then_folds_exactly() -> None:
    parametrization = BinaryLocalAssignmentParametrization(
        state=_state(),
        alternative_assignments=torch.tensor([1, 3]),
        initial_logit_margin=0.25,
        expectation_chunk=1,
        train_group_scale=True,
    )
    assert parametrization.log_group_scale_delta is not None
    parametrization.set_soft_schedule(temperature=0.7, kappa=2.0)
    loss = parametrization(torch.zeros(1, 4)).square().sum()
    loss.backward()
    assert torch.count_nonzero(parametrization.switch_logits.grad) > 0
    assert torch.count_nonzero(parametrization.log_group_scale_delta.grad) > 0

    with torch.no_grad():
        parametrization.switch_logits[0] = 1.0
        parametrization.log_group_scale_delta.copy_(torch.tensor([[0.1, -0.1]]))
    state = _state()
    parametrization.fold_deployment_state(state)
    assert state["assignments"].tolist() == [1, 2]
    assert state["group_scale"].dtype == torch.float16
    assert torch.count_nonzero(parametrization.log_group_scale_delta) == 0
    parametrization.set_hard_mode(True)
    expected = torch.tensor(
        [
            [
                float(state["group_scale"][0, 0]) * 6.0,
                0.0,
                0.0,
                float(state["group_scale"][0, 1]) * 22.0,
            ]
        ]
    )
    assert torch.equal(parametrization(torch.zeros(1, 4)), expected)


def test_fold_hard_assignments_changes_only_ids_and_preserves_rate_metadata() -> None:
    state = _state()
    before = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
    }
    fold_hard_assignments(state, torch.tensor([1, 2], dtype=torch.long))
    assert state["assignments"].dtype == torch.int32
    assert state["assignments"].tolist() == [1, 2]
    for key in set(before) - {"assignments"}:
        assert torch.equal(state[key], before[key])


def test_joint_projection_prefers_lower_task_loss_before_sparsity() -> None:
    assert prefer_task_loss_projection(
        incumbent_switches=0,
        incumbent_loss=0.70,
        candidate_switches=10,
        candidate_loss=0.68,
    )
    assert not prefer_task_loss_projection(
        incumbent_switches=0,
        incumbent_loss=0.70,
        candidate_switches=10,
        candidate_loss=0.71,
    )
    assert prefer_task_loss_projection(
        incumbent_switches=10,
        incumbent_loss=0.68,
        candidate_switches=5,
        candidate_loss=0.68,
    )


def _validation_metrics(
    *,
    balanced_loss: float = 1.0,
    macro_accuracy: float = 0.70,
    task_losses: tuple[float, float] = (1.0, 2.0),
    text_loss: float = 2.0,
) -> dict[str, object]:
    return {
        "balanced_loss": balanced_loss,
        "macro_accuracy": macro_accuracy,
        "per_task": {
            "task_a": {"loss": task_losses[0]},
            "task_b": {"loss": task_losses[1]},
        },
        "text_guard_cross_entropy": text_loss,
    }


def test_loss_dominant_selector_accepts_loss_gain_with_bounded_guards() -> None:
    reference = _validation_metrics()
    candidate = _validation_metrics(
        balanced_loss=0.98,
        macro_accuracy=0.691,
        task_losses=(0.99, 2.009),
        text_loss=2.009,
    )
    assert choose_loss_dominant_snapshot(
        reference,
        candidate,
        min_relative_loss_improvement=0.005,
        max_macro_regression=0.01,
        max_per_task_loss_regression=0.005,
        max_text_loss_regression=0.005,
    )


@pytest.mark.parametrize(
    "candidate",
    [
        _validation_metrics(balanced_loss=0.997),
        _validation_metrics(balanced_loss=0.98, macro_accuracy=0.689),
        _validation_metrics(balanced_loss=0.98, task_losses=(1.006, 1.9)),
        _validation_metrics(balanced_loss=0.98, text_loss=2.011),
    ],
    ids=[
        "insufficient-loss-gain",
        "macro-regression",
        "per-task-loss-regression",
        "text-loss-regression",
    ],
)
def test_loss_dominant_selector_rejects_failed_guard(
    candidate: dict[str, object],
) -> None:
    assert not choose_loss_dominant_snapshot(
        _validation_metrics(),
        candidate,
        min_relative_loss_improvement=0.005,
        max_macro_regression=0.01,
        max_per_task_loss_regression=0.005,
        max_text_loss_regression=0.005,
    )


def test_confidence_projection_counts_form_fixed_logarithmic_path() -> None:
    assert build_confidence_projection_counts(320, stages=4) == [40, 80, 160, 320]
    assert build_confidence_projection_counts(3, stages=4) == [1, 2, 3]
    assert build_confidence_projection_counts(0, stages=4) == []


def test_projection_path_is_not_rejected_by_a_bad_dense_endpoint() -> None:
    # Full can violate validation and rate guards while a prefix remains valid.
    assert should_evaluate_projection_path(full_switches=11_090_515)
    assert not should_evaluate_projection_path(full_switches=0)
    with pytest.raises(ValueError, match="non-negative"):
        should_evaluate_projection_path(full_switches=-1)


def test_confidence_projection_is_global_exact_and_nested() -> None:
    snapshot = {
        (0, "a"): torch.tensor([0.2, -0.4, 0.9]),
        (1, "b"): torch.tensor([0.7, 0.1, -0.3]),
    }
    projected_two = project_snapshot_by_confidence(snapshot, max_switches=2)
    projected_three = project_snapshot_by_confidence(snapshot, max_switches=3)
    assert sum(int((value > 0).sum()) for value in projected_two.values()) == 2
    assert sum(int((value > 0).sum()) for value in projected_three.values()) == 3
    assert projected_two[(0, "a")].tolist() == pytest.approx([-0.2, -0.4, 0.9])
    assert projected_two[(1, "b")].tolist() == pytest.approx([0.7, -0.1, -0.3])
    for key in snapshot:
        assert torch.logical_or(projected_two[key] <= 0, projected_three[key] > 0).all()


def test_projection_selection_prefers_sparsity_then_loss() -> None:
    assert prefer_sparse_projection(
        incumbent_switches=None,
        incumbent_loss=1.0,
        candidate_switches=80,
        candidate_loss=0.78,
    )
    assert prefer_sparse_projection(
        incumbent_switches=80,
        incumbent_loss=0.78,
        candidate_switches=40,
        candidate_loss=0.795,
    )
    assert not prefer_sparse_projection(
        incumbent_switches=40,
        incumbent_loss=0.795,
        candidate_switches=80,
        candidate_loss=0.78,
    )
    assert prefer_sparse_projection(
        incumbent_switches=40,
        incumbent_loss=0.795,
        candidate_switches=40,
        candidate_loss=0.79,
    )


def test_text_only_selector_requires_teacher_text_improvement() -> None:
    reference = {"text_guard_cross_entropy": 2.0}
    assert choose_text_loss_dominant_snapshot(
        reference,
        {"text_guard_cross_entropy": 1.98},
        min_relative_loss_improvement=0.005,
    )
    assert not choose_text_loss_dominant_snapshot(
        reference,
        {"text_guard_cross_entropy": 1.995},
        min_relative_loss_improvement=0.005,
    )
    assert not choose_text_loss_dominant_snapshot(
        reference,
        {"text_guard_cross_entropy": None},
        min_relative_loss_improvement=0.005,
    )


def test_projection_source_ignores_full_macro_but_keeps_loss_guards() -> None:
    reference = _validation_metrics()
    macro_regressed = _validation_metrics(
        balanced_loss=0.98,
        macro_accuracy=0.60,
        task_losses=(0.99, 1.99),
    )
    assert choose_projection_source_snapshot(
        reference,
        macro_regressed,
        min_relative_loss_improvement=0.005,
        max_per_task_loss_regression=0.005,
    )
    task_regressed = _validation_metrics(
        balanced_loss=0.98,
        macro_accuracy=0.70,
        task_losses=(1.006, 1.9),
    )
    assert not choose_projection_source_snapshot(
        reference,
        task_regressed,
        min_relative_loss_improvement=0.005,
        max_per_task_loss_regression=0.005,
    )


def test_set_conditional_bundle_accepts_only_a_strict_hard_objective_gain() -> None:
    decision = accept_conditional_bundle(
        incumbent_loss=1.0,
        candidate_losses={"q_proj": 1.01, "o_proj": 0.97, "down_proj": 0.99},
        tolerance=1e-7,
    )
    assert decision == {
        "accepted": True,
        "key": "o_proj",
        "loss": pytest.approx(0.97),
        "gain": pytest.approx(0.03),
    }
    rejected = accept_conditional_bundle(
        incumbent_loss=1.0,
        candidate_losses={"q_proj": 1.0, "o_proj": 1.01},
        tolerance=1e-7,
    )
    assert rejected["accepted"] is False
    assert rejected["key"] is None
    assert rejected["loss"] == pytest.approx(1.0)


def test_cached_task_score_summary_balances_tasks_and_includes_teacher_kl() -> None:
    examples = [
        {"task": "a", "gold": 0, "teacher_scores": torch.tensor([2.0, 0.0])},
        {"task": "a", "gold": 1, "teacher_scores": torch.tensor([0.0, 2.0])},
        {"task": "b", "gold": 0, "teacher_scores": torch.tensor([1.0, 0.0])},
    ]
    scores = [
        torch.tensor([2.0, 0.0]),
        torch.tensor([0.0, 2.0]),
        torch.tensor([0.5, 0.0]),
    ]
    summary = summarize_cached_task_scores(
        scores,
        examples,
        teacher_kl_weight=0.25,
        teacher_temperature=1.0,
    )
    loss_a = float(
        torch.nn.functional.cross_entropy(torch.tensor([[2.0, 0.0]]), torch.tensor([0]))
    )
    loss_b = float(
        torch.nn.functional.cross_entropy(torch.tensor([[0.5, 0.0]]), torch.tensor([0]))
    )
    assert summary["balanced_supervised_loss"] == pytest.approx((loss_a + loss_b) / 2)
    assert summary["balanced_teacher_kl"] > 0
    assert summary["balanced_combined_loss"] > summary["balanced_supervised_loss"]
    assert summary["macro_accuracy"] == pytest.approx(1.0)


def test_batched_task_scoring_preserves_choice_alignment() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def __call__(
            self, *, input_ids: torch.Tensor, use_cache: bool
        ) -> SimpleNamespace:
            assert use_cache is False
            self.batch_sizes.append(input_ids.shape[0])
            vocabulary = torch.arange(16, dtype=torch.float32).view(1, 1, -1)
            position = torch.arange(input_ids.shape[1], dtype=torch.float32).view(
                1, -1, 1
            )
            batch = torch.zeros(input_ids.shape[0], 1, 1)
            return SimpleNamespace(logits=batch + vocabulary + position)

    examples = [
        {
            "task": "a",
            "choices": [
                {"input_ids": [1, 2, 3, 4], "continuation_start": 2},
                {"input_ids": [1, 5, 6, 7, 8], "continuation_start": 3},
            ],
        },
        {
            "task": "b",
            "choices": [
                {"input_ids": [1, 9, 10, 11], "continuation_start": 2},
                {"input_ids": [1, 12, 13, 14], "continuation_start": 2},
            ],
        },
    ]
    model = FakeModel()

    scores = score_batched_task_examples(
        model, examples, batch_size=2, device=torch.device("cpu")
    )

    assert [row.shape for row in scores] == [torch.Size([2]), torch.Size([2])]
    assert sorted(model.batch_sizes) == [1, 1, 2]
    assert scores[0][0] != scores[0][1]
    assert scores[1][0] != scores[1][1]


def test_spearman_positive_control_handles_monotonic_and_constant_inputs() -> None:
    assert spearman_rank_correlation(
        [1.0, 2.0, 4.0], [10.0, 20.0, 40.0]
    ) == pytest.approx(1.0)
    assert spearman_rank_correlation(
        [1.0, 2.0, 4.0], [40.0, 20.0, 10.0]
    ) == pytest.approx(-1.0)
    assert spearman_rank_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


def test_cache_parity_slice_stratifies_tasks_and_sequence_shapes() -> None:
    examples = []
    for task in ("a", "b", "c"):
        for length in range(4, 14):
            examples.append(
                {
                    "task": task,
                    "choices": [
                        {
                            "input_ids": list(range(length)),
                            "continuation_start": length - 2,
                        }
                    ],
                }
            )
    indices = select_cache_parity_indices(examples, count=9)
    selected = [examples[index] for index in indices]
    assert len(indices) == 9
    assert {task: sum(row["task"] == task for row in selected) for task in "abc"} == {
        "a": 3,
        "b": 3,
        "c": 3,
    }
    lengths = [len(row["choices"][0]["input_ids"]) for row in selected]
    assert min(lengths) == 4
    assert max(lengths) == 13

