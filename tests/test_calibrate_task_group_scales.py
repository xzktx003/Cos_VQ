from __future__ import annotations

import pytest
import torch

from experiments.nowag_d6.calibrate_task_group_scales import (
    GroupScaleDelta,
    choose_guarded_validation_snapshot,
    choose_loss_dominant_snapshot,
    choose_validation_snapshot,
    encode_loglikelihood_request,
    fold_group_scale_delta,
    gold_label,
    split_task_indices,
    topk_teacher_cross_entropy,
)


class _Tokenizer:
    bos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


def test_task_split_is_deterministic_disjoint_and_task_specific() -> None:
    first = split_task_indices(100, train_count=8, validation_count=4, audit_count=3, seed=7, task_name="arc_easy")
    second = split_task_indices(100, train_count=8, validation_count=4, audit_count=3, seed=7, task_name="arc_easy")
    other = split_task_indices(100, train_count=8, validation_count=4, audit_count=3, seed=7, task_name="piqa")
    assert first == second
    assert first != other
    assert len(first["train"]) == 8
    assert len(first["validation"]) == 4
    assert len(first["audit"]) == 3
    assert not (set(first["train"]) & set(first["validation"]))
    assert not (set(first["train"]) & set(first["audit"]))
    assert not (set(first["validation"]) & set(first["audit"]))


def test_task_split_offset_builds_disjoint_refinement_stages() -> None:
    first = split_task_indices(
        100,
        train_count=8,
        validation_count=4,
        audit_count=3,
        seed=7,
        task_name="arc_easy",
        offset=0,
    )
    second = split_task_indices(
        100,
        train_count=8,
        validation_count=4,
        audit_count=3,
        seed=7,
        task_name="arc_easy",
        offset=15,
    )
    first_indices = set().union(*map(set, first.values()))
    second_indices = set().union(*map(set, second.values()))
    assert len(first_indices) == 15
    assert len(second_indices) == 15
    assert first_indices.isdisjoint(second_indices)


def test_task_split_supports_a_separate_never_seen_final_audit_window() -> None:
    split = split_task_indices(
        30,
        train_count=8,
        validation_count=4,
        audit_count=3,
        seed=7,
        task_name="arc_easy",
        offset=0,
        audit_offset=20,
    )
    assert len(split["train"]) == 8
    assert len(split["validation"]) == 4
    assert len(split["audit"]) == 3
    assert not (set(split["train"]) & set(split["audit"]))
    assert not (set(split["validation"]) & set(split["audit"]))


def test_task_split_rejects_overlapping_or_overflowing_separate_audit() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        split_task_indices(
            30,
            train_count=8,
            validation_count=4,
            audit_count=3,
            seed=7,
            task_name="arc_easy",
            audit_offset=10,
        )
    with pytest.raises(ValueError, match="audit window"):
        split_task_indices(
            30,
            train_count=8,
            validation_count=4,
            audit_count=3,
            seed=7,
            task_name="arc_easy",
            audit_offset=28,
        )


def test_task_split_offset_rejects_negative_or_overflowing_window() -> None:
    with pytest.raises(ValueError, match="offset"):
        split_task_indices(
            20,
            train_count=8,
            validation_count=4,
            audit_count=3,
            seed=7,
            task_name="arc_easy",
            offset=-1,
        )
    with pytest.raises(ValueError, match="requested window"):
        split_task_indices(
            20,
            train_count=8,
            validation_count=4,
            audit_count=3,
            seed=7,
            task_name="arc_easy",
            offset=6,
        )


def test_gold_labels_cover_all_target_accuracy_tasks() -> None:
    assert gold_label("arc_easy", {"choices": {"label": ["A", "B"]}, "answerKey": "B"}) == 1
    assert gold_label("arc_challenge", {"choices": {"label": ["1", "2"]}, "answerKey": "1"}) == 0
    assert gold_label("hellaswag", {"gold": "3"}) == 3
    assert gold_label("piqa", {"label": 1}) == 1
    assert gold_label("winogrande", {"answer": "2"}) == 1


def test_loglikelihood_encoding_marks_only_continuation_and_left_truncates() -> None:
    encoded = encode_loglikelihood_request(_Tokenizer(), "ab", " cd", max_length=16)
    assert encoded["input_ids"] == [99, 97, 98, 32, 99, 100]
    assert encoded["continuation_start"] == 3

    truncated = encode_loglikelihood_request(_Tokenizer(), "abcdefgh", " XY", max_length=6)
    assert len(truncated["input_ids"]) == 6
    continuation = truncated["input_ids"][truncated["continuation_start"] :]
    assert continuation == [32, 88, 89]


def test_group_scale_delta_is_differentiable_and_folds_into_fp16_state() -> None:
    module = GroupScaleDelta(rows=2, in_features=5, group_size=3)
    weight = torch.ones(2, 5)
    with torch.no_grad():
        module.log_delta[0, 0] = torch.log(torch.tensor(2.0))
        module.log_delta[0, 1] = torch.log(torch.tensor(0.5))
    output = module(weight)
    assert torch.allclose(output[0], torch.tensor([2.0, 2.0, 2.0, 0.5, 0.5]))
    output.sum().backward()
    assert module.log_delta.grad is not None
    assert torch.count_nonzero(module.log_delta.grad).item() == 4

    state = {"group_scale": torch.ones(2, 2, dtype=torch.float16), "group_size": 3}
    folded = fold_group_scale_delta(state, module.log_delta.detach())
    assert folded.dtype == torch.float16
    assert torch.allclose(folded.float(), torch.tensor([[2.0, 0.5], [1.0, 1.0]]))


def test_validation_snapshot_prefers_macro_accuracy_then_lower_loss() -> None:
    best = {"macro_accuracy": 0.5, "balanced_loss": 0.9}
    assert choose_validation_snapshot(best, {"macro_accuracy": 0.6, "balanced_loss": 1.0})
    assert choose_validation_snapshot(best, {"macro_accuracy": 0.5, "balanced_loss": 0.8})
    assert not choose_validation_snapshot(best, {"macro_accuracy": 0.5, "balanced_loss": 1.0})
    assert not choose_validation_snapshot(best, {"macro_accuracy": 0.4, "balanced_loss": 0.1})


def test_guarded_snapshot_requires_text_teacher_non_regression() -> None:
    best = {"macro_accuracy": 0.5, "balanced_loss": 0.9}
    improved = {
        "macro_accuracy": 0.6,
        "balanced_loss": 0.8,
        "text_guard_cross_entropy": 1.0005,
    }
    assert choose_guarded_validation_snapshot(
        best,
        improved,
        reference_text_loss=1.0,
        max_relative_regression=0.001,
    )
    improved["text_guard_cross_entropy"] = 1.002
    assert not choose_guarded_validation_snapshot(
        best,
        improved,
        reference_text_loss=1.0,
        max_relative_regression=0.001,
    )


def test_loss_dominant_snapshot_matches_assignment_selection_contract() -> None:
    reference = {
        "balanced_loss": 1.0,
        "macro_accuracy": 0.70,
        "text_guard_cross_entropy": 2.0,
        "per_task": {
            "arc_easy": {"loss": 1.0},
            "piqa": {"loss": 0.8},
        },
    }
    candidate = {
        "balanced_loss": 0.99,
        "macro_accuracy": 0.695,
        "text_guard_cross_entropy": 2.005,
        "per_task": {
            "arc_easy": {"loss": 1.004},
            "piqa": {"loss": 0.79},
        },
    }
    assert choose_loss_dominant_snapshot(
        reference,
        candidate,
        min_relative_loss_improvement=0.005,
        max_macro_regression=0.01,
        max_per_task_loss_regression=0.005,
        max_text_loss_regression=0.005,
    )

    candidate["per_task"]["arc_easy"]["loss"] = 1.006
    assert not choose_loss_dominant_snapshot(
        reference,
        candidate,
        min_relative_loss_improvement=0.005,
        max_macro_regression=0.01,
        max_per_task_loss_regression=0.005,
        max_text_loss_regression=0.005,
    )


def test_topk_teacher_cross_entropy_uses_normalized_teacher_mass() -> None:
    student_logits = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]])
    teacher_indices = torch.tensor([[0, 1], [2, 1]])
    teacher_probabilities = torch.tensor([[0.75, 0.25], [0.75, 0.25]])
    value = topk_teacher_cross_entropy(
        student_logits, teacher_indices, teacher_probabilities
    )
    log_probabilities = student_logits.log_softmax(dim=-1)[0]
    expected = -torch.stack(
        [
            0.75 * log_probabilities[0, 0] + 0.25 * log_probabilities[0, 1],
            0.75 * log_probabilities[1, 2] + 0.25 * log_probabilities[1, 1],
        ]
    ).mean()
    assert torch.allclose(value, expected)
