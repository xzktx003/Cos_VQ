"""Functionally calibrate anchor-local Vector-GSQ assignments.

This post-quantization stage keeps each serialized codebook, normalizer, and
group scale fixed.  A task-functional gradient first selects one promising
alternative from the geometric neighborhood of every deployed assignment.
A binary Concrete relaxation then learns whether each vector should retain its
anchor ID or switch to that alternative.  Formal benchmark validation/test
examples are never used for optimization or checkpoint selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.nowag_d6.calibrate_task_group_scales import (
    DEFAULT_TASKS,
    _atomic_json,
    _build_examples,
    _build_text_guard_examples,
    _evaluate_examples,
    _evaluate_text_guard,
    _teacher_scores,
    _text_guard_loss,
    _train_one_example,
    fold_group_scale_delta,
)
from experiments.nowag_d6.diagnose_block_vector_gsq import (
    LINEAR_NAMES,
    get_linear_modules,
)
from experiments.nowag_d6.run_prefix_vector_gsq import reconstruct_logical_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=int, nargs="+", default=[28, 29, 30, 31])
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument(
        "--optimization-objective",
        choices=("task_adapted", "text_teacher_only"),
        default="task_adapted",
        help=(
            "task_adapted uses labeled multiple-choice losses plus teacher/text "
            "guards; text_teacher_only excludes task examples from gradients, "
            "checkpoint selection, and the audit gate"
        ),
    )
    parser.add_argument("--train-per-task", type=int, default=64)
    parser.add_argument("--validation-per-task", type=int, default=64)
    parser.add_argument("--audit-per-task", type=int, default=64)
    parser.add_argument("--task-split-offset", type=int, default=0)
    parser.add_argument("--task-audit-offset", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument(
        "--proposal-gradient-shards",
        type=int,
        default=1,
        help=(
            "number of task-stratified, disjoint gradient views used to select "
            "an alternative; values above one require one neighbor to descend "
            "in every view"
        ),
    )
    parser.add_argument("--teacher-kl-weight", type=float, default=0.25)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--switch-penalty", type=float, default=1e-4)
    parser.add_argument("--max-switch-rate", type=float, default=0.05)
    parser.add_argument("--initial-logit-margin", type=float, default=0.25)
    parser.add_argument("--initial-temperature", type=float, default=1.0)
    parser.add_argument("--final-temperature", type=float, default=0.25)
    parser.add_argument("--initial-kappa", type=float, default=1.0)
    parser.add_argument("--final-kappa", type=float, default=4.0)
    parser.add_argument("--local-candidates", type=int, default=8)
    parser.add_argument("--candidate-chunk", type=int, default=65536)
    parser.add_argument("--expectation-chunk", type=int, default=65536)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--text-token-cache", required=True)
    parser.add_argument("--text-train-rows", type=int, nargs="+", required=True)
    parser.add_argument("--text-validation-rows", type=int, nargs="+", required=True)
    parser.add_argument("--text-audit-rows", type=int, nargs="+", required=True)
    parser.add_argument("--text-sequence-length", type=int, default=256)
    parser.add_argument("--text-teacher-topk", type=int, default=32)
    parser.add_argument("--text-guard-weight", type=float, default=0.25)
    parser.add_argument("--text-max-relative-regression", type=float, default=0.005)
    parser.add_argument(
        "--selection-min-relative-loss-improvement", type=float, default=0.005
    )
    parser.add_argument(
        "--selection-max-macro-regression", type=float, default=0.01
    )
    parser.add_argument(
        "--selection-max-per-task-loss-regression", type=float, default=0.005
    )
    parser.add_argument("--projection-stages", type=int, default=4)
    parser.add_argument("--train-group-scales", action="store_true")
    parser.add_argument("--scale-learning-rate", type=float, default=2e-3)
    parser.add_argument("--scale-l2-weight", type=float, default=1e-4)
    parser.add_argument("--max-log-scale-delta", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trusted-checkpoint", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def build_codebook_neighbor_table(
    codebook: torch.Tensor, *, candidates: int
) -> torch.Tensor:
    """Return geometric neighbors with each codeword itself in column zero."""
    if codebook.ndim != 2 or codebook.shape[0] < 2:
        raise ValueError("codebook must contain at least two vectors")
    if not 2 <= candidates <= codebook.shape[0]:
        raise ValueError("candidates must be in [2, codebook_size]")
    values = codebook.float()
    distances = torch.cdist(values, values).square()
    ids = distances.topk(candidates, dim=1, largest=False, sorted=True).indices
    anchors = torch.arange(codebook.shape[0], device=codebook.device)
    present = ids.eq(anchors[:, None])
    if not bool(present.any(dim=1).all()):
        raise RuntimeError("a codeword is absent from its own nearest-neighbor set")
    positions = present.float().argmax(dim=1)
    rows = torch.arange(ids.shape[0], device=ids.device)
    first = ids[:, 0].clone()
    ids[:, 0] = anchors
    ids[rows, positions] = first
    return ids


@torch.no_grad()
def select_gradient_directed_alternative(
    *,
    assignments: torch.Tensor,
    codebook: torch.Tensor,
    neighbor_table: torch.Tensor,
    gradient_vectors: torch.Tensor,
    vector_scales: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Choose the non-anchor neighbor with the best first-order loss delta."""
    if assignments.ndim != 1:
        raise ValueError("assignments must be flat")
    if gradient_vectors.shape != vector_scales.shape:
        raise ValueError("gradient vectors and vector scales must match")
    if gradient_vectors.shape != (assignments.numel(), codebook.shape[1]):
        raise ValueError("gradient geometry does not match assignments/codebook")
    if neighbor_table.shape[0] != codebook.shape[0] or neighbor_table.shape[1] < 2:
        raise ValueError("neighbor table has incompatible shape")
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")

    alternatives = torch.empty_like(assignments, dtype=torch.long)
    negative = 0
    score_sum = 0.0
    score_min = math.inf
    score_max = -math.inf
    for start in range(0, assignments.numel(), chunk_size):
        end = min(start + chunk_size, assignments.numel())
        anchors = assignments[start:end].long()
        candidate_ids = neighbor_table[anchors, 1:].long()
        anchor_vectors = codebook[anchors].float()
        candidate_vectors = codebook[candidate_ids].float()
        scaled_gradient = (
            gradient_vectors[start:end].float() * vector_scales[start:end].float()
        )
        scores = (
            scaled_gradient[:, None, :]
            * (candidate_vectors - anchor_vectors[:, None, :])
        ).sum(dim=2)
        best_scores, positions = scores.min(dim=1)
        rows = torch.arange(end - start, device=assignments.device)
        alternatives[start:end] = candidate_ids[rows, positions]
        negative += int((best_scores < 0).sum())
        score_sum += float(best_scores.sum())
        score_min = min(score_min, float(best_scores.min()))
        score_max = max(score_max, float(best_scores.max()))
    count = max(assignments.numel(), 1)
    return alternatives, {
        "negative_first_order_rate": negative / count,
        "mean_best_first_order_delta": score_sum / count,
        "min_best_first_order_delta": score_min,
        "max_best_first_order_delta": score_max,
    }


@torch.no_grad()
def select_consensus_gradient_directed_alternative(
    *,
    assignments: torch.Tensor,
    codebook: torch.Tensor,
    neighbor_table: torch.Tensor,
    gradient_vector_views: list[torch.Tensor],
    vector_scales: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select a neighbor whose first-order delta is negative in every view.

    Taking the minimum over several neighbors makes a negative score almost
    inevitable under a single noisy aggregate gradient.  This selector uses
    disjoint, task-stratified gradient views and minimizes the worst view only
    among neighbors that descend in every view.  Vectors without such a
    neighbor keep their deployed anchor ID and are therefore frozen exactly.
    """
    if assignments.ndim != 1:
        raise ValueError("assignments must be flat")
    if len(gradient_vector_views) < 2:
        raise ValueError("consensus selection requires at least two gradient views")
    expected_shape = (assignments.numel(), codebook.shape[1])
    if vector_scales.shape != expected_shape:
        raise ValueError("vector scales do not match assignments/codebook")
    if any(view.shape != expected_shape for view in gradient_vector_views):
        raise ValueError("gradient view geometry does not match assignments/codebook")
    if neighbor_table.shape[0] != codebook.shape[0] or neighbor_table.shape[1] < 2:
        raise ValueError("neighbor table has incompatible shape")
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")

    alternatives = assignments.clone().long()
    aggregate_negative = 0
    consensus_eligible = 0
    consensus_neighbor_sum = 0
    selected_mean_sum = 0.0
    selected_worst_sum = 0.0
    selected_worst_min = math.inf
    selected_worst_max = -math.inf
    for start in range(0, assignments.numel(), chunk_size):
        end = min(start + chunk_size, assignments.numel())
        anchors = assignments[start:end].long()
        candidate_ids = neighbor_table[anchors, 1:].long()
        anchor_vectors = codebook[anchors].float()
        candidate_delta = codebook[candidate_ids].float() - anchor_vectors[:, None, :]
        scales = vector_scales[start:end].float()
        view_scores = torch.stack(
            [
                (
                    view[start:end].float()[:, None, :]
                    * scales[:, None, :]
                    * candidate_delta
                ).sum(dim=2)
                for view in gradient_vector_views
            ],
            dim=0,
        )
        mean_scores = view_scores.mean(dim=0)
        aggregate_negative += int((mean_scores.min(dim=1).values < 0).sum())

        eligible_neighbors = (view_scores < 0).all(dim=0)
        eligible_counts = eligible_neighbors.sum(dim=1)
        has_consensus = eligible_counts > 0
        consensus_eligible += int(has_consensus.sum())
        consensus_neighbor_sum += int(eligible_counts.sum())
        worst_scores = view_scores.max(dim=0).values
        robust_scores = worst_scores.masked_fill(~eligible_neighbors, math.inf)
        selected_worst, positions = robust_scores.min(dim=1)
        rows = torch.arange(end - start, device=assignments.device)
        chunk_alternatives = anchors.clone()
        chunk_alternatives[has_consensus] = candidate_ids[
            rows[has_consensus], positions[has_consensus]
        ]
        alternatives[start:end] = chunk_alternatives
        if bool(has_consensus.any()):
            chosen_mean = mean_scores[rows[has_consensus], positions[has_consensus]]
            chosen_worst = selected_worst[has_consensus]
            selected_mean_sum += float(chosen_mean.sum())
            selected_worst_sum += float(chosen_worst.sum())
            selected_worst_min = min(selected_worst_min, float(chosen_worst.min()))
            selected_worst_max = max(selected_worst_max, float(chosen_worst.max()))

    count = max(assignments.numel(), 1)
    selected_count = consensus_eligible
    return alternatives, {
        "gradient_views": len(gradient_vector_views),
        "aggregate_negative_first_order_rate": aggregate_negative / count,
        "strict_consensus_eligible_rate": selected_count / count,
        "frozen_no_consensus": assignments.numel() - selected_count,
        "mean_consensus_neighbor_count": consensus_neighbor_sum / count,
        "mean_selected_first_order_delta": (
            selected_mean_sum / selected_count if selected_count else None
        ),
        "mean_selected_worst_view_delta": (
            selected_worst_sum / selected_count if selected_count else None
        ),
        "min_selected_worst_view_delta": (
            selected_worst_min if selected_count else None
        ),
        "max_selected_worst_view_delta": (
            selected_worst_max if selected_count else None
        ),
    }


def partition_proposal_examples(
    task_examples: list[dict[str, Any]],
    text_examples: list[dict[str, torch.Tensor | int]],
    *,
    shards: int,
    optimization_objective: str,
) -> list[tuple[list[dict[str, Any]], list[dict[str, torch.Tensor | int]]]]:
    """Create deterministic, disjoint gradient views balanced by task."""
    if shards <= 0:
        raise ValueError("proposal gradient shards must be positive")
    if optimization_objective not in {"task_adapted", "text_teacher_only"}:
        raise ValueError(f"unsupported optimization objective: {optimization_objective}")
    if len(text_examples) < shards:
        raise ValueError("text proposal examples must cover every gradient shard")

    task_views: list[list[dict[str, Any]]] = [[] for _ in range(shards)]
    if optimization_objective == "task_adapted":
        by_task: dict[str, list[dict[str, Any]]] = {}
        for example in task_examples:
            task = str(example.get("task"))
            by_task.setdefault(task, []).append(example)
        if not by_task or any(len(items) < shards for items in by_task.values()):
            raise ValueError("every task must cover every proposal gradient shard")
        for items in by_task.values():
            for position, example in enumerate(items):
                task_views[position % shards].append(example)

    text_views: list[list[dict[str, torch.Tensor | int]]] = [
        [] for _ in range(shards)
    ]
    for position, example in enumerate(text_examples):
        text_views[position % shards].append(example)
    return list(zip(task_views, text_views))


class _BinaryLocalExpectation(torch.autograd.Function):
    """Interpolate anchor/alternative codewords without retaining [N, d]."""

    @staticmethod
    def forward(
        ctx: Any,
        switch_probability: torch.Tensor,
        anchor_ids: torch.Tensor,
        alternative_ids: torch.Tensor,
        codebook: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        output = torch.empty(
            switch_probability.numel(),
            codebook.shape[1],
            device=switch_probability.device,
            dtype=switch_probability.dtype,
        )
        for start in range(0, switch_probability.numel(), chunk_size):
            end = min(start + chunk_size, switch_probability.numel())
            anchors = codebook[anchor_ids[start:end].long()].to(output.dtype)
            alternatives = codebook[alternative_ids[start:end].long()].to(output.dtype)
            probability = switch_probability[start:end, None]
            output[start:end] = anchors + probability * (alternatives - anchors)
        ctx.save_for_backward(anchor_ids, alternative_ids, codebook)
        ctx.chunk_size = int(chunk_size)
        return output

    @staticmethod
    def backward(ctx: Any, output_gradient: torch.Tensor) -> tuple[Any, ...]:
        anchor_ids, alternative_ids, codebook = ctx.saved_tensors
        probability_gradient = torch.empty(
            output_gradient.shape[0],
            device=output_gradient.device,
            dtype=output_gradient.dtype,
        )
        for start in range(0, output_gradient.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, output_gradient.shape[0])
            delta = (
                codebook[alternative_ids[start:end].long()]
                - codebook[anchor_ids[start:end].long()]
            ).to(output_gradient.dtype)
            probability_gradient[start:end] = (
                output_gradient[start:end] * delta
            ).sum(dim=1)
        return probability_gradient, None, None, None, None


class BinaryLocalAssignmentParametrization(nn.Module):
    """Anchor-versus-neighbor Concrete relaxation for one logical Linear."""

    def __init__(
        self,
        *,
        state: dict[str, Any],
        alternative_assignments: torch.Tensor,
        initial_logit_margin: float,
        expectation_chunk: int,
        train_group_scale: bool = False,
    ) -> None:
        super().__init__()
        rows, columns = (int(value) for value in state["shape"])
        vector_dim = int(state["vector_dim"])
        padded_columns = math.ceil(columns / vector_dim) * vector_dim
        anchors = state["assignments"].reshape(-1).long()
        alternatives = alternative_assignments.reshape(-1).long()
        if anchors.shape != alternatives.shape:
            raise ValueError("alternative assignments must match anchors")
        codebook = state["codebook"].float()
        if anchors.numel() != rows * (padded_columns // vector_dim):
            raise ValueError("logical assignments do not match weight shape")
        if min(int(anchors.min()), int(alternatives.min())) < 0:
            raise ValueError("assignment IDs must be non-negative")
        if max(int(anchors.max()), int(alternatives.max())) >= codebook.shape[0]:
            raise ValueError("assignment ID exceeds codebook")
        if initial_logit_margin <= 0 or expectation_chunk <= 0:
            raise ValueError("initial margin and expectation chunk must be positive")

        id_dtype = torch.int16 if codebook.shape[0] <= 32767 else torch.int32
        self.register_buffer("anchor_ids", anchors.to(id_dtype))
        self.register_buffer("alternative_ids", alternatives.to(id_dtype))
        self.register_buffer("codebook", codebook)
        self.register_buffer("row_norm", state["row_norm"].float())
        self.register_buffer("col_norm", state["col_norm"].float())
        if "group_scale" in state:
            self.register_buffer("group_scale", state["group_scale"].float())
            self.group_size = int(state["group_size"])
        else:
            self.group_scale = None
            self.group_size = 0
        if train_group_scale:
            if self.group_scale is None:
                raise ValueError("joint training requires a deployable group scale")
            self.log_group_scale_delta = nn.Parameter(
                torch.zeros_like(self.group_scale)
            )
        else:
            self.register_parameter("log_group_scale_delta", None)
        self.switch_logits = nn.Parameter(
            torch.full((anchors.numel(),), -float(initial_logit_margin))
        )
        self.rows = rows
        self.columns = columns
        self.padded_columns = padded_columns
        self.vector_dim = vector_dim
        self.expectation_chunk = int(expectation_chunk)
        self.hard_mode = True
        self.temperature = 1.0
        self.kappa = 1.0

    def set_hard_mode(self, enabled: bool) -> None:
        self.hard_mode = bool(enabled)

    def set_soft_schedule(self, *, temperature: float, kappa: float) -> None:
        if min(temperature, kappa) <= 0:
            raise ValueError("temperature and kappa must be positive")
        self.temperature = float(temperature)
        self.kappa = float(kappa)
        self.hard_mode = False

    def switch_probabilities(self) -> torch.Tensor:
        return torch.sigmoid(
            self.switch_logits.float() * self.kappa / self.temperature
        )

    @torch.no_grad()
    def hard_assignments(self) -> torch.Tensor:
        return torch.where(
            self.switch_logits > 0,
            self.alternative_ids,
            self.anchor_ids,
        ).to(torch.int32)

    def current_group_scale(self) -> torch.Tensor | None:
        if self.group_scale is None:
            return None
        if self.log_group_scale_delta is None:
            return self.group_scale
        return self.group_scale * self.log_group_scale_delta.float().exp()

    @torch.no_grad()
    def fold_deployment_state(self, state: dict[str, Any]) -> None:
        fold_hard_assignments(state, self.hard_assignments())
        if self.log_group_scale_delta is None:
            return
        folded = fold_group_scale_delta(state, self.log_group_scale_delta)
        state["group_scale"] = folded
        self.group_scale.copy_(folded.to(self.group_scale.device).float())
        self.log_group_scale_delta.zero_()

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        if self.hard_mode:
            probabilities = (self.switch_logits > 0).to(torch.float32)
        else:
            probabilities = self.switch_probabilities()
        vectors = _BinaryLocalExpectation.apply(
            probabilities,
            self.anchor_ids,
            self.alternative_ids,
            self.codebook,
            self.expectation_chunk,
        )
        normalized = vectors.reshape(self.rows, self.padded_columns)[:, : self.columns]
        weight = (
            normalized
            * self.row_norm[:, None]
            * self.col_norm[None, :]
        )
        group_scale = self.current_group_scale()
        if group_scale is not None:
            expanded = group_scale.repeat_interleave(
                self.group_size, dim=1
            )[:, : self.columns]
            weight = weight * expanded
        return weight.to(dtype=base_weight.dtype)


def fold_hard_assignments(
    state: dict[str, Any], assignments: torch.Tensor
) -> None:
    expected = state["assignments"].numel()
    flat = assignments.detach().reshape(-1).cpu().long()
    if flat.numel() != expected:
        raise ValueError("hard assignments do not match logical state")
    codebook_size = int(state["codebook"].shape[0])
    if int(flat.min()) < 0 or int(flat.max()) >= codebook_size:
        raise ValueError("hard assignment is outside the codebook")
    state["assignments"] = flat.to(torch.int32)


def choose_loss_dominant_snapshot(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    min_relative_loss_improvement: float,
    max_macro_regression: float,
    max_per_task_loss_regression: float,
    max_text_loss_regression: float,
) -> bool:
    """Accept a material loss gain only when every quality guard is bounded."""
    tolerances = (
        min_relative_loss_improvement,
        max_macro_regression,
        max_per_task_loss_regression,
        max_text_loss_regression,
    )
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in tolerances):
        raise ValueError("selection tolerances must be finite values in [0, 1]")

    required_scalars = (
        "balanced_loss",
        "macro_accuracy",
        "text_guard_cross_entropy",
    )
    try:
        reference_values = {key: float(reference[key]) for key in required_scalars}
        candidate_values = {key: float(candidate[key]) for key in required_scalars}
        reference_tasks = reference["per_task"]
        candidate_tasks = candidate["per_task"]
    except (KeyError, TypeError, ValueError):
        return False
    if not all(
        math.isfinite(value)
        for value in (*reference_values.values(), *candidate_values.values())
    ):
        return False
    if reference_values["balanced_loss"] <= 0:
        return False
    if reference_values["text_guard_cross_entropy"] <= 0:
        return False
    if not isinstance(reference_tasks, dict) or not isinstance(candidate_tasks, dict):
        return False
    if not reference_tasks or set(reference_tasks) != set(candidate_tasks):
        return False

    epsilon = 1e-12
    if candidate_values["balanced_loss"] > (
        reference_values["balanced_loss"]
        * (1.0 - min_relative_loss_improvement)
        + epsilon
    ):
        return False
    if candidate_values["macro_accuracy"] < (
        reference_values["macro_accuracy"] - max_macro_regression - epsilon
    ):
        return False
    if candidate_values["text_guard_cross_entropy"] > (
        reference_values["text_guard_cross_entropy"]
        * (1.0 + max_text_loss_regression)
        + epsilon
    ):
        return False

    for task_name, reference_task in reference_tasks.items():
        try:
            reference_loss = float(reference_task["loss"])
            candidate_loss = float(candidate_tasks[task_name]["loss"])
        except (KeyError, TypeError, ValueError):
            return False
        if not all(
            math.isfinite(value) and value > 0
            for value in (reference_loss, candidate_loss)
        ):
            return False
        if candidate_loss > (
            reference_loss * (1.0 + max_per_task_loss_regression) + epsilon
        ):
            return False
    return True


def choose_task_loss_dominant_snapshot(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    min_relative_loss_improvement: float,
    max_macro_regression: float,
    max_per_task_loss_regression: float,
) -> bool:
    """Apply the task side of the loss-dominance rule before text evaluation."""
    reference_with_text = dict(reference)
    candidate_with_text = dict(candidate)
    reference_with_text["text_guard_cross_entropy"] = 1.0
    candidate_with_text["text_guard_cross_entropy"] = 1.0
    return choose_loss_dominant_snapshot(
        reference_with_text,
        candidate_with_text,
        min_relative_loss_improvement=min_relative_loss_improvement,
        max_macro_regression=max_macro_regression,
        max_per_task_loss_regression=max_per_task_loss_regression,
        max_text_loss_regression=0.0,
    )


def choose_projection_source_snapshot(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    min_relative_loss_improvement: float,
    max_per_task_loss_regression: float,
) -> bool:
    """Retain logits with continuous task benefit before hard-state guards."""
    return choose_task_loss_dominant_snapshot(
        reference,
        candidate,
        min_relative_loss_improvement=min_relative_loss_improvement,
        max_macro_regression=1.0,
        max_per_task_loss_regression=max_per_task_loss_regression,
    )


def build_confidence_projection_counts(
    full_switches: int, *, stages: int
) -> list[int]:
    """Return a deterministic logarithmic hard path ending at the full state."""
    if full_switches < 0 or stages <= 0:
        raise ValueError("full switches must be non-negative and stages positive")
    if full_switches == 0:
        return []
    counts = [
        math.ceil(full_switches / (2**power))
        for power in range(stages - 1, -1, -1)
    ]
    return sorted(set(min(full_switches, count) for count in counts))


def should_evaluate_projection_path(*, full_switches: int) -> bool:
    """Inspect sparse prefixes even when the dense hard endpoint is poor.

    Confidence projection is explicitly a non-monotone deployment path. A
    dense endpoint therefore cannot soundly admit or reject its sparse
    prefixes; validation, text, and rate guards are applied separately to
    every projected state below.
    """
    if full_switches < 0:
        raise ValueError("full switch count must be non-negative")
    return full_switches > 0


def project_snapshot_by_confidence(
    snapshot: dict[tuple[int, str], torch.Tensor], *, max_switches: int
) -> dict[tuple[int, str], torch.Tensor]:
    """Keep the globally highest-confidence positive logits as a nested hard state."""
    if max_switches < 0:
        raise ValueError("max switches must be non-negative")
    positive_indices: dict[tuple[int, str], torch.Tensor] = {}
    positive_values: list[torch.Tensor] = []
    for key, values in snapshot.items():
        if values.ndim != 1 or not bool(torch.isfinite(values).all()):
            raise ValueError("snapshot logits must be finite flat tensors")
        indices = torch.nonzero(values > 0, as_tuple=False).flatten()
        positive_indices[key] = indices
        positive_values.append(values[indices].float())

    total_positive = sum(values.numel() for values in positive_values)
    keep = min(max_switches, total_positive)
    projected = {
        key: -values.abs().clamp_min(1e-6)
        for key, values in snapshot.items()
    }
    if keep == 0:
        return projected

    flattened = torch.cat(positive_values)
    ranking = torch.argsort(flattened, descending=True, stable=True)[:keep]
    offset = 0
    for key, values in snapshot.items():
        length = positive_indices[key].numel()
        local = ranking[(ranking >= offset) & (ranking < offset + length)] - offset
        if local.numel():
            original = positive_indices[key][local]
            projected[key][original] = values[original].abs().clamp_min(1e-6)
        offset += length
    return projected


def prefer_sparse_projection(
    *,
    incumbent_switches: int | None,
    incumbent_loss: float,
    candidate_switches: int,
    candidate_loss: float,
) -> bool:
    """Prefer minimum deployment disturbance, using task loss only as a tie-break."""
    if candidate_switches < 0:
        raise ValueError("candidate switches must be non-negative")
    if not all(math.isfinite(value) for value in (incumbent_loss, candidate_loss)):
        raise ValueError("projection losses must be finite")
    if incumbent_switches is None:
        return True
    if incumbent_switches < 0:
        raise ValueError("incumbent switches must be non-negative")
    return candidate_switches < incumbent_switches or (
        candidate_switches == incumbent_switches
        and candidate_loss < incumbent_loss - 1e-12
    )


def prefer_task_loss_projection(
    *,
    incumbent_switches: int | None,
    incumbent_loss: float,
    candidate_switches: int,
    candidate_loss: float,
) -> bool:
    """For joint training, minimize task loss and use sparsity only as a tie-break."""
    if candidate_switches < 0:
        raise ValueError("candidate switches must be non-negative")
    if not all(math.isfinite(value) for value in (incumbent_loss, candidate_loss)):
        raise ValueError("projection losses must be finite")
    if incumbent_switches is None:
        return True
    if incumbent_switches < 0:
        raise ValueError("incumbent switches must be non-negative")
    return candidate_loss < incumbent_loss - 1e-12 or (
        abs(candidate_loss - incumbent_loss) <= 1e-12
        and candidate_switches < incumbent_switches
    )


def choose_text_loss_dominant_snapshot(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    min_relative_loss_improvement: float,
) -> bool:
    """Select a checkpoint using only held-out teacher-text cross-entropy."""
    if not math.isfinite(min_relative_loss_improvement) or not (
        0 <= min_relative_loss_improvement < 1
    ):
        raise ValueError("text loss improvement must be finite and in [0, 1)")
    reference_loss = reference.get("text_guard_cross_entropy")
    candidate_loss = candidate.get("text_guard_cross_entropy")
    if not isinstance(reference_loss, (int, float)) or not isinstance(
        candidate_loss, (int, float)
    ):
        return False
    if not all(math.isfinite(value) and value > 0 for value in (
        float(reference_loss),
        float(candidate_loss),
    )):
        return False
    return float(candidate_loss) <= float(reference_loss) * (
        1.0 - min_relative_loss_improvement
    )


def _validate_args(args: argparse.Namespace) -> None:
    positive_ints = (
        args.train_per_task,
        args.validation_per_task,
        args.audit_per_task,
        args.epochs,
        args.gradient_accumulation,
        args.proposal_gradient_shards,
        args.local_candidates,
        args.candidate_chunk,
        args.expectation_chunk,
        args.max_length,
        args.text_sequence_length,
        args.text_teacher_topk,
        args.projection_stages,
    )
    if min(positive_ints) <= 0:
        raise ValueError("counts, chunks, epochs, and lengths must be positive")
    positive_floats = (
        args.learning_rate,
        args.scale_learning_rate,
        args.max_log_scale_delta,
        args.teacher_temperature,
        args.initial_logit_margin,
        args.initial_temperature,
        args.final_temperature,
        args.initial_kappa,
        args.final_kappa,
        args.text_guard_weight,
    )
    if not all(math.isfinite(value) and value > 0 for value in positive_floats):
        raise ValueError("learning/schedule/guard values must be finite positive")
    bounded = (
        args.teacher_kl_weight,
        args.switch_penalty,
        args.scale_l2_weight,
        args.text_max_relative_regression,
        args.selection_min_relative_loss_improvement,
        args.selection_max_macro_regression,
        args.selection_max_per_task_loss_regression,
    )
    if not all(math.isfinite(value) and value >= 0 for value in bounded):
        raise ValueError("loss weights and guard tolerance must be finite non-negative")
    if not 0 <= args.max_switch_rate <= 1:
        raise ValueError("max switch rate must be in [0, 1]")
    if len(set(args.layers)) != len(args.layers):
        raise ValueError("calibrated layers must be unique")
    if args.task_split_offset < 0:
        raise ValueError("task split offset must be non-negative")
    if args.task_audit_offset is not None and args.task_audit_offset < 0:
        raise ValueError("task audit offset must be non-negative")
    if args.train_group_scales and args.optimization_objective != "task_adapted":
        raise ValueError("joint group-scale training requires task_adapted objective")
    if set(args.tasks) != set(DEFAULT_TASKS):
        raise ValueError(f"tasks must be exactly {list(DEFAULT_TASKS)}")
    flat_rows = [
        *args.text_train_rows,
        *args.text_validation_rows,
        *args.text_audit_rows,
    ]
    if len(flat_rows) != len(set(flat_rows)):
        raise ValueError("text train/validation/audit rows must be disjoint")


def _load_logical_weights(
    model: nn.Module, checkpoint: dict[str, Any], *, device: torch.device
) -> None:
    expected_layers = len(model.model.layers)
    if sorted(int(key) for key in checkpoint["layers"]) != list(range(expected_layers)):
        raise ValueError("logical checkpoint does not cover every decoder layer")
    with torch.no_grad():
        for layer in range(expected_layers):
            state = checkpoint["layers"][str(layer)]
            if set(state) != set(LINEAR_NAMES):
                raise ValueError(f"layer {layer} has incomplete Linear coverage")
            modules = get_linear_modules(model, layer)
            for name in LINEAR_NAMES:
                weight = reconstruct_logical_state(state[name], device)
                modules[name].weight.copy_(weight.to(modules[name].weight.dtype))


def _proposal_gradients(
    model: nn.Module,
    examples: list[dict[str, Any]],
    text_examples: list[dict[str, torch.Tensor | int]],
    *,
    layers: list[int],
    device: torch.device,
    teacher_kl_weight: float,
    teacher_temperature: float,
    text_guard_weight: float,
    optimization_objective: str,
    gradient_shards: int,
    capture_activation_second_moment: bool = False,
) -> tuple[
    dict[tuple[int, str], list[torch.Tensor]],
    dict[tuple[int, str], torch.Tensor] | None,
]:
    modules = {
        (layer, name): get_linear_modules(model, layer)[name]
        for layer in layers
        for name in LINEAR_NAMES
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in modules.values():
        module.weight.requires_grad_(True)
    views = partition_proposal_examples(
        examples,
        text_examples,
        shards=gradient_shards,
        optimization_objective=optimization_objective,
    )
    gradients: dict[tuple[int, str], list[torch.Tensor]] = {
        key: [] for key in modules
    }
    activation_sums = (
        {
            key: torch.zeros(
                module.in_features, device=device, dtype=torch.float32
            )
            for key, module in modules.items()
        }
        if capture_activation_second_moment
        else None
    )
    activation_counts = (
        {key: 0 for key in modules} if capture_activation_second_moment else None
    )
    handles: list[Any] = []
    if capture_activation_second_moment:
        assert activation_sums is not None and activation_counts is not None

        def capture_input_on_backward(key: tuple[int, str]):
            def hook(
                _module: nn.Module,
                inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
            ) -> None:
                if not torch.is_grad_enabled() or not output.requires_grad:
                    return
                captured = inputs[0].detach()

                def accumulate(_output_gradient: torch.Tensor) -> None:
                    flattened = captured.reshape(-1, captured.shape[-1]).float()
                    activation_sums[key].add_(flattened.square().sum(dim=0))
                    activation_counts[key] += int(flattened.shape[0])

                output.register_hook(accumulate)

            return hook

        handles = [
            module.register_forward_hook(capture_input_on_backward(key))
            for key, module in modules.items()
        ]
    try:
        for view_index, (task_view, text_view) in enumerate(views, start=1):
            model.zero_grad(set_to_none=True)
            if optimization_objective == "task_adapted":
                task_scale = 1.0 / max(len(task_view), 1)
                for example in task_view:
                    _train_one_example(
                        model,
                        example,
                        device=device,
                        teacher_kl_weight=teacher_kl_weight,
                        teacher_temperature=teacher_temperature,
                        loss_scale=task_scale,
                    )
            text_scale = text_guard_weight / max(len(text_view), 1)
            for example in text_view:
                (text_scale * _text_guard_loss(model, example, device=device)).backward()
            for key, module in modules.items():
                if module.weight.grad is None or not bool(
                    torch.isfinite(module.weight.grad).all()
                ):
                    raise RuntimeError(
                        f"missing/non-finite proposal gradient for {key} view {view_index}"
                    )
                gradients[key].append(module.weight.grad.detach().cpu().clone())
                module.weight.grad = None
            print(
                f"Collected proposal gradient view {view_index}/{len(views)} "
                f"({len(task_view)} task, {len(text_view)} text examples)",
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    for module in modules.values():
        module.weight.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    activation_second_moment: dict[tuple[int, str], torch.Tensor] | None = None
    if capture_activation_second_moment:
        assert activation_sums is not None and activation_counts is not None
        if any(count <= 0 for count in activation_counts.values()):
            raise RuntimeError("activation curvature collector observed no backward inputs")
        activation_second_moment = {
            key: (values / activation_counts[key]).detach().cpu()
            for key, values in activation_sums.items()
        }
    return gradients, activation_second_moment


def _vector_gradient_geometry(
    state: dict[str, Any], gradient: torch.Tensor, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = (int(value) for value in state["shape"])
    vector_dim = int(state["vector_dim"])
    padded_columns = math.ceil(columns / vector_dim) * vector_dim
    if tuple(gradient.shape) != (rows, columns):
        raise ValueError("weight gradient does not match logical state")
    padded_gradient = F.pad(
        gradient.to(device=device, dtype=torch.float32),
        (0, padded_columns - columns),
    )
    row_norm = state["row_norm"].to(device=device, dtype=torch.float32)
    col_norm = F.pad(
        state["col_norm"].to(device=device, dtype=torch.float32),
        (0, padded_columns - columns),
    )
    scales = row_norm[:, None] * col_norm[None, :]
    if "group_scale" in state:
        group_size = int(state["group_size"])
        group_scale = state["group_scale"].to(device=device, dtype=torch.float32)
        scales = scales * group_scale.repeat_interleave(
            group_size, dim=1
        )[:, :padded_columns]
    return (
        padded_gradient.reshape(-1, vector_dim),
        scales.reshape(-1, vector_dim),
    )


def _install_local_assignments(
    model: nn.Module,
    checkpoint: dict[str, Any],
    gradients: dict[tuple[int, str], list[torch.Tensor]],
    *,
    layers: list[int],
    device: torch.device,
    local_candidates: int,
    candidate_chunk: int,
    expectation_chunk: int,
    initial_logit_margin: float,
    train_group_scales: bool = False,
) -> tuple[
    dict[tuple[int, str], BinaryLocalAssignmentParametrization],
    dict[str, Any],
]:
    installed: dict[tuple[int, str], BinaryLocalAssignmentParametrization] = {}
    proposal: dict[str, Any] = {}
    for layer in layers:
        modules = get_linear_modules(model, layer)
        for name in LINEAR_NAMES:
            state = checkpoint["layers"][str(layer)][name]
            codebook = state["codebook"].to(device=device, dtype=torch.float32)
            assignments = state["assignments"].to(device=device, dtype=torch.long)
            neighbors = build_codebook_neighbor_table(
                codebook, candidates=min(local_candidates, codebook.shape[0])
            )
            gradient_vector_views: list[torch.Tensor] = []
            vector_scales: torch.Tensor | None = None
            for gradient in gradients[(layer, name)]:
                gradient_vectors, current_scales = _vector_gradient_geometry(
                    state, gradient, device=device
                )
                gradient_vector_views.append(gradient_vectors)
                if vector_scales is None:
                    vector_scales = current_scales
                else:
                    del current_scales
            assert vector_scales is not None
            if len(gradient_vector_views) == 1:
                alternatives, stats = select_gradient_directed_alternative(
                    assignments=assignments,
                    codebook=codebook,
                    neighbor_table=neighbors,
                    gradient_vectors=gradient_vector_views[0],
                    vector_scales=vector_scales,
                    chunk_size=candidate_chunk,
                )
                stats = {"gradient_views": 1, **stats}
            else:
                alternatives, stats = select_consensus_gradient_directed_alternative(
                    assignments=assignments,
                    codebook=codebook,
                    neighbor_table=neighbors,
                    gradient_vector_views=gradient_vector_views,
                    vector_scales=vector_scales,
                    chunk_size=candidate_chunk,
                )
            parametrization = BinaryLocalAssignmentParametrization(
                state=state,
                alternative_assignments=alternatives.cpu(),
                initial_logit_margin=initial_logit_margin,
                expectation_chunk=expectation_chunk,
                train_group_scale=train_group_scales,
            ).to(device)
            parametrize.register_parametrization(
                modules[name], "weight", parametrization
            )
            installed[(layer, name)] = parametrization
            proposal[f"{layer}:{name}"] = {
                "vectors": int(assignments.numel()),
                **stats,
            }
            del codebook, assignments, neighbors, gradient_vector_views, vector_scales
            torch.cuda.empty_cache()
    return installed, proposal


def _set_hard_mode(
    installed: dict[tuple[int, str], BinaryLocalAssignmentParametrization]
) -> None:
    for module in installed.values():
        module.set_hard_mode(True)


def _set_soft_schedule(
    installed: dict[tuple[int, str], BinaryLocalAssignmentParametrization],
    *,
    temperature: float,
    kappa: float,
) -> None:
    for module in installed.values():
        module.set_soft_schedule(temperature=temperature, kappa=kappa)


def _snapshot_logits(
    installed: dict[tuple[int, str], BinaryLocalAssignmentParametrization]
) -> dict[tuple[int, str], torch.Tensor]:
    return {
        key: module.switch_logits.detach().cpu().clone()
        for key, module in installed.items()
    }


def _restore_logits(
    installed: dict[tuple[int, str], BinaryLocalAssignmentParametrization],
    snapshot: dict[tuple[int, str], torch.Tensor],
) -> None:
    with torch.no_grad():
        for key, module in installed.items():
            module.switch_logits.copy_(snapshot[key].to(module.switch_logits.device))


def _snapshot_scale_deltas(
    installed: dict[tuple[int, str], BinaryLocalAssignmentParametrization]
) -> dict[tuple[int, str], torch.Tensor]:
    return {
        key: module.log_group_scale_delta.detach().cpu().clone()
        for key, module in installed.items()
        if module.log_group_scale_delta is not None
    }


def _restore_scale_deltas(
    installed: dict[tuple[int, str], BinaryLocalAssignmentParametrization],
    snapshot: dict[tuple[int, str], torch.Tensor],
) -> None:
    with torch.no_grad():
        for key, values in snapshot.items():
            parameter = installed[key].log_group_scale_delta
            if parameter is None:
                raise ValueError("scale snapshot supplied to assignment-only module")
            parameter.copy_(values.to(parameter.device))


def _switch_statistics(
    installed: dict[tuple[int, str], BinaryLocalAssignmentParametrization]
) -> dict[str, Any]:
    switched = 0
    total = 0
    per_linear: dict[str, Any] = {}
    for (layer, name), module in installed.items():
        count = int((module.switch_logits > 0).sum())
        vectors = module.switch_logits.numel()
        switched += count
        total += vectors
        per_linear[f"{layer}:{name}"] = {
            "switched": count,
            "vectors": vectors,
            "rate": count / max(vectors, 1),
        }
    return {
        "switched": switched,
        "vectors": total,
        "rate": switched / max(total, 1),
        "per_linear": per_linear,
    }


def _interpolate_schedule(initial: float, final: float, fraction: float) -> float:
    return initial + (final - initial) * fraction


def main() -> None:
    args = parse_args()
    _validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    started = time.time()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=not args.trusted_checkpoint,
    )
    original_checkpoint = copy.deepcopy(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    if min(args.layers) < 0 or max(args.layers) >= len(model.model.layers):
        raise ValueError("calibrated layer is outside the decoder range")

    examples, split_manifest = _build_examples(
        tokenizer,
        task_names=list(args.tasks),
        train_per_task=args.train_per_task,
        validation_per_task=args.validation_per_task,
        audit_per_task=args.audit_per_task,
        seed=args.seed,
        max_length=args.max_length,
        split_offset=args.task_split_offset,
        audit_offset=args.task_audit_offset,
    )
    if args.optimization_objective == "task_adapted":
        print("Collecting dense-teacher multiple-choice scores", flush=True)
        _teacher_scores(model, examples, device=device)
    else:
        print(
            "Skipping task labels and multiple-choice teacher scores for "
            "text_teacher_only optimization",
            flush=True,
        )
    print("Collecting dense-teacher long-text top-k distributions", flush=True)
    text_examples = _build_text_guard_examples(
        model,
        token_cache=args.text_token_cache,
        rows_by_split={
            "train": list(args.text_train_rows),
            "validation": list(args.text_validation_rows),
            "audit": list(args.text_audit_rows),
        },
        sequence_length=args.text_sequence_length,
        teacher_topk=args.text_teacher_topk,
        device=device,
    )

    _load_logical_weights(model, checkpoint, device=device)
    baseline_validation = _evaluate_examples(
        model, examples["validation"], device=device
    )
    baseline_text_validation = _evaluate_text_guard(
        model, text_examples["validation"], device=device
    )
    baseline_validation["text_guard_cross_entropy"] = baseline_text_validation
    baseline_audit = _evaluate_examples(model, examples["audit"], device=device)
    baseline_text_audit = _evaluate_text_guard(
        model, text_examples["audit"], device=device
    )
    baseline_audit["text_guard_cross_entropy"] = baseline_text_audit

    print(
        f"Collecting {args.optimization_objective} assignment proposal gradients",
        flush=True,
    )
    gradients, _activation_second_moment = _proposal_gradients(
        model,
        examples["train"],
        text_examples["train"],
        layers=list(args.layers),
        device=device,
        teacher_kl_weight=args.teacher_kl_weight,
        teacher_temperature=args.teacher_temperature,
        text_guard_weight=args.text_guard_weight,
        optimization_objective=args.optimization_objective,
        gradient_shards=args.proposal_gradient_shards,
    )
    print("Building gradient-directed anchor-local alternatives", flush=True)
    installed, proposal = _install_local_assignments(
        model,
        checkpoint,
        gradients,
        layers=list(args.layers),
        device=device,
        local_candidates=args.local_candidates,
        candidate_chunk=args.candidate_chunk,
        expectation_chunk=args.expectation_chunk,
        initial_logit_margin=args.initial_logit_margin,
        train_group_scales=args.train_group_scales,
    )
    del gradients
    torch.cuda.empty_cache()

    switch_parameters = [module.switch_logits for module in installed.values()]
    scale_parameters = [
        module.log_group_scale_delta
        for module in installed.values()
        if module.log_group_scale_delta is not None
    ]
    trainable = [*switch_parameters, *scale_parameters]
    optimizer_groups: list[dict[str, Any]] = [
        {"params": switch_parameters, "lr": args.learning_rate}
    ]
    if scale_parameters:
        optimizer_groups.append(
            {"params": scale_parameters, "lr": args.scale_learning_rate}
        )
    optimizer = torch.optim.Adam(optimizer_groups)
    anchor_snapshot = _snapshot_logits(installed)
    anchor_scale_snapshot = _snapshot_scale_deltas(installed)
    best_validation = baseline_validation
    best_epoch = 0
    best_snapshot = anchor_snapshot
    best_scale_snapshot = anchor_scale_snapshot
    best_projection_switches: int | None = None
    task_epoch_candidates: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "validation": baseline_validation,
            "switch": _switch_statistics(installed),
        }
    ]
    generator = random.Random(args.seed)
    text_train_position = 0

    for epoch in range(1, args.epochs + 1):
        fraction = (epoch - 1) / max(args.epochs - 1, 1)
        temperature = _interpolate_schedule(
            args.initial_temperature, args.final_temperature, fraction
        )
        kappa = _interpolate_schedule(args.initial_kappa, args.final_kappa, fraction)
        _set_soft_schedule(installed, temperature=temperature, kappa=kappa)
        order = list(examples["train"])
        generator.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        losses: list[dict[str, float]] = []
        if args.optimization_objective == "task_adapted":
            for position, example in enumerate(order, start=1):
                losses.append(
                    _train_one_example(
                        model,
                        example,
                        device=device,
                        teacher_kl_weight=args.teacher_kl_weight,
                        teacher_temperature=args.teacher_temperature,
                        loss_scale=1.0 / args.gradient_accumulation,
                    )
                )
                boundary = (
                    position % args.gradient_accumulation == 0
                    or position == len(order)
                )
                if not boundary:
                    continue
                text_example = text_examples["train"][
                    text_train_position % len(text_examples["train"])
                ]
                text_train_position += 1
                (
                    args.text_guard_weight
                    * _text_guard_loss(model, text_example, device=device)
                ).backward()
                if args.switch_penalty:
                    penalty = torch.stack(
                        [module.switch_probabilities().mean() for module in installed.values()]
                    ).mean()
                    (args.switch_penalty * penalty).backward()
                if scale_parameters and args.scale_l2_weight:
                    scale_regularizer = torch.stack(
                        [parameter.square().mean() for parameter in scale_parameters]
                    ).mean()
                    (args.scale_l2_weight * scale_regularizer).backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                with torch.no_grad():
                    for parameter in scale_parameters:
                        parameter.clamp_(
                            -args.max_log_scale_delta,
                            args.max_log_scale_delta,
                        )
                optimizer.zero_grad(set_to_none=True)
        else:
            # Match the task-adapted path's number of optimizer updates while
            # excluding every multiple-choice example from gradient computation.
            text_steps = math.ceil(len(order) / args.gradient_accumulation)
            for _ in range(text_steps):
                text_example = text_examples["train"][
                    text_train_position % len(text_examples["train"])
                ]
                text_train_position += 1
                text_loss = _text_guard_loss(model, text_example, device=device)
                (args.text_guard_weight * text_loss).backward()
                if args.switch_penalty:
                    penalty = torch.stack(
                        [module.switch_probabilities().mean() for module in installed.values()]
                    ).mean()
                    (args.switch_penalty * penalty).backward()
                if scale_parameters and args.scale_l2_weight:
                    scale_regularizer = torch.stack(
                        [parameter.square().mean() for parameter in scale_parameters]
                    ).mean()
                    (args.scale_l2_weight * scale_regularizer).backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                with torch.no_grad():
                    for parameter in scale_parameters:
                        parameter.clamp_(
                            -args.max_log_scale_delta,
                            args.max_log_scale_delta,
                        )
                optimizer.zero_grad(set_to_none=True)
                losses.append(
                    {
                        "supervised_loss": 0.0,
                        "teacher_kl": 0.0,
                        "combined_loss": float(text_loss.detach()),
                    }
                )

        _set_hard_mode(installed)
        validation = _evaluate_examples(
            model, examples["validation"], device=device
        )
        if args.optimization_objective == "task_adapted":
            projection_source_eligible = (
                True
                if args.train_group_scales
                else choose_projection_source_snapshot(
                    baseline_validation,
                    validation,
                    min_relative_loss_improvement=(
                        args.selection_min_relative_loss_improvement
                    ),
                    max_per_task_loss_regression=(
                        args.selection_max_per_task_loss_regression
                    ),
                )
            )
            validation["text_guard_cross_entropy"] = None
        else:
            validation["text_guard_cross_entropy"] = _evaluate_text_guard(
                model, text_examples["validation"], device=device
            )
            # A dense hard state may be poor even when its high-confidence
            # prefix is useful. Always inspect nested projections; the held-out
            # text-only selector below remains the acceptance criterion.
            projection_source_eligible = True
        switches = _switch_statistics(installed)
        # A bad or over-budget dense state can still contain a useful sparse
        # confidence prefix. Every projected state receives its own validation,
        # text, and rate gate, so the full state must not reject the path.
        candidate_for_projection = should_evaluate_projection_path(
            full_switches=int(switches["switched"])
        )
        record = {
            "epoch": epoch,
            "temperature": temperature,
            "kappa": kappa,
            "mean_train_combined_loss": sum(
                item["combined_loss"] for item in losses
            )
            / len(losses),
            "validation": validation,
            "switch": switches,
            "projection_source_eligible_against_anchor": (
                projection_source_eligible
            ),
            "candidate_for_confidence_projection": candidate_for_projection,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if candidate_for_projection:
            task_epoch_candidates.append(
                {
                    "epoch": epoch,
                    "snapshot": _snapshot_logits(installed),
                    "scale_snapshot": _snapshot_scale_deltas(installed),
                    "validation": validation,
                    "switch": switches,
                }
            )

    projection_history: list[dict[str, Any]] = []
    for epoch_candidate in task_epoch_candidates:
        epoch = int(epoch_candidate["epoch"])
        full_snapshot = epoch_candidate["snapshot"]
        scale_snapshot = epoch_candidate["scale_snapshot"]
        full_switches = int(epoch_candidate["switch"]["switched"])
        projection_counts = build_confidence_projection_counts(
            full_switches, stages=args.projection_stages
        )
        if args.train_group_scales:
            projection_counts = sorted(set([0, *projection_counts]))
        for keep_switches in projection_counts:
            projected_snapshot = project_snapshot_by_confidence(
                full_snapshot, max_switches=keep_switches
            )
            _restore_logits(installed, projected_snapshot)
            _restore_scale_deltas(installed, scale_snapshot)
            _set_hard_mode(installed)
            projection_validation = _evaluate_examples(
                model, examples["validation"], device=device
            )
            if args.optimization_objective == "task_adapted":
                task_projection_eligible = choose_task_loss_dominant_snapshot(
                    baseline_validation,
                    projection_validation,
                    min_relative_loss_improvement=(
                        args.selection_min_relative_loss_improvement
                    ),
                    max_macro_regression=args.selection_max_macro_regression,
                    max_per_task_loss_regression=(
                        args.selection_max_per_task_loss_regression
                    ),
                )
                projection_validation["text_guard_cross_entropy"] = (
                    _evaluate_text_guard(
                        model, text_examples["validation"], device=device
                    )
                    if task_projection_eligible
                    else None
                )
                eligible = choose_loss_dominant_snapshot(
                    baseline_validation,
                    projection_validation,
                    min_relative_loss_improvement=(
                        args.selection_min_relative_loss_improvement
                    ),
                    max_macro_regression=args.selection_max_macro_regression,
                    max_per_task_loss_regression=(
                        args.selection_max_per_task_loss_regression
                    ),
                    max_text_loss_regression=args.text_max_relative_regression,
                )
                selection_loss_key = "balanced_loss"
            else:
                projection_validation["text_guard_cross_entropy"] = (
                    _evaluate_text_guard(
                        model, text_examples["validation"], device=device
                    )
                )
                eligible = choose_text_loss_dominant_snapshot(
                    baseline_validation,
                    projection_validation,
                    min_relative_loss_improvement=(
                        args.selection_min_relative_loss_improvement
                    ),
                )
                selection_loss_key = "text_guard_cross_entropy"
            switches = _switch_statistics(installed)
            projection_preference = (
                prefer_task_loss_projection
                if args.train_group_scales
                else prefer_sparse_projection
            )
            improved_incumbent = (
                eligible
                and switches["rate"] <= args.max_switch_rate
                and projection_preference(
                    incumbent_switches=best_projection_switches,
                    incumbent_loss=float(best_validation[selection_loss_key]),
                    candidate_switches=int(switches["switched"]),
                    candidate_loss=float(projection_validation[selection_loss_key]),
                )
            )
            projection_record = {
                "epoch": epoch,
                "keep_switches": keep_switches,
                "full_switches": full_switches,
                "keep_fraction": keep_switches / max(full_switches, 1),
                "validation": projection_validation,
                "switch": switches,
                "eligible_against_anchor": eligible,
                "improved_incumbent": improved_incumbent,
            }
            projection_history.append(projection_record)
            print(json.dumps({"projection": projection_record}, ensure_ascii=False), flush=True)
            if improved_incumbent:
                best_validation = projection_validation
                best_epoch = epoch
                best_snapshot = projected_snapshot
                best_scale_snapshot = scale_snapshot
                best_projection_switches = int(switches["switched"])

    _restore_logits(installed, best_snapshot)
    _restore_scale_deltas(installed, best_scale_snapshot)
    _set_hard_mode(installed)
    selected_validation = _evaluate_examples(
        model, examples["validation"], device=device
    )
    selected_validation["text_guard_cross_entropy"] = _evaluate_text_guard(
        model, text_examples["validation"], device=device
    )
    candidate_audit = _evaluate_examples(model, examples["audit"], device=device)
    candidate_audit["text_guard_cross_entropy"] = _evaluate_text_guard(
        model, text_examples["audit"], device=device
    )
    final_switches = _switch_statistics(installed)
    if args.optimization_objective == "task_adapted":
        audit_objective_passed = choose_loss_dominant_snapshot(
            baseline_audit,
            candidate_audit,
            min_relative_loss_improvement=(
                args.selection_min_relative_loss_improvement
            ),
            max_macro_regression=args.selection_max_macro_regression,
            max_per_task_loss_regression=(
                args.selection_max_per_task_loss_regression
            ),
            max_text_loss_regression=args.text_max_relative_regression,
        )
    else:
        audit_objective_passed = choose_text_loss_dominant_snapshot(
            baseline_audit,
            candidate_audit,
            min_relative_loss_improvement=(
                args.selection_min_relative_loss_improvement
            ),
        )
    audit_gate_passed = (
        best_epoch > 0
        and audit_objective_passed
        and final_switches["rate"] <= args.max_switch_rate
    )
    final_audit = candidate_audit
    if not audit_gate_passed:
        _restore_logits(installed, anchor_snapshot)
        _restore_scale_deltas(installed, anchor_scale_snapshot)
        _set_hard_mode(installed)
        best_epoch = 0
        selected_validation = baseline_validation
        final_audit = baseline_audit
        final_switches = _switch_statistics(installed)

    final_scale_snapshot = _snapshot_scale_deltas(installed)
    if final_scale_snapshot:
        selected_scale_deltas = torch.cat(
            [values.reshape(-1) for values in final_scale_snapshot.values()]
        )
        scale_delta_stats: dict[str, Any] | None = {
            "count": int(selected_scale_deltas.numel()),
            "nonzero": int(torch.count_nonzero(selected_scale_deltas)),
            "min": float(selected_scale_deltas.min()),
            "max": float(selected_scale_deltas.max()),
            "mean_abs": float(selected_scale_deltas.abs().mean()),
        }
    else:
        scale_delta_stats = None
    group_scales_changed = bool(
        scale_delta_stats is not None and scale_delta_stats["nonzero"] > 0
    )

    reconstruction_max_abs = 0.0
    fixed_metadata_exact = True
    for (layer, name), module in installed.items():
        state = checkpoint["layers"][str(layer)][name]
        module.fold_deployment_state(state)
        reference = original_checkpoint["layers"][str(layer)][name]
        for key, value in state.items():
            if key == "assignments" or (
                args.train_group_scales and key == "group_scale"
            ):
                continue
            if isinstance(value, torch.Tensor) and not torch.equal(
                value, reference[key]
            ):
                fixed_metadata_exact = False
        reconstructed = reconstruct_logical_state(state, device)
        deployed = module(get_linear_modules(model, layer)[name].parametrizations.weight.original)
        reconstruction_max_abs = max(
            reconstruction_max_abs,
            float(
                (
                    reconstructed.to(deployed.dtype).float()
                    - deployed.float()
                ).abs().max()
            ),
        )
    if reconstruction_max_abs != 0.0:
        raise RuntimeError(
            f"serialized assignment reconstruction is not exact: {reconstruction_max_abs}"
        )
    if not fixed_metadata_exact:
        raise RuntimeError("codebook, normalizer, or fixed metadata changed")

    untouched_exact = True
    for layer_key, layer_state in checkpoint["layers"].items():
        for name, state in layer_state.items():
            if (int(layer_key), name) in installed:
                continue
            reference = original_checkpoint["layers"][layer_key][name]
            for key in state:
                if isinstance(state[key], torch.Tensor) and not torch.equal(
                    state[key], reference[key]
                ):
                    untouched_exact = False
    if not untouched_exact:
        raise RuntimeError("a logical state outside the calibrated scope changed")

    checkpoint["task_local_assignment_calibration"] = {
        "method": (
            "joint_group_scale_and_gradient_directed_anchor_local_binary_concrete"
            if args.train_group_scales
            else "strict_cross_view_consensus_anchor_local_binary_concrete"
            if args.proposal_gradient_shards > 1
            else "gradient_directed_anchor_local_binary_concrete"
        ),
        "optimization_objective": args.optimization_objective,
        "layers": list(args.layers),
        "tasks": list(args.tasks),
        "seed": args.seed,
        "task_split_offset": args.task_split_offset,
        "task_audit_offset": args.task_audit_offset,
        "best_epoch": best_epoch,
        "local_candidates": args.local_candidates,
        "proposal_gradient_shards": args.proposal_gradient_shards,
        "bitrate_changed": False,
        "switch_rate": final_switches["rate"],
        "group_scales_trained": args.train_group_scales,
        "selection": (
            "task_loss_ranked_joint_projection_with_independent_audit"
            if args.train_group_scales
            else "confidence_projected_task_and_text_loss_dominance_with_independent_audit"
            if args.optimization_objective == "task_adapted"
            else "confidence_projected_teacher_text_loss_dominance_with_independent_audit"
        ),
        "projection_stages": args.projection_stages,
    }
    output_checkpoint = Path(args.output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_checkpoint.with_suffix(output_checkpoint.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, output_checkpoint)

    token_cache_payload = torch.load(
        args.text_token_cache, map_location="cpu", weights_only=False
    )
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment": (
            "llama_vector_gsq_joint_group_scale_assignment_calibration"
            if args.train_group_scales
            else "llama_vector_gsq_task_local_assignment_calibration"
        ),
        "completed": True,
        "arguments": vars(args),
        "method": {
            "proposal": (
                "geometric_topk_then_strict_cross_view_gradient_consensus"
                if args.proposal_gradient_shards > 1
                else "geometric_topk_then_task_and_text_gradient_first_order_selection"
                if args.optimization_objective == "task_adapted"
                else "geometric_topk_then_teacher_text_gradient_first_order_selection"
            ),
            "proposal_gradient_shards": args.proposal_gradient_shards,
            "optimization_objective": args.optimization_objective,
            "relaxation": "anchor_local_binary_concrete",
            "trained_variables": (
                "assignment_logits_and_group_scale_log_deltas"
                if args.train_group_scales
                else "assignment_logits_only"
            ),
            "codebook_changed": False,
            "normalizers_changed": False,
            "group_scales_changed": group_scales_changed,
            "logical_bitrate_changed": False,
            "selection": {
                "criterion": (
                    "task_loss_ranked_joint_projection_with_independent_audit"
                    if args.train_group_scales
                    else "confidence_projected_task_and_text_loss_dominance_with_independent_audit"
                    if args.optimization_objective == "task_adapted"
                    else "confidence_projected_teacher_text_loss_dominance_with_independent_audit"
                ),
                "min_relative_balanced_loss_improvement": (
                    args.selection_min_relative_loss_improvement
                ),
                "max_macro_accuracy_regression": (
                    args.selection_max_macro_regression
                ),
                "max_per_task_loss_regression": (
                    args.selection_max_per_task_loss_regression
                ),
                "max_text_loss_regression": args.text_max_relative_regression,
            },
        },
        "data_contract": {
            "source": (
                "lm_eval_training_splits_and_fineweb_teacher"
                if args.optimization_objective == "task_adapted"
                else "fineweb_and_c4_dense_teacher_only"
            ),
            "labeled_task_examples_used_for_optimization_or_selection": (
                args.optimization_objective == "task_adapted"
            ),
            "formal_validation_or_test_used_for_tuning": False,
            "split_manifest": split_manifest,
            "request_hashes": {
                split: [example["request_hash"] for example in split_examples]
                for split, split_examples in examples.items()
            },
            "text_guard": {
                "token_cache_identity": token_cache_payload.get("identity"),
                "rows": {
                    "train": list(args.text_train_rows),
                    "validation": list(args.text_validation_rows),
                    "audit": list(args.text_audit_rows),
                },
                "sequence_length": args.text_sequence_length,
                "teacher_topk": args.text_teacher_topk,
                "formal_test_data": False,
            },
        },
        "proposal": proposal,
        "baseline_validation": baseline_validation,
        "baseline_audit": baseline_audit,
        "selected_validation": selected_validation,
        "candidate_audit": candidate_audit,
        "final_audit": final_audit,
        "audit_gate_passed": audit_gate_passed,
        "best_epoch": best_epoch,
        "history": history,
        "projection_history": projection_history,
        "switch": final_switches,
        "scale_delta": scale_delta_stats,
        "checkpoint_contract": {
            "assignments_changed": final_switches["switched"] > 0,
            "assignment_dtype": "int32",
            "codebooks_changed": False,
            "normalizers_changed": False,
            "group_scales_changed": group_scales_changed,
            "logical_bitrate_changed": False,
            "untouched_states_exact": untouched_exact,
            "fixed_metadata_exact": fixed_metadata_exact,
            "reconstruction_max_abs": reconstruction_max_abs,
        },
        "elapsed_seconds": time.time() - started,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    _atomic_json(Path(args.output_summary), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
