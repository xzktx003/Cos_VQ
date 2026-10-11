"""Test whether scale-conditioned curvature predicts deployed assignment loss.

This is a formal pre-training gate, not an assignment hyperparameter sweep.  It
uses the same complete task/text calibration protocol as the V11 consensus run,
then compares three matched-cardinality, nested hard-assignment rankings:

* aggregate first-order score;
* strict cross-view worst-case first-order score;
* strict consensus normalized by a scale-conditioned local activation-Hessian cost.

Only if the curvature ranking predicts lower real hard validation loss at most
matched budgets is another full assignment-training run methodologically
authorized.  Validation selects at most one direct endpoint; audit is touched
once and formal test data is never used here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.nowag_d6.calibrate_task_group_scales import (
    DEFAULT_TASKS,
    _atomic_json,
    _build_examples,
    _build_text_guard_examples,
    _evaluate_examples,
    _evaluate_text_guard,
    _teacher_scores,
)
from experiments.nowag_d6.calibrate_task_local_assignments import (
    _load_logical_weights,
    _proposal_gradients,
    _vector_gradient_geometry,
    build_codebook_neighbor_table,
    choose_loss_dominant_snapshot,
    choose_task_loss_dominant_snapshot,
)
from experiments.nowag_d6.diagnose_block_vector_gsq import (
    LINEAR_NAMES,
    get_linear_modules,
)


METHODS = ("aggregate_first_order", "strict_consensus", "curvature_normalized")


@dataclass
class RankedCandidates:
    vector_indices: torch.Tensor
    alternative_ids: torch.Tensor
    rank_scores: torch.Tensor
    mean_first_order: torch.Tensor
    worst_view_first_order: torch.Tensor
    curvature_cost: torch.Tensor
    eligible_count: int

    def prefix(self, count: int) -> "RankedCandidates":
        if count <= 0 or count > self.vector_indices.numel():
            raise ValueError("candidate prefix is outside the retained ranking")
        return RankedCandidates(
            vector_indices=self.vector_indices[:count],
            alternative_ids=self.alternative_ids[:count],
            rank_scores=self.rank_scores[:count],
            mean_first_order=self.mean_first_order[:count],
            worst_view_first_order=self.worst_view_first_order[:count],
            curvature_cost=self.curvature_cost[:count],
            eligible_count=self.eligible_count,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=int, nargs="+", default=[28, 29, 30, 31])
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--train-per-task", type=int, default=512)
    parser.add_argument("--validation-per-task", type=int, default=256)
    parser.add_argument("--audit-per-task", type=int, default=31)
    parser.add_argument("--task-split-offset", type=int, default=0)
    parser.add_argument("--task-audit-offset", type=int, default=1088)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--teacher-kl-weight", type=float, default=0.25)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--proposal-gradient-shards", type=int, default=4)
    parser.add_argument("--local-candidates", type=int, default=8)
    parser.add_argument("--candidate-chunk", type=int, default=65536)
    parser.add_argument(
        "--budgets-per-linear", type=int, nargs="+", default=[128, 512, 2048, 8192]
    )
    parser.add_argument("--predictor-min-wins", type=int, default=3)
    parser.add_argument("--predictor-loss-tolerance", type=float, default=1e-7)
    parser.add_argument("--text-token-cache", required=True)
    parser.add_argument("--text-train-rows", type=int, nargs="+", required=True)
    parser.add_argument("--text-validation-rows", type=int, nargs="+", required=True)
    parser.add_argument("--text-audit-rows", type=int, nargs="+", required=True)
    parser.add_argument("--text-sequence-length", type=int, default=4096)
    parser.add_argument("--text-teacher-topk", type=int, default=32)
    parser.add_argument("--text-guard-weight", type=float, default=0.25)
    parser.add_argument("--text-max-relative-regression", type=float, default=0.005)
    parser.add_argument(
        "--selection-min-relative-loss-improvement", type=float, default=0.005
    )
    parser.add_argument("--selection-max-macro-regression", type=float, default=0.01)
    parser.add_argument(
        "--selection-max-per-task-loss-regression", type=float, default=0.005
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trusted-checkpoint", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if set(args.tasks) != set(DEFAULT_TASKS):
        raise ValueError(f"tasks must be exactly {list(DEFAULT_TASKS)}")
    if args.proposal_gradient_shards < 2:
        raise ValueError("curvature probe requires at least two gradient views")
    if len(set(args.layers)) != len(args.layers) or min(args.layers) < 0:
        raise ValueError("layers must be unique and non-negative")
    if args.local_candidates < 2:
        raise ValueError("local candidate count must include anchor and a neighbor")
    if args.candidate_chunk <= 0:
        raise ValueError("candidate chunk must be positive")
    if not args.budgets_per_linear or min(args.budgets_per_linear) <= 0:
        raise ValueError("candidate budgets must be positive")
    if sorted(set(args.budgets_per_linear)) != list(args.budgets_per_linear):
        raise ValueError("candidate budgets must be unique and increasing")
    if not 1 <= args.predictor_min_wins <= len(args.budgets_per_linear):
        raise ValueError("predictor win requirement must fit the budget count")
    if args.predictor_loss_tolerance < 0:
        raise ValueError("predictor loss tolerance must be non-negative")
    positive = (
        args.train_per_task,
        args.validation_per_task,
        args.audit_per_task,
        args.max_length,
        args.text_sequence_length,
        args.text_teacher_topk,
    )
    if min(positive) <= 0:
        raise ValueError("data and sequence budgets must be positive")
    rows = [
        *args.text_train_rows,
        *args.text_validation_rows,
        *args.text_audit_rows,
    ]
    if len(rows) != len(set(rows)):
        raise ValueError("text train/validation/audit rows must be disjoint")


def _merge_smallest(
    incumbent: dict[str, torch.Tensor] | None,
    candidate: dict[str, torch.Tensor],
    *,
    retain: int,
) -> dict[str, torch.Tensor]:
    if retain <= 0:
        raise ValueError("retained ranking size must be positive")
    merged = (
        candidate
        if incumbent is None
        else {
            key: torch.cat((incumbent[key], candidate[key]), dim=0) for key in candidate
        }
    )
    keep = min(retain, merged["rank_scores"].numel())
    positions = torch.topk(
        merged["rank_scores"], k=keep, largest=False, sorted=False
    ).indices
    return {key: value[positions] for key, value in merged.items()}


@torch.no_grad()
def rank_scale_conditioned_candidates(
    *,
    assignments: torch.Tensor,
    codebook: torch.Tensor,
    neighbor_table: torch.Tensor,
    gradient_vector_views: list[torch.Tensor],
    vector_scales: torch.Tensor,
    activation_second_moment: torch.Tensor,
    rows: int,
    columns: int,
    chunk_size: int,
    retain: int,
) -> dict[str, RankedCandidates]:
    """Retain the best matched hard candidates under three fixed rankings."""
    if assignments.ndim != 1 or len(gradient_vector_views) < 2:
        raise ValueError("flat assignments and multiple gradient views are required")
    vector_dim = int(codebook.shape[1])
    padded_columns = math.ceil(columns / vector_dim) * vector_dim
    blocks_per_row = padded_columns // vector_dim
    if assignments.numel() != rows * blocks_per_row:
        raise ValueError("assignment count does not match padded weight geometry")
    expected = (assignments.numel(), vector_dim)
    if vector_scales.shape != expected or any(
        view.shape != expected for view in gradient_vector_views
    ):
        raise ValueError("gradient/scale geometry does not match assignments")
    if activation_second_moment.shape != (columns,):
        raise ValueError("activation second moment does not match input columns")
    if neighbor_table.shape[0] != codebook.shape[0] or neighbor_table.shape[1] < 2:
        raise ValueError("neighbor table does not match codebook")
    if chunk_size <= 0 or retain <= 0:
        raise ValueError("chunk and retained ranking sizes must be positive")

    padded_moment = F.pad(
        activation_second_moment.to(codebook.device, dtype=torch.float32),
        (0, padded_columns - columns),
    ).reshape(blocks_per_row, vector_dim)
    retained: dict[str, dict[str, torch.Tensor] | None] = {
        method: None for method in METHODS
    }
    eligible_counts = {method: 0 for method in METHODS}
    for start in range(0, assignments.numel(), chunk_size):
        end = min(start + chunk_size, assignments.numel())
        anchors = assignments[start:end].long()
        candidate_ids = neighbor_table[anchors, 1:].long()
        anchor_vectors = codebook[anchors].float()
        code_delta = codebook[candidate_ids].float() - anchor_vectors[:, None, :]
        scales = vector_scales[start:end].float()
        weight_delta = scales[:, None, :] * code_delta
        block_ids = torch.arange(start, end, device=assignments.device) % blocks_per_row
        curvature = (weight_delta.square() * padded_moment[block_ids, None, :]).sum(
            dim=2
        )
        view_scores = torch.stack(
            [
                (view[start:end].float()[:, None, :] * weight_delta).sum(dim=2)
                for view in gradient_vector_views
            ],
            dim=0,
        )
        mean_scores = view_scores.mean(dim=0)
        worst_scores = view_scores.max(dim=0).values
        rows_in_chunk = torch.arange(end - start, device=assignments.device)
        global_indices = torch.arange(start, end, device=assignments.device)

        best_mean, aggregate_positions = mean_scores.min(dim=1)
        aggregate_payload = {
            "vector_indices": global_indices,
            "alternative_ids": candidate_ids[rows_in_chunk, aggregate_positions],
            "rank_scores": best_mean,
            "mean_first_order": best_mean,
            "worst_view_first_order": worst_scores[rows_in_chunk, aggregate_positions],
            "curvature_cost": curvature[rows_in_chunk, aggregate_positions],
        }
        aggregate_mask = best_mean < 0

        consensus_neighbors = (view_scores < 0).all(dim=0)
        robust_scores = worst_scores.masked_fill(~consensus_neighbors, math.inf)
        best_worst, consensus_positions = robust_scores.min(dim=1)
        consensus_mask = torch.isfinite(best_worst)
        consensus_payload = {
            "vector_indices": global_indices,
            "alternative_ids": candidate_ids[rows_in_chunk, consensus_positions],
            "rank_scores": best_worst,
            "mean_first_order": mean_scores[rows_in_chunk, consensus_positions],
            "worst_view_first_order": best_worst,
            "curvature_cost": curvature[rows_in_chunk, consensus_positions],
        }

        normalized = worst_scores / curvature.clamp_min(1e-30).sqrt()
        normalized = normalized.masked_fill(~consensus_neighbors, math.inf)
        best_normalized, curvature_positions = normalized.min(dim=1)
        curvature_mask = torch.isfinite(best_normalized)
        curvature_payload = {
            "vector_indices": global_indices,
            "alternative_ids": candidate_ids[rows_in_chunk, curvature_positions],
            "rank_scores": best_normalized,
            "mean_first_order": mean_scores[rows_in_chunk, curvature_positions],
            "worst_view_first_order": worst_scores[rows_in_chunk, curvature_positions],
            "curvature_cost": curvature[rows_in_chunk, curvature_positions],
        }

        for method, payload, mask in (
            ("aggregate_first_order", aggregate_payload, aggregate_mask),
            ("strict_consensus", consensus_payload, consensus_mask),
            ("curvature_normalized", curvature_payload, curvature_mask),
        ):
            eligible_counts[method] += int(mask.sum())
            if not bool(mask.any()):
                continue
            masked = {key: value[mask] for key, value in payload.items()}
            if masked["rank_scores"].numel() > retain:
                positions = torch.topk(
                    masked["rank_scores"], k=retain, largest=False, sorted=False
                ).indices
                masked = {key: value[positions] for key, value in masked.items()}
            retained[method] = _merge_smallest(retained[method], masked, retain=retain)

    output: dict[str, RankedCandidates] = {}
    for method in METHODS:
        payload = retained[method]
        if payload is None or payload["rank_scores"].numel() < retain:
            raise RuntimeError(f"{method} retained fewer than {retain} candidates")
        order = payload["rank_scores"].argsort()
        output[method] = RankedCandidates(
            vector_indices=payload["vector_indices"][order].cpu(),
            alternative_ids=payload["alternative_ids"][order].to(torch.int32).cpu(),
            rank_scores=payload["rank_scores"][order].cpu(),
            mean_first_order=payload["mean_first_order"][order].cpu(),
            worst_view_first_order=payload["worst_view_first_order"][order].cpu(),
            curvature_cost=payload["curvature_cost"][order].cpu(),
            eligible_count=eligible_counts[method],
        )
    return output


def curvature_predictor_gate(
    evaluations: list[dict[str, Any]],
    *,
    budgets: list[int],
    min_wins: int,
    tolerance: float,
) -> dict[str, Any]:
    """Pre-registered matched-budget test for curvature ranking utility."""
    lookup = {
        (str(record["method"]), int(record["budget_per_linear"])): float(
            record["validation"]["balanced_loss"]
        )
        for record in evaluations
    }
    missing = [
        (method, budget)
        for method in METHODS
        for budget in budgets
        if (method, budget) not in lookup
    ]
    if missing:
        raise ValueError(f"matched predictor records are missing: {missing}")
    curvature_losses = [lookup[("curvature_normalized", budget)] for budget in budgets]
    controls: dict[str, Any] = {}
    passed = True
    for method in ("aggregate_first_order", "strict_consensus"):
        control_losses = [lookup[(method, budget)] for budget in budgets]
        wins = sum(
            curved + tolerance < control
            for curved, control in zip(curvature_losses, control_losses)
        )
        curved_mean = sum(curvature_losses) / len(curvature_losses)
        control_mean = sum(control_losses) / len(control_losses)
        mean_better = curved_mean + tolerance < control_mean
        controls[method] = {
            "wins": wins,
            "required_wins": min_wins,
            "curvature_mean_loss": curved_mean,
            "control_mean_loss": control_mean,
            "mean_better": mean_better,
            "passed": wins >= min_wins and mean_better,
        }
        passed = passed and bool(controls[method]["passed"])
    return {
        "passed": passed,
        "criterion": (
            "curvature loss must be strictly lower at >=min_wins matched budgets "
            "and lower on average than both controls"
        ),
        "loss_tolerance": tolerance,
        "controls": controls,
    }


class SparseAssignmentApplicator:
    """Apply and exactly restore small hard-assignment sets in model weights."""

    def __init__(
        self,
        model: nn.Module,
        checkpoint: dict[str, Any],
        *,
        layers: list[int],
        device: torch.device,
    ) -> None:
        self.model = model
        self.checkpoint = checkpoint
        self.layers = layers
        self.device = device
        self.active: dict[tuple[int, str], torch.Tensor] = {}

    @torch.no_grad()
    def _write(
        self,
        key: tuple[int, str],
        vector_indices: torch.Tensor,
        assignment_ids: torch.Tensor,
    ) -> None:
        layer, name = key
        state = self.checkpoint["layers"][str(layer)][name]
        rows, columns = (int(value) for value in state["shape"])
        vector_dim = int(state["vector_dim"])
        padded_columns = math.ceil(columns / vector_dim) * vector_dim
        blocks_per_row = padded_columns // vector_dim
        indices = vector_indices.to(self.device, dtype=torch.long)
        ids = assignment_ids.to(self.device, dtype=torch.long)
        if indices.numel() != ids.numel():
            raise ValueError("vector indices and assignment IDs must match")
        if indices.numel() == 0:
            return
        row_ids = torch.div(indices, blocks_per_row, rounding_mode="floor")
        block_ids = indices.remainder(blocks_per_row)
        offsets = torch.arange(vector_dim, device=self.device)
        column_ids = block_ids[:, None] * vector_dim + offsets[None, :]
        valid = column_ids < columns
        codebook_values = state["codebook"].to(self.device, dtype=torch.float32)[ids]
        row_scale = state["row_norm"].to(self.device, dtype=torch.float32)[
            row_ids, None
        ]
        safe_columns = column_ids.clamp_max(columns - 1)
        column_scale = state["col_norm"].to(self.device, dtype=torch.float32)[
            safe_columns
        ]
        scales = row_scale * column_scale
        if "group_scale" in state:
            group_size = int(state["group_size"])
            group_ids = torch.div(safe_columns, group_size, rounding_mode="floor")
            scales = (
                scales
                * state["group_scale"].to(self.device, dtype=torch.float32)[
                    row_ids[:, None], group_ids
                ]
            )
        values = codebook_values * scales
        module = get_linear_modules(self.model, layer)[name]
        expanded_rows = row_ids[:, None].expand_as(column_ids)
        module.weight[expanded_rows[valid], column_ids[valid]] = values[valid].to(
            module.weight.dtype
        )

    @torch.no_grad()
    def restore(self) -> None:
        for key, indices in self.active.items():
            layer, name = key
            assignments = self.checkpoint["layers"][str(layer)][name]["assignments"]
            self._write(key, indices, assignments[indices.long()])
        self.active.clear()

    @torch.no_grad()
    def apply(
        self,
        selections: dict[tuple[int, str], RankedCandidates],
    ) -> None:
        self.restore()
        for key, selection in selections.items():
            self._write(key, selection.vector_indices, selection.alternative_ids)
            self.active[key] = selection.vector_indices.clone()


def _ranking_summary(ranking: RankedCandidates) -> dict[str, Any]:
    return {
        "eligible_count": ranking.eligible_count,
        "retained_count": int(ranking.vector_indices.numel()),
        "rank_score_min": float(ranking.rank_scores.min()),
        "rank_score_max": float(ranking.rank_scores.max()),
        "rank_score_mean": float(ranking.rank_scores.mean()),
        "mean_first_order_mean": float(ranking.mean_first_order.mean()),
        "worst_view_first_order_mean": float(ranking.worst_view_first_order.mean()),
        "curvature_cost_mean": float(ranking.curvature_cost.mean()),
    }


def _selection_prediction(
    selections: dict[tuple[int, str], RankedCandidates],
) -> dict[str, float]:
    return {
        "rank_score_sum": sum(
            float(selection.rank_scores.double().sum())
            for selection in selections.values()
        ),
        "mean_first_order_sum": sum(
            float(selection.mean_first_order.double().sum())
            for selection in selections.values()
        ),
        "worst_view_first_order_sum": sum(
            float(selection.worst_view_first_order.double().sum())
            for selection in selections.values()
        ),
        "curvature_cost_sum": sum(
            float(selection.curvature_cost.double().sum())
            for selection in selections.values()
        ),
    }


def _checkpoint_with_selection(
    checkpoint: dict[str, Any],
    selections: dict[tuple[int, str], RankedCandidates],
    *,
    metadata: dict[str, Any],
    output: Path,
) -> None:
    for (layer, name), selection in selections.items():
        assignments = checkpoint["layers"][str(layer)][name]["assignments"]
        assignments[selection.vector_indices.long()] = selection.alternative_ids.to(
            assignments.dtype
        )
    checkpoint["scale_conditioned_curvature_assignment"] = metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, output)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    started = time.time()
    output_summary = Path(args.output_summary)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=not args.trusted_checkpoint,
    )
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
    if max(args.layers) >= len(model.model.layers):
        raise ValueError("probe layer is outside the decoder range")

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
    print("Collecting dense-teacher multiple-choice scores", flush=True)
    _teacher_scores(model, examples, device=device)
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
    baseline_validation["text_guard_cross_entropy"] = _evaluate_text_guard(
        model, text_examples["validation"], device=device
    )
    baseline_audit = _evaluate_examples(model, examples["audit"], device=device)
    baseline_audit["text_guard_cross_entropy"] = _evaluate_text_guard(
        model, text_examples["audit"], device=device
    )

    print("Collecting four-view gradients and activation curvature", flush=True)
    gradients, activation_second_moment = _proposal_gradients(
        model,
        examples["train"],
        text_examples["train"],
        layers=list(args.layers),
        device=device,
        teacher_kl_weight=args.teacher_kl_weight,
        teacher_temperature=args.teacher_temperature,
        text_guard_weight=args.text_guard_weight,
        optimization_objective="task_adapted",
        gradient_shards=args.proposal_gradient_shards,
        capture_activation_second_moment=True,
    )
    assert activation_second_moment is not None

    retained_per_linear = max(args.budgets_per_linear)
    rankings: dict[tuple[int, str], dict[str, RankedCandidates]] = {}
    proposal_summary: dict[str, Any] = {}
    total_vectors = 0
    for layer in args.layers:
        for name in LINEAR_NAMES:
            key = (layer, name)
            state = checkpoint["layers"][str(layer)][name]
            codebook = state["codebook"].to(device=device, dtype=torch.float32)
            assignments = state["assignments"].to(device=device, dtype=torch.long)
            neighbors = build_codebook_neighbor_table(
                codebook,
                candidates=min(args.local_candidates, codebook.shape[0]),
            )
            gradient_views: list[torch.Tensor] = []
            vector_scales: torch.Tensor | None = None
            for gradient in gradients.pop(key):
                vector_gradient, current_scales = _vector_gradient_geometry(
                    state, gradient, device=device
                )
                gradient_views.append(vector_gradient)
                if vector_scales is None:
                    vector_scales = current_scales
            assert vector_scales is not None
            rows, columns = (int(value) for value in state["shape"])
            ranked = rank_scale_conditioned_candidates(
                assignments=assignments,
                codebook=codebook,
                neighbor_table=neighbors,
                gradient_vector_views=gradient_views,
                vector_scales=vector_scales,
                activation_second_moment=activation_second_moment.pop(key).to(device),
                rows=rows,
                columns=columns,
                chunk_size=args.candidate_chunk,
                retain=retained_per_linear,
            )
            rankings[key] = ranked
            total_vectors += int(assignments.numel())
            proposal_summary[f"{layer}:{name}"] = {
                "vectors": int(assignments.numel()),
                **{method: _ranking_summary(ranked[method]) for method in METHODS},
            }
            print(
                json.dumps(
                    {
                        "ranked_linear": f"{layer}:{name}",
                        **proposal_summary[f"{layer}:{name}"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            del codebook, assignments, neighbors, gradient_views, vector_scales
            torch.cuda.empty_cache()
    if gradients or activation_second_moment:
        raise RuntimeError("proposal geometry contains unconsumed Linear states")

    applicator = SparseAssignmentApplicator(
        model, checkpoint, layers=list(args.layers), device=device
    )
    evaluations: list[dict[str, Any]] = []
    selected_prefixes: dict[
        tuple[str, int], dict[tuple[int, str], RankedCandidates]
    ] = {}
    progress = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment": "scale_conditioned_assignment_curvature_predictor_probe",
        "completed": False,
        "arguments": vars(args),
        "baseline_validation": baseline_validation,
        "baseline_audit": baseline_audit,
        "proposal": proposal_summary,
        "evaluations": evaluations,
    }
    _atomic_json(output_summary, progress)

    for method in METHODS:
        for budget in args.budgets_per_linear:
            selections = {
                key: per_method[method].prefix(budget)
                for key, per_method in rankings.items()
            }
            selected_prefixes[(method, budget)] = selections
            applicator.apply(selections)
            validation = _evaluate_examples(
                model, examples["validation"], device=device
            )
            validation["text_guard_cross_entropy"] = None
            switches = sum(
                int(selection.vector_indices.numel())
                for selection in selections.values()
            )
            record = {
                "method": method,
                "budget_per_linear": budget,
                "switches": switches,
                "switch_rate": switches / total_vectors,
                "prediction": _selection_prediction(selections),
                "validation": validation,
                "balanced_loss_delta": (
                    float(validation["balanced_loss"])
                    - float(baseline_validation["balanced_loss"])
                ),
                "macro_accuracy_delta": (
                    float(validation["macro_accuracy"])
                    - float(baseline_validation["macro_accuracy"])
                ),
            }
            evaluations.append(record)
            progress["evaluations"] = evaluations
            _atomic_json(output_summary, progress)
            print(json.dumps({"hard_set": record}, ensure_ascii=False), flush=True)
    applicator.restore()

    predictor = curvature_predictor_gate(
        evaluations,
        budgets=list(args.budgets_per_linear),
        min_wins=args.predictor_min_wins,
        tolerance=args.predictor_loss_tolerance,
    )
    curvature_records = [
        record for record in evaluations if record["method"] == "curvature_normalized"
    ]
    best_record = min(
        curvature_records,
        key=lambda record: (
            float(record["validation"]["balanced_loss"]),
            int(record["switches"]),
        ),
    )
    best_budget = int(best_record["budget_per_linear"])
    best_selections = selected_prefixes[("curvature_normalized", best_budget)]
    applicator.apply(best_selections)
    task_validation_eligible = choose_task_loss_dominant_snapshot(
        baseline_validation,
        best_record["validation"],
        min_relative_loss_improvement=args.selection_min_relative_loss_improvement,
        max_macro_regression=args.selection_max_macro_regression,
        max_per_task_loss_regression=args.selection_max_per_task_loss_regression,
    )
    if task_validation_eligible:
        best_record["validation"]["text_guard_cross_entropy"] = _evaluate_text_guard(
            model, text_examples["validation"], device=device
        )
    validation_gate_passed = predictor["passed"] and choose_loss_dominant_snapshot(
        baseline_validation,
        best_record["validation"],
        min_relative_loss_improvement=args.selection_min_relative_loss_improvement,
        max_macro_regression=args.selection_max_macro_regression,
        max_per_task_loss_regression=args.selection_max_per_task_loss_regression,
        max_text_loss_regression=args.text_max_relative_regression,
    )

    candidate_audit: dict[str, Any] | None = None
    audit_gate_passed = False
    if validation_gate_passed:
        candidate_audit = _evaluate_examples(model, examples["audit"], device=device)
        candidate_audit["text_guard_cross_entropy"] = _evaluate_text_guard(
            model, text_examples["audit"], device=device
        )
        audit_gate_passed = choose_loss_dominant_snapshot(
            baseline_audit,
            candidate_audit,
            min_relative_loss_improvement=args.selection_min_relative_loss_improvement,
            max_macro_regression=args.selection_max_macro_regression,
            max_per_task_loss_regression=args.selection_max_per_task_loss_regression,
            max_text_loss_regression=args.text_max_relative_regression,
        )
    applicator.restore()

    checkpoint_written = False
    if audit_gate_passed:
        if not args.output_checkpoint:
            raise ValueError("an accepted endpoint requires --output-checkpoint")
        _checkpoint_with_selection(
            checkpoint,
            best_selections,
            metadata={
                "method": "strict_consensus_scale_conditioned_curvature_ranking",
                "layers": list(args.layers),
                "budget_per_linear": best_budget,
                "switches": int(best_record["switches"]),
                "switch_rate": float(best_record["switch_rate"]),
                "logical_bitrate_changed": False,
            },
            output=Path(args.output_checkpoint),
        )
        checkpoint_written = True

    progress.update(
        {
            "completed": True,
            "method": {
                "candidate_pool": "current_codeword_8_way_geometric_neighborhood",
                "gradient": "four_disjoint_task_stratified_views",
                "curvature": (
                    "scale_conditioned_diagonal_local_activation_hessian_cost"
                ),
                "curvature_scope": (
                    "exact_for_diagonalized_linear_output_quadratic_not_claimed_as_"
                    "the_full_task_loss_hessian"
                ),
                "curvature_rank": "worst_view_first_order_over_sqrt_curvature",
                "hard_set_design": "matched_cardinality_nested_per_linear",
                "trained_parameters": 0,
                "logical_bitrate_changed": False,
            },
            "data_contract": {
                "split_manifest": split_manifest,
                "task_train_per_task": args.train_per_task,
                "task_validation_per_task": args.validation_per_task,
                "task_audit_per_task": args.audit_per_task,
                "text_rows": {
                    "train": list(args.text_train_rows),
                    "validation": list(args.text_validation_rows),
                    "audit": list(args.text_audit_rows),
                },
                "text_sequence_length": args.text_sequence_length,
                "formal_test_data_used": False,
            },
            "total_vectors": total_vectors,
            "evaluations": evaluations,
            "predictor_gate": predictor,
            "full_assignment_training_authorized": bool(predictor["passed"]),
            "selected_curvature_budget_per_linear": best_budget,
            "selected_validation": best_record["validation"],
            "validation_gate_passed": validation_gate_passed,
            "candidate_audit": candidate_audit,
            "audit_gate_passed": audit_gate_passed,
            "checkpoint_written": checkpoint_written,
            "output_checkpoint": args.output_checkpoint if checkpoint_written else None,
            "elapsed_seconds": time.time() - started,
            "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
    )
    _atomic_json(output_summary, progress)
    print(json.dumps(progress, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
