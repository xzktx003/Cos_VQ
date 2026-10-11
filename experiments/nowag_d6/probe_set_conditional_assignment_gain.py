"""Search hard VQ assignment bundles by their conditional end-to-end gain.

V12 showed that an independent candidate curvature score can improve ranking
without certifying a deployable hard set.  This probe therefore treats one
fixed-size Linear-local bundle as a discrete coordinate.  In causal layer
order it evaluates all seven coordinates under the already accepted hard
state and accepts at most one bundle per Transformer block, only when the full
task-training objective strictly decreases.  Validation and audit remain
unseen by the coordinate search.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable

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
    _training_coefficients,
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
from experiments.nowag_d6.probe_scale_conditioned_assignment_curvature import (
    RankedCandidates,
    SparseAssignmentApplicator,
    _checkpoint_with_selection,
    _ranking_summary,
    _selection_prediction,
    rank_scale_conditioned_candidates,
)
from experiments.nowag_d6.run_prefix_vector_gsq import run_hidden_block_output


@dataclass
class CachedChoiceBatch:
    hidden_states: torch.Tensor
    input_ids: torch.Tensor
    metadata: tuple[tuple[int, int, int], ...]


@dataclass
class PrefixTaskCache:
    start_layer: int
    batches: list[CachedChoiceBatch]
    example_count: int
    choice_count: int
    token_count: int


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
    parser.add_argument("--bundle-size", type=int, default=128)
    parser.add_argument("--conditional-tolerance", type=float, default=1e-7)
    parser.add_argument("--cache-start-layer", type=int, default=28)
    parser.add_argument("--cache-batch-size", type=int, default=4)
    parser.add_argument("--cache-parity-examples", type=int, default=32)
    parser.add_argument("--cache-parity-tolerance", type=float, default=0.02)
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


def _validate_args(
    args: argparse.Namespace, *, model_layers: int | None = None
) -> None:
    positive = (
        args.train_per_task,
        args.validation_per_task,
        args.audit_per_task,
        args.max_length,
        args.proposal_gradient_shards,
        args.local_candidates,
        args.candidate_chunk,
        args.bundle_size,
        args.cache_batch_size,
        args.cache_parity_examples,
        args.text_sequence_length,
        args.text_teacher_topk,
    )
    if min(positive) <= 0:
        raise ValueError(
            "counts, chunks, cache sizes, and sequence lengths must be positive"
        )
    nonnegative = (
        args.task_split_offset,
        args.task_audit_offset,
        args.conditional_tolerance,
        args.cache_parity_tolerance,
        args.teacher_kl_weight,
        args.text_guard_weight,
        args.text_max_relative_regression,
        args.selection_min_relative_loss_improvement,
        args.selection_max_macro_regression,
        args.selection_max_per_task_loss_regression,
    )
    if not all(math.isfinite(value) and value >= 0 for value in nonnegative):
        raise ValueError(
            "offsets, tolerances, and loss weights must be finite non-negative"
        )
    if not math.isfinite(args.teacher_temperature) or args.teacher_temperature <= 0:
        raise ValueError("teacher temperature must be finite positive")
    if args.layers != sorted(set(args.layers)):
        raise ValueError("layers must be unique and in causal order")
    if args.cache_start_layer != min(args.layers):
        raise ValueError("cache start must equal the first searched layer")
    if args.layers != list(range(min(args.layers), max(args.layers) + 1)):
        raise ValueError("set-conditional search requires consecutive layers")
    if set(args.tasks) != set(DEFAULT_TASKS):
        raise ValueError(f"tasks must be exactly {list(DEFAULT_TASKS)}")
    flat_rows = [
        *args.text_train_rows,
        *args.text_validation_rows,
        *args.text_audit_rows,
    ]
    if len(flat_rows) != len(set(flat_rows)):
        raise ValueError("text train/validation/audit rows must be disjoint")
    if model_layers is not None and not 0 <= args.cache_start_layer < model_layers:
        raise ValueError("cache start layer is outside the model")
    if model_layers is not None and max(args.layers) >= model_layers:
        raise ValueError("searched layer is outside the model")


def accept_conditional_bundle(
    *,
    incumbent_loss: float,
    candidate_losses: dict[Hashable, float],
    tolerance: float,
) -> dict[str, Any]:
    """Choose the best coordinate only when its measured hard gain is strict."""
    if (
        not math.isfinite(incumbent_loss)
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("incumbent loss and tolerance must be finite")
    if not candidate_losses:
        raise ValueError("conditional bundle decision requires candidates")
    if not all(math.isfinite(value) for value in candidate_losses.values()):
        raise ValueError("candidate losses must be finite")
    key, loss = min(candidate_losses.items(), key=lambda item: (item[1], str(item[0])))
    accepted = loss + tolerance < incumbent_loss
    return {
        "accepted": accepted,
        "key": key if accepted else None,
        "loss": float(loss if accepted else incumbent_loss),
        "gain": float(incumbent_loss - loss) if accepted else 0.0,
    }


def _average_tie_ranks(values: list[float]) -> torch.Tensor:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = torch.empty(len(values), dtype=torch.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + end - 1) / 2.0
        for index in order[cursor:end]:
            ranks[index] = average
        cursor = end
    return ranks


def spearman_rank_correlation(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Spearman inputs must have equal length of at least two")
    if not all(math.isfinite(value) for value in (*first, *second)):
        raise ValueError("Spearman inputs must be finite")
    x = _average_tie_ranks(first)
    y = _average_tie_ranks(second)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y))
    if denominator == 0:
        return None
    return float(torch.dot(x, y) / denominator)


def summarize_cached_task_scores(
    scores_by_example: list[torch.Tensor],
    examples: list[dict[str, Any]],
    *,
    teacher_kl_weight: float,
    teacher_temperature: float,
) -> dict[str, Any]:
    if len(scores_by_example) != len(examples) or not examples:
        raise ValueError("cached scores must cover every task example")
    by_task: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    for scores, example in zip(scores_by_example, examples, strict=True):
        if scores.ndim != 1 or scores.numel() != len(example.get("teacher_scores", [])):
            raise ValueError("cached choice scores do not match teacher choices")
        _coefficients, losses = _training_coefficients(
            scores.float(),
            gold=int(example["gold"]),
            teacher_scores=example["teacher_scores"].float(),
            teacher_kl_weight=teacher_kl_weight,
            teacher_temperature=teacher_temperature,
        )
        by_task[str(example["task"])].append(
            {
                **losses,
                "correct": int(scores.argmax()) == int(example["gold"]),
            }
        )
    per_task: dict[str, dict[str, float | int]] = {}
    for task, records in sorted(by_task.items()):
        count = len(records)
        per_task[task] = {
            "count": count,
            "accuracy": sum(bool(record["correct"]) for record in records) / count,
            "supervised_loss": sum(
                float(record["supervised_loss"]) for record in records
            )
            / count,
            "teacher_kl": sum(float(record["teacher_kl"]) for record in records)
            / count,
            "combined_loss": sum(float(record["combined_loss"]) for record in records)
            / count,
        }
    return {
        "macro_accuracy": sum(float(item["accuracy"]) for item in per_task.values())
        / len(per_task),
        "balanced_supervised_loss": sum(
            float(item["supervised_loss"]) for item in per_task.values()
        )
        / len(per_task),
        "balanced_teacher_kl": sum(
            float(item["teacher_kl"]) for item in per_task.values()
        )
        / len(per_task),
        "balanced_combined_loss": sum(
            float(item["combined_loss"]) for item in per_task.values()
        )
        / len(per_task),
        "per_task": per_task,
    }


def _choice_records_by_shape(
    examples: list[dict[str, Any]],
) -> dict[tuple[int, int], list[tuple[int, int, int, torch.Tensor]]]:
    records_by_shape: dict[
        tuple[int, int], list[tuple[int, int, int, torch.Tensor]]
    ] = defaultdict(list)
    for example_index, example in enumerate(examples):
        for choice_index, encoded in enumerate(example["choices"]):
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
            continuation_start = int(encoded["continuation_start"])
            if not 0 < continuation_start < input_ids.numel():
                raise ValueError("cached choice has an invalid continuation boundary")
            records_by_shape[(int(input_ids.numel()), continuation_start)].append(
                (example_index, choice_index, continuation_start, input_ids)
            )
    return records_by_shape


def _pack_task_choice_scores(
    scores: list[list[torch.Tensor | None]],
) -> list[torch.Tensor]:
    packed: list[torch.Tensor] = []
    for example_scores in scores:
        if not example_scores or any(score is None for score in example_scores):
            raise RuntimeError("task batches did not produce every choice score")
        packed.append(
            torch.stack([score for score in example_scores if score is not None])
        )
    return packed


def _score_choice_logits(
    logits: torch.Tensor, input_ids: torch.Tensor, continuation_start: int
) -> torch.Tensor:
    continuation_logits = logits[:, continuation_start - 1 : -1].float()
    targets = input_ids[:, continuation_start:]
    if continuation_logits.shape[:2] != targets.shape or targets.numel() == 0:
        raise RuntimeError("batched continuation/logit alignment failed")
    token_scores = F.log_softmax(continuation_logits, dim=-1).gather(
        2, targets.unsqueeze(2)
    )[:, :, 0]
    return token_scores.mean(dim=1)


@torch.inference_mode()
def score_batched_task_examples(
    model: nn.Module,
    examples: list[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Score choices with the exact shape buckets used by the prefix cache."""
    if batch_size <= 0:
        raise ValueError("task scoring batch size must be positive")
    scores: list[list[torch.Tensor | None]] = [
        [None] * len(example["choices"]) for example in examples
    ]
    records_by_shape = _choice_records_by_shape(examples)
    for shape in sorted(records_by_shape):
        records = records_by_shape[shape]
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            input_ids = torch.stack([record[3] for record in chunk]).to(device)
            continuation_start = chunk[0][2]
            output = model(input_ids=input_ids, use_cache=False)
            choice_scores = _score_choice_logits(
                output.logits, input_ids, continuation_start
            ).cpu()
            for row, (example_index, choice_index, _start, _ids) in enumerate(chunk):
                scores[example_index][choice_index] = choice_scores[row]
    return _pack_task_choice_scores(scores)


@torch.inference_mode()
def build_prefix_task_cache(
    model: nn.Module,
    examples: list[dict[str, Any]],
    *,
    start_layer: int,
    batch_size: int,
    device: torch.device,
) -> PrefixTaskCache:
    records_by_shape = _choice_records_by_shape(examples)
    token_count = sum(
        int(record[3].numel())
        for records in records_by_shape.values()
        for record in records
    )
    choice_count = sum(len(records) for records in records_by_shape.values())

    batches: list[CachedChoiceBatch] = []
    for shape in sorted(records_by_shape):
        records = records_by_shape[shape]
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            input_ids = torch.stack([record[3] for record in chunk]).to(device)
            hidden = model.model.embed_tokens(input_ids)
            for layer_index in range(start_layer):
                hidden = run_hidden_block_output(
                    model,
                    layer_index,
                    get_linear_modules(model, layer_index),
                    hidden,
                    weights=None,
                )
            batches.append(
                CachedChoiceBatch(
                    hidden_states=hidden.to("cpu", torch.bfloat16).contiguous(),
                    input_ids=input_ids.to("cpu"),
                    metadata=tuple(
                        (record[0], record[1], record[2]) for record in chunk
                    ),
                )
            )
            del hidden, input_ids
    return PrefixTaskCache(
        start_layer=start_layer,
        batches=batches,
        example_count=len(examples),
        choice_count=choice_count,
        token_count=token_count,
    )


@torch.inference_mode()
def collect_cached_layer_outputs(
    model: nn.Module,
    cache: PrefixTaskCache,
    *,
    target_layer: int,
    device: torch.device,
) -> list[torch.Tensor]:
    if target_layer < cache.start_layer:
        raise ValueError("target layer precedes the task cache")
    outputs: list[torch.Tensor] = []
    for batch in cache.batches:
        hidden = batch.hidden_states.to(device)
        for layer_index in range(cache.start_layer, target_layer + 1):
            hidden = run_hidden_block_output(
                model,
                layer_index,
                get_linear_modules(model, layer_index),
                hidden,
                weights=None,
            )
        outputs.append(hidden.to("cpu", torch.bfloat16).contiguous())
    return outputs


@torch.inference_mode()
def score_cached_task_cache(
    model: nn.Module,
    cache: PrefixTaskCache,
    *,
    device: torch.device,
    probe_layer: int | None = None,
    reference_outputs: list[torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], float | None]:
    if (probe_layer is None) != (reference_outputs is None):
        raise ValueError("probe layer and reference outputs must be supplied together")
    if reference_outputs is not None and len(reference_outputs) != len(cache.batches):
        raise ValueError("reference outputs do not match task cache batches")
    scores: list[list[torch.Tensor | None]] = [
        list() for _ in range(cache.example_count)
    ]
    squared_error = 0.0
    element_count = 0
    for batch_index, batch in enumerate(cache.batches):
        hidden = batch.hidden_states.to(device)
        for layer_index in range(cache.start_layer, len(model.model.layers)):
            hidden = run_hidden_block_output(
                model,
                layer_index,
                get_linear_modules(model, layer_index),
                hidden,
                weights=None,
            )
            if layer_index == probe_layer:
                assert reference_outputs is not None
                difference = hidden.float() - reference_outputs[batch_index].to(
                    device=device, dtype=torch.float32
                )
                squared_error += float(difference.square().sum())
                element_count += difference.numel()
        hidden = model.model.norm(hidden)
        continuation_start = batch.metadata[0][2]
        if any(metadata[2] != continuation_start for metadata in batch.metadata):
            raise RuntimeError("cached batch mixed continuation boundaries")
        continuation_hidden = hidden[:, continuation_start - 1 : -1]
        logits = model.lm_head(continuation_hidden).float()
        targets = batch.input_ids[:, continuation_start:].to(device)
        if logits.shape[:2] != targets.shape or targets.numel() == 0:
            raise RuntimeError("cached continuation/logit alignment failed")
        token_scores = F.log_softmax(logits, dim=-1).gather(2, targets.unsqueeze(2))[
            :, :, 0
        ]
        choice_scores = token_scores.mean(dim=1).cpu()
        for row, (example_index, choice_index, _start) in enumerate(batch.metadata):
            while len(scores[example_index]) <= choice_index:
                scores[example_index].append(None)
            scores[example_index][choice_index] = choice_scores[row]
        del hidden
    mse = squared_error / element_count if element_count else None
    return _pack_task_choice_scores(scores), mse


def select_cache_parity_indices(
    examples: list[dict[str, Any]], *, count: int
) -> list[int]:
    """Select a deterministic task-balanced parity slice with shape diversity."""
    if count <= 0:
        raise ValueError("cache parity count must be positive")
    if not examples:
        raise ValueError("cache parity requires examples")
    by_task: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        task = str(example["task"])
        if not example.get("choices"):
            raise ValueError("cache parity example has no choices")
        by_task[task].append(index)
    if count < len(by_task):
        raise ValueError("cache parity count must cover every task")

    def shape_key(index: int) -> tuple[int, int, int]:
        choices = examples[index]["choices"]
        return (
            max(len(choice["input_ids"]) for choice in choices),
            min(int(choice["continuation_start"]) for choice in choices),
            index,
        )

    tasks = sorted(by_task)
    for indices in by_task.values():
        indices.sort(key=shape_key)
    target = min(count, len(examples))
    allocations = {task: 0 for task in tasks}
    while sum(allocations.values()) < target:
        progressed = False
        for task in tasks:
            if allocations[task] < len(by_task[task]):
                allocations[task] += 1
                progressed = True
                if sum(allocations.values()) == target:
                    break
        if not progressed:
            break

    selected: list[int] = []
    for task in tasks:
        indices = by_task[task]
        allocation = allocations[task]
        if allocation == 1:
            selected.append(indices[len(indices) // 2])
        elif allocation > 1:
            positions = [
                round(step * (len(indices) - 1) / (allocation - 1))
                for step in range(allocation)
            ]
            selected.extend(indices[position] for position in positions)
    return sorted(selected)


@torch.inference_mode()
def _full_scores_for_cache_indices(
    model: nn.Module,
    cache: PrefixTaskCache,
    examples: list[dict[str, Any]],
    *,
    indices: list[int],
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Run full HF forward with the cache's exact batches for parity."""
    selected = set(indices)
    scores: dict[int, list[torch.Tensor | None]] = {
        index: [None] * len(examples[index]["choices"]) for index in indices
    }
    for batch in cache.batches:
        if not any(
            example_index in selected
            for example_index, _choice, _start in batch.metadata
        ):
            continue
        input_ids = batch.input_ids.to(device)
        continuation_start = batch.metadata[0][2]
        output = model(input_ids=input_ids, use_cache=False)
        choice_scores = _score_choice_logits(
            output.logits, input_ids, continuation_start
        ).cpu()
        for row, (example_index, choice_index, _start) in enumerate(batch.metadata):
            if example_index in selected:
                scores[example_index][choice_index] = choice_scores[row]
    packed: dict[int, torch.Tensor] = {}
    for index, example_scores in scores.items():
        if any(score is None for score in example_scores):
            raise RuntimeError("cache parity reference missed a selected choice")
        packed[index] = torch.stack(
            [score for score in example_scores if score is not None]
        )
    return packed


def _cache_parity(
    model: nn.Module,
    cache: PrefixTaskCache,
    cache_scores: list[torch.Tensor],
    examples: list[dict[str, Any]],
    *,
    count: int,
    tolerance: float,
    device: torch.device,
) -> dict[str, Any]:
    indices = select_cache_parity_indices(examples, count=count)
    reference_scores = _full_scores_for_cache_indices(
        model,
        cache,
        examples,
        indices=indices,
        device=device,
    )
    checked = len(indices)
    maximum = 0.0
    label_matches = 0
    task_counts: dict[str, int] = defaultdict(int)
    total_lengths: list[int] = []
    continuation_starts: list[int] = []
    for index in indices:
        full_scores = reference_scores[index]
        cached = cache_scores[index].cpu()
        maximum = max(maximum, float((full_scores - cached).abs().max()))
        label_matches += int(full_scores.argmax() == cached.argmax())
        task_counts[str(examples[index]["task"])] += 1
        total_lengths.extend(
            len(choice["input_ids"]) for choice in examples[index]["choices"]
        )
        continuation_starts.extend(
            int(choice["continuation_start"]) for choice in examples[index]["choices"]
        )
    passed = maximum <= tolerance and label_matches == checked
    return {
        "checked_examples": checked,
        "indices": indices,
        "task_counts": dict(sorted(task_counts.items())),
        "total_length_range": [min(total_lengths), max(total_lengths)],
        "continuation_start_range": [
            min(continuation_starts),
            max(continuation_starts),
        ],
        "reference_mode": "same_full_forward_batches",
        "max_abs_choice_score_error": maximum,
        "argmax_matches": label_matches,
        "tolerance": tolerance,
        "passed": passed,
    }


def _build_curvature_bundles(
    model: nn.Module,
    checkpoint: dict[str, Any],
    gradients: dict[tuple[int, str], list[torch.Tensor]],
    activation_second_moment: dict[tuple[int, str], torch.Tensor],
    *,
    layers: list[int],
    local_candidates: int,
    candidate_chunk: int,
    bundle_size: int,
    device: torch.device,
) -> tuple[dict[tuple[int, str], RankedCandidates], dict[str, Any], int]:
    bundles: dict[tuple[int, str], RankedCandidates] = {}
    proposal_summary: dict[str, Any] = {}
    total_vectors = 0
    for layer in layers:
        for name in LINEAR_NAMES:
            key = (layer, name)
            state = checkpoint["layers"][str(layer)][name]
            codebook = state["codebook"].to(device=device, dtype=torch.float32)
            assignments = state["assignments"].to(device=device, dtype=torch.long)
            neighbors = build_codebook_neighbor_table(
                codebook, candidates=min(local_candidates, codebook.shape[0])
            )
            views: list[torch.Tensor] = []
            vector_scales: torch.Tensor | None = None
            for gradient in gradients.pop(key):
                vector_gradient, current_scales = _vector_gradient_geometry(
                    state, gradient, device=device
                )
                views.append(vector_gradient)
                if vector_scales is None:
                    vector_scales = current_scales
            assert vector_scales is not None
            rows, columns = (int(value) for value in state["shape"])
            rankings = rank_scale_conditioned_candidates(
                assignments=assignments,
                codebook=codebook,
                neighbor_table=neighbors,
                gradient_vector_views=views,
                vector_scales=vector_scales,
                activation_second_moment=activation_second_moment.pop(key).to(device),
                rows=rows,
                columns=columns,
                chunk_size=candidate_chunk,
                retain=bundle_size,
            )
            bundles[key] = rankings["curvature_normalized"].prefix(bundle_size)
            proposal_summary[f"{layer}:{name}"] = {
                "vectors": int(assignments.numel()),
                **{
                    method: _ranking_summary(ranking)
                    for method, ranking in rankings.items()
                },
            }
            total_vectors += int(assignments.numel())
            print(
                json.dumps(
                    {
                        "conditional_bundle": f"{layer}:{name}",
                        **proposal_summary[f"{layer}:{name}"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            del codebook, assignments, neighbors, views, vector_scales, rankings
            torch.cuda.empty_cache()
    if gradients or activation_second_moment:
        raise RuntimeError("conditional bundle geometry contains unconsumed states")
    return bundles, proposal_summary, total_vectors


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
    _validate_args(args, model_layers=len(model.model.layers))

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
    print("Collecting batch-matched dense-teacher task scores", flush=True)
    train_teacher_scores = score_batched_task_examples(
        model,
        examples["train"],
        batch_size=args.cache_batch_size,
        device=device,
    )
    for example, scores in zip(examples["train"], train_teacher_scores, strict=True):
        example["teacher_scores"] = scores.cpu()
    print("Collecting dense-teacher text distributions", flush=True)
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

    print("Building full task-training prefix hidden cache", flush=True)
    task_cache = build_prefix_task_cache(
        model,
        examples["train"],
        start_layer=args.cache_start_layer,
        batch_size=args.cache_batch_size,
        device=device,
    )
    source_scores, _unused_mse = score_cached_task_cache(
        model, task_cache, device=device
    )
    cache_parity = _cache_parity(
        model,
        task_cache,
        source_scores,
        examples["train"],
        count=args.cache_parity_examples,
        tolerance=args.cache_parity_tolerance,
        device=device,
    )
    if not cache_parity["passed"]:
        raise RuntimeError(f"prefix task cache failed parity: {cache_parity}")
    incumbent_objective = summarize_cached_task_scores(
        source_scores,
        examples["train"],
        teacher_kl_weight=args.teacher_kl_weight,
        teacher_temperature=args.teacher_temperature,
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
    bundles, proposal_summary, total_vectors = _build_curvature_bundles(
        model,
        checkpoint,
        gradients,
        activation_second_moment,
        layers=list(args.layers),
        local_candidates=args.local_candidates,
        candidate_chunk=args.candidate_chunk,
        bundle_size=args.bundle_size,
        device=device,
    )

    applicator = SparseAssignmentApplicator(
        model, checkpoint, layers=list(args.layers), device=device
    )
    accepted: dict[tuple[int, str], RankedCandidates] = {}
    layer_steps: list[dict[str, Any]] = []
    progress: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment": "set_conditional_hard_assignment_gain",
        "completed": False,
        "arguments": vars(args),
        "data_contract": {
            "task_split_manifest": split_manifest,
            "formal_test_data_used": False,
        },
        "cache_contract": {
            "start_layer": task_cache.start_layer,
            "batches": len(task_cache.batches),
            "examples": task_cache.example_count,
            "choices": task_cache.choice_count,
            "tokens": task_cache.token_count,
            "parity": cache_parity,
        },
        "baseline_train_objective": incumbent_objective,
        "baseline_validation": baseline_validation,
        "proposal": proposal_summary,
        "layer_steps": layer_steps,
    }
    _atomic_json(output_summary, progress)

    for layer in args.layers:
        applicator.apply(accepted)
        reference_outputs = collect_cached_layer_outputs(
            model, task_cache, target_layer=layer, device=device
        )
        candidate_records: list[dict[str, Any]] = []
        for name in LINEAR_NAMES:
            key = (layer, name)
            candidate_state = dict(accepted)
            candidate_state[key] = bundles[key]
            applicator.apply(candidate_state)
            scores, block_mse = score_cached_task_cache(
                model,
                task_cache,
                device=device,
                probe_layer=layer,
                reference_outputs=reference_outputs,
            )
            objective = summarize_cached_task_scores(
                scores,
                examples["train"],
                teacher_kl_weight=args.teacher_kl_weight,
                teacher_temperature=args.teacher_temperature,
            )
            record = {
                "layer": layer,
                "linear": name,
                "bundle_size": args.bundle_size,
                "conditioned_on": [f"{k[0]}:{k[1]}" for k in accepted],
                "prediction": _selection_prediction({key: bundles[key]}),
                "block_output_mse_vs_incumbent": block_mse,
                "train_objective": objective,
                "conditional_gain": (
                    float(incumbent_objective["balanced_combined_loss"])
                    - float(objective["balanced_combined_loss"])
                ),
            }
            candidate_records.append(record)
            print(
                json.dumps({"conditional_candidate": record}, ensure_ascii=False),
                flush=True,
            )
        decision = accept_conditional_bundle(
            incumbent_loss=float(incumbent_objective["balanced_combined_loss"]),
            candidate_losses={
                str(record["linear"]): float(
                    record["train_objective"]["balanced_combined_loss"]
                )
                for record in candidate_records
            },
            tolerance=args.conditional_tolerance,
        )
        if decision["accepted"]:
            selected_name = str(decision["key"])
            selected_record = next(
                record
                for record in candidate_records
                if record["linear"] == selected_name
            )
            accepted[(layer, selected_name)] = bundles[(layer, selected_name)]
            incumbent_objective = selected_record["train_objective"]
        curvature_costs = [
            float(record["prediction"]["curvature_cost_sum"])
            for record in candidate_records
        ]
        block_mses = [
            float(record["block_output_mse_vs_incumbent"])
            for record in candidate_records
        ]
        step = {
            "layer": layer,
            "incumbent_before": candidate_records[0]["conditioned_on"],
            "candidates": candidate_records,
            "decision": decision,
            "positive_control": {
                "quantity": "bundle curvature cost vs measured immediate block-output MSE",
                "spearman": spearman_rank_correlation(curvature_costs, block_mses),
                "count": len(candidate_records),
            },
        }
        layer_steps.append(step)
        progress["layer_steps"] = layer_steps
        progress["accepted_bundles"] = [f"{key[0]}:{key[1]}" for key in accepted]
        progress["current_train_objective"] = incumbent_objective
        _atomic_json(output_summary, progress)
        applicator.apply(accepted)
        del reference_outputs
        torch.cuda.empty_cache()

    applicator.apply(accepted)
    switched = sum(selection.vector_indices.numel() for selection in accepted.values())
    if switched:
        selected_validation = _evaluate_examples(
            model, examples["validation"], device=device
        )
        task_gate_passed = choose_task_loss_dominant_snapshot(
            baseline_validation,
            selected_validation,
            min_relative_loss_improvement=args.selection_min_relative_loss_improvement,
            max_macro_regression=args.selection_max_macro_regression,
            max_per_task_loss_regression=args.selection_max_per_task_loss_regression,
        )
        if task_gate_passed:
            selected_validation["text_guard_cross_entropy"] = _evaluate_text_guard(
                model, text_examples["validation"], device=device
            )
        else:
            selected_validation["text_guard_cross_entropy"] = None
        validation_gate_passed = task_gate_passed and choose_loss_dominant_snapshot(
            baseline_validation,
            selected_validation,
            min_relative_loss_improvement=args.selection_min_relative_loss_improvement,
            max_macro_regression=args.selection_max_macro_regression,
            max_per_task_loss_regression=args.selection_max_per_task_loss_regression,
            max_text_loss_regression=args.text_max_relative_regression,
        )
    else:
        selected_validation = copy.deepcopy(baseline_validation)
        task_gate_passed = False
        validation_gate_passed = False

    baseline_audit: dict[str, Any] | None = None
    candidate_audit: dict[str, Any] | None = None
    audit_gate_passed = False
    if validation_gate_passed:
        applicator.restore()
        baseline_audit = _evaluate_examples(model, examples["audit"], device=device)
        baseline_audit["text_guard_cross_entropy"] = _evaluate_text_guard(
            model, text_examples["audit"], device=device
        )
        applicator.apply(accepted)
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

    checkpoint_written = False
    if audit_gate_passed:
        if not args.output_checkpoint:
            raise ValueError(
                "accepted set-conditional endpoint requires --output-checkpoint"
            )
        _checkpoint_with_selection(
            copy.deepcopy(checkpoint),
            accepted,
            metadata={
                "method": "causal_one_pass_linear_bundle_conditional_gain",
                "bundle_size": args.bundle_size,
                "accepted_bundles": [f"{key[0]}:{key[1]}" for key in accepted],
                "train_objective": incumbent_objective,
                "validation": selected_validation,
                "audit": candidate_audit,
            },
            output=Path(args.output_checkpoint),
        )
        checkpoint_written = True
    applicator.restore()

    peak_memory = int(torch.cuda.max_memory_allocated(device))
    summary = {
        **progress,
        "completed": True,
        "method": {
            "coordinate": "one fixed curvature-ranked assignment bundle per Linear",
            "search": "causal one-pass, seven conditionally evaluated coordinates per block",
            "acceptance": "strict decrease of full task-training supervised plus teacher-KL objective",
            "trained_parameters": 0,
            "logical_bitrate_changed": False,
        },
        "total_vectors": total_vectors,
        "accepted_bundles": [f"{key[0]}:{key[1]}" for key in accepted],
        "accepted_switches": int(switched),
        "accepted_switch_rate": int(switched) / total_vectors,
        "selected_train_objective": incumbent_objective,
        "selected_validation": selected_validation,
        "task_gate_passed": task_gate_passed,
        "validation_gate_passed": validation_gate_passed,
        "baseline_audit": baseline_audit,
        "candidate_audit": candidate_audit,
        "audit_gate_passed": audit_gate_passed,
        "checkpoint_written": checkpoint_written,
        "output_checkpoint": args.output_checkpoint if checkpoint_written else None,
        "formal_test_data_used": False,
        "elapsed_seconds": time.time() - started,
        "gpu_peak_memory_bytes": peak_memory,
    }
    _atomic_json(output_summary, summary)
    print(
        json.dumps({"set_conditional_summary": summary}, ensure_ascii=False), flush=True
    )


if __name__ == "__main__":
    main()
