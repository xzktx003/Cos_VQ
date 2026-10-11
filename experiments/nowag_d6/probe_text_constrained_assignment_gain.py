"""Search assignment bundles inside an exact text-preserving feasible set.

V13 showed that task-only set-conditional gains can improve held-out task loss
while consuming language-model preservation margin.  This experiment keeps the
same fixed curvature bundles, but changes the optimization problem: full
text-training teacher cross entropy defines feasibility, and task-training
conditional loss selects only among feasible hard bundles.  Validation and
audit remain unseen until the complete coordinate search has terminated.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
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
)
from experiments.nowag_d6.calibrate_task_local_assignments import (
    _load_logical_weights,
    _proposal_gradients,
    choose_loss_dominant_snapshot,
    choose_task_loss_dominant_snapshot,
)
from experiments.nowag_d6.diagnose_block_vector_gsq import (
    LINEAR_NAMES,
    get_linear_modules,
)
from experiments.nowag_d6.hard_scale_compensation import (
    CompensationInfeasibleError,
    HardScaleCompensatedApplicator,
    apply_compensation_to_checkpoint,
    audit_fresh_compensated_checkpoint,
)
from experiments.nowag_d6.probe_scale_conditioned_assignment_curvature import (
    RankedCandidates,
    SparseAssignmentApplicator,
    _checkpoint_with_selection,
    _selection_prediction,
)
from experiments.nowag_d6.probe_set_conditional_assignment_gain import (
    _build_curvature_bundles,
    _cache_parity,
    build_prefix_task_cache,
    collect_cached_layer_outputs,
    score_batched_task_examples,
    score_cached_task_cache,
    spearman_rank_correlation,
    summarize_cached_task_scores,
)
from experiments.nowag_d6.run_prefix_vector_gsq import run_hidden_block_output


@dataclass
class CachedTextBatch:
    hidden_states: torch.Tensor
    input_ids: torch.Tensor
    teacher_indices: torch.Tensor
    teacher_probabilities: torch.Tensor
    rows: tuple[int, ...]


@dataclass
class PrefixTextCache:
    start_layer: int
    batches: list[CachedTextBatch]
    example_count: int
    token_count: int
    hidden_bytes: int


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
    parser.add_argument("--text-train-max-relative-regression", type=float, default=0.0)
    parser.add_argument("--hard-scale-compensation", action="store_true")
    parser.add_argument("--text-cache-start-layer", type=int, default=28)
    parser.add_argument("--text-cache-batch-size", type=int, default=1)
    parser.add_argument("--text-cache-parity-examples", type=int, default=8)
    parser.add_argument("--text-cache-parity-tolerance", type=float, default=0.001)
    parser.add_argument("--text-lm-head-vocab-chunk", type=int, default=8192)
    parser.add_argument("--text-cache-max-gib", type=float, default=160.0)
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
        args.text_cache_batch_size,
        args.text_cache_parity_examples,
        args.text_lm_head_vocab_chunk,
        args.text_cache_max_gib,
    )
    if not all(math.isfinite(value) and value > 0 for value in positive):
        raise ValueError("counts, chunks, cache limits, and sequence lengths must be positive")
    nonnegative = (
        args.task_split_offset,
        args.task_audit_offset,
        args.conditional_tolerance,
        args.cache_parity_tolerance,
        args.teacher_kl_weight,
        args.text_guard_weight,
        args.text_train_max_relative_regression,
        args.text_cache_parity_tolerance,
        args.text_max_relative_regression,
        args.selection_min_relative_loss_improvement,
        args.selection_max_macro_regression,
        args.selection_max_per_task_loss_regression,
    )
    if not all(math.isfinite(value) and value >= 0 for value in nonnegative):
        raise ValueError("offsets, weights, and tolerances must be finite non-negative")
    if not math.isfinite(args.teacher_temperature) or args.teacher_temperature <= 0:
        raise ValueError("teacher temperature must be finite positive")
    if len(args.layers) != len(set(args.layers)) or args.layers != sorted(args.layers):
        raise ValueError("layers must be unique and sorted")
    if set(args.tasks) != set(DEFAULT_TASKS):
        raise ValueError(f"tasks must be exactly {list(DEFAULT_TASKS)}")
    if args.cache_start_layer != min(args.layers):
        raise ValueError("task cache start must equal the first searched layer")
    if args.text_cache_start_layer != min(args.layers):
        raise ValueError("text cache start must equal the first searched layer")
    if args.text_cache_batch_size != 1:
        raise ValueError("exact candidate-batched text scoring currently requires batch size 1")
    flat_rows = [
        *args.text_train_rows,
        *args.text_validation_rows,
        *args.text_audit_rows,
    ]
    if len(flat_rows) != len(set(flat_rows)):
        raise ValueError("text train/validation/audit rows must be disjoint")
    if model_layers is not None:
        if not 0 <= args.cache_start_layer < model_layers:
            raise ValueError("task cache start layer is outside the model")
        if not 0 <= args.text_cache_start_layer < model_layers:
            raise ValueError("text cache start layer is outside the model")


def estimate_prefix_text_cache_bytes(
    *, sequences: int, sequence_length: int, hidden_size: int, dtype_bytes: int
) -> int:
    values = (sequences, sequence_length, hidden_size, dtype_bytes)
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("text cache dimensions and dtype size must be positive integers")
    return sequences * sequence_length * hidden_size * dtype_bytes


def select_text_cache_parity_indices(
    *, example_count: int, count: int
) -> list[int]:
    if example_count <= 0 or count <= 0:
        raise ValueError("text parity sizes must be positive")
    if count > example_count:
        raise ValueError("text parity count cannot exceed the cache")
    if count == 1:
        return [0]
    indices = [round(index * (example_count - 1) / (count - 1)) for index in range(count)]
    if len(indices) != len(set(indices)):
        raise RuntimeError("text parity selection produced duplicate indices")
    return indices


def choose_text_constrained_bundle(
    *,
    incumbent_task_loss: float,
    incumbent_text_loss: float,
    candidates: dict[Hashable, dict[str, float | str | None]],
    task_tolerance: float,
    text_tolerance: float,
) -> dict[str, Any]:
    scalars = (
        incumbent_task_loss,
        incumbent_text_loss,
        task_tolerance,
        text_tolerance,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("incumbent losses and tolerances must be finite")
    if incumbent_task_loss <= 0 or incumbent_text_loss <= 0:
        raise ValueError("incumbent losses must be positive")
    if task_tolerance < 0 or text_tolerance < 0:
        raise ValueError("acceptance tolerances must be non-negative")
    if not candidates:
        raise ValueError("text-constrained decision requires candidates")

    feasible: list[tuple[Hashable, float, float]] = []
    rejections: dict[str, str] = {}
    for key, metrics in candidates.items():
        infeasible_reason = metrics.get("infeasible_reason")
        if infeasible_reason is not None:
            rejections[str(key)] = str(infeasible_reason)
            continue
        task_loss = metrics.get("task_loss")
        text_loss = metrics.get("text_loss")
        if task_loss is None or not math.isfinite(float(task_loss)):
            raise ValueError("every candidate requires a finite task loss")
        task_value = float(task_loss)
        if task_value + task_tolerance >= incumbent_task_loss:
            rejections[str(key)] = "task_non_improving"
            continue
        if text_loss is None or not math.isfinite(float(text_loss)):
            raise ValueError("every task-improving candidate requires a finite text loss")
        text_value = float(text_loss)
        if text_value > incumbent_text_loss + text_tolerance:
            rejections[str(key)] = "text_constraint"
            continue
        feasible.append((key, task_value, text_value))

    feasible.sort(key=lambda item: (item[1], str(item[0])))
    if not feasible:
        return {
            "accepted": False,
            "key": None,
            "task_loss": incumbent_task_loss,
            "text_loss": incumbent_text_loss,
            "task_gain": 0.0,
            "text_change": 0.0,
            "feasible_keys": [],
            "rejections": dict(sorted(rejections.items())),
        }
    key, task_loss, text_loss = feasible[0]
    return {
        "accepted": True,
        "key": key,
        "task_loss": task_loss,
        "text_loss": text_loss,
        "task_gain": incumbent_task_loss - task_loss,
        "text_change": text_loss - incumbent_text_loss,
        "feasible_keys": [str(item[0]) for item in feasible],
        "rejections": dict(sorted(rejections.items())),
    }


def chunked_topk_teacher_cross_entropy(
    hidden_states: torch.Tensor,
    output_weight: torch.Tensor,
    teacher_indices: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    *,
    vocab_chunk_size: int,
) -> torch.Tensor:
    """Compute exact top-k-teacher CE without materializing full-vocab logits."""
    if hidden_states.ndim != 3 or output_weight.ndim != 2:
        raise ValueError("hidden states and output weight must be rank 3 and 2")
    if hidden_states.shape[-1] != output_weight.shape[-1]:
        raise ValueError("hidden and output weight dimensions do not match")
    if teacher_indices.shape != teacher_probabilities.shape:
        raise ValueError("teacher indices and probabilities must match")
    if teacher_indices.ndim != 3 or teacher_indices.shape[:2] != hidden_states.shape[:2]:
        raise ValueError("teacher tensors must cover every batch and token")
    if vocab_chunk_size <= 0:
        raise ValueError("vocabulary chunk size must be positive")
    vocabulary = int(output_weight.shape[0])
    if int(teacher_indices.min()) < 0 or int(teacher_indices.max()) >= vocabulary:
        raise ValueError("teacher index is outside the vocabulary")

    probabilities = teacher_probabilities.to(
        device=hidden_states.device, dtype=torch.float32
    )
    probabilities = probabilities / probabilities.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    indices = teacher_indices.to(device=hidden_states.device, dtype=torch.long)
    running_logsumexp = torch.full(
        hidden_states.shape[:2],
        -torch.inf,
        device=hidden_states.device,
        dtype=torch.float32,
    )
    expected_teacher_logit = torch.zeros_like(running_logsumexp)
    for start in range(0, vocabulary, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocabulary)
        logits = F.linear(hidden_states, output_weight[start:end]).float()
        running_logsumexp = torch.logaddexp(
            running_logsumexp, torch.logsumexp(logits, dim=-1)
        )
        in_chunk = (indices >= start) & (indices < end)
        local_indices = (indices - start).clamp(0, end - start - 1)
        selected = logits.gather(2, local_indices)
        expected_teacher_logit.add_(
            (probabilities * selected * in_chunk).sum(dim=-1)
        )
        del logits, selected
    return (running_logsumexp - expected_teacher_logit).mean(dim=-1)


@torch.inference_mode()
def build_prefix_text_cache(
    model: nn.Module,
    examples: list[dict[str, torch.Tensor | int]],
    *,
    start_layer: int,
    batch_size: int,
    max_bytes: int,
    device: torch.device,
) -> PrefixTextCache:
    if batch_size != 1:
        raise ValueError("candidate-batched exact text cache requires batch size 1")
    if not examples:
        raise ValueError("text prefix cache requires examples")
    first_ids = examples[0].get("input_ids")
    if not isinstance(first_ids, torch.Tensor) or first_ids.ndim != 1:
        raise ValueError("text examples require one-dimensional input IDs")
    expected_bytes = estimate_prefix_text_cache_bytes(
        sequences=len(examples),
        sequence_length=int(first_ids.numel()),
        hidden_size=int(model.config.hidden_size),
        dtype_bytes=torch.tensor([], dtype=torch.bfloat16).element_size(),
    )
    if expected_bytes > max_bytes:
        raise RuntimeError(
            f"text prefix cache needs {expected_bytes} bytes, exceeds {max_bytes}"
        )

    batches: list[CachedTextBatch] = []
    hidden_bytes = 0
    token_count = 0
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        input_ids = torch.stack(
            [example["input_ids"] for example in chunk]  # type: ignore[list-item]
        ).to(device)
        hidden = model.model.embed_tokens(input_ids)
        for layer_index in range(start_layer):
            hidden = run_hidden_block_output(
                model,
                layer_index,
                get_linear_modules(model, layer_index),
                hidden,
                weights=None,
            )
        cached_hidden = hidden.to("cpu", torch.bfloat16).contiguous()
        teacher_indices = torch.stack(
            [example["teacher_indices"] for example in chunk]  # type: ignore[list-item]
        ).contiguous()
        teacher_probabilities = torch.stack(
            [example["teacher_probabilities"] for example in chunk]  # type: ignore[list-item]
        ).contiguous()
        rows = tuple(int(example["row"]) for example in chunk)
        cached_ids = input_ids.to("cpu").contiguous()
        batches.append(
            CachedTextBatch(
                hidden_states=cached_hidden,
                input_ids=cached_ids,
                teacher_indices=teacher_indices,
                teacher_probabilities=teacher_probabilities,
                rows=rows,
            )
        )
        hidden_bytes += cached_hidden.numel() * cached_hidden.element_size()
        token_count += cached_ids.numel()
        del hidden, cached_hidden, input_ids
    if hidden_bytes != expected_bytes:
        raise RuntimeError("materialized text prefix cache size differs from contract")
    return PrefixTextCache(
        start_layer=start_layer,
        batches=batches,
        example_count=len(examples),
        token_count=token_count,
        hidden_bytes=hidden_bytes,
    )


def _normalized_suffix_hidden(
    model: nn.Module,
    hidden: torch.Tensor,
    *,
    start_layer: int,
    device: torch.device,
    first_layer_weights: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    current = hidden.to(device)
    for layer_index in range(start_layer, len(model.model.layers)):
        current = run_hidden_block_output(
            model,
            layer_index,
            get_linear_modules(model, layer_index),
            current,
            weights=first_layer_weights if layer_index == start_layer else None,
        )
    return model.model.norm(current)


@torch.inference_mode()
def score_cached_text_state(
    model: nn.Module,
    cache: PrefixTextCache,
    *,
    vocab_chunk_size: int,
    device: torch.device,
    batch_indices: list[int] | None = None,
) -> tuple[float, list[float]]:
    selected = batch_indices if batch_indices is not None else list(range(len(cache.batches)))
    if not selected:
        raise ValueError("cached text scoring requires batches")
    losses: list[float] = []
    for batch_index in selected:
        batch = cache.batches[batch_index]
        hidden = _normalized_suffix_hidden(
            model,
            batch.hidden_states,
            start_layer=cache.start_layer,
            device=device,
        )
        batch_losses = chunked_topk_teacher_cross_entropy(
            hidden[:, :-1],
            model.lm_head.weight,
            batch.teacher_indices.to(device),
            batch.teacher_probabilities.to(device),
            vocab_chunk_size=vocab_chunk_size,
        )
        losses.extend(float(value) for value in batch_losses.cpu())
        del hidden, batch_losses
    return sum(losses) / len(losses), losses


@torch.inference_mode()
def text_cache_parity(
    model: nn.Module,
    cache: PrefixTextCache,
    *,
    count: int,
    tolerance: float,
    vocab_chunk_size: int,
    device: torch.device,
) -> dict[str, Any]:
    indices = select_text_cache_parity_indices(
        example_count=cache.example_count, count=count
    )
    _mean, cached_losses = score_cached_text_state(
        model,
        cache,
        vocab_chunk_size=vocab_chunk_size,
        device=device,
        batch_indices=indices,
    )
    full_losses: list[float] = []
    for batch_index in indices:
        batch = cache.batches[batch_index]
        input_ids = batch.input_ids.to(device)
        hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        loss = chunked_topk_teacher_cross_entropy(
            hidden[:, :-1],
            model.lm_head.weight,
            batch.teacher_indices.to(device),
            batch.teacher_probabilities.to(device),
            vocab_chunk_size=vocab_chunk_size,
        )
        full_losses.extend(float(value) for value in loss.cpu())
        del hidden, loss, input_ids
    differences = [
        abs(cached - full)
        for cached, full in zip(cached_losses, full_losses, strict=True)
    ]
    maximum = max(differences)
    return {
        "checked_examples": len(indices),
        "indices": indices,
        "rows": [cache.batches[index].rows[0] for index in indices],
        "reference_mode": "same_batch_full_hf_transformer_with_chunked_exact_lm_head",
        "max_abs_cross_entropy_error": maximum,
        "mean_abs_cross_entropy_error": sum(differences) / len(differences),
        "tolerance": tolerance,
        "passed": maximum <= tolerance,
    }


@torch.no_grad()
def materialize_candidate_weight_overrides(
    model: nn.Module,
    applicator: SparseAssignmentApplicator | HardScaleCompensatedApplicator,
    accepted: dict[tuple[int, str], RankedCandidates],
    bundles: dict[tuple[int, str], RankedCandidates],
    keys: list[tuple[int, str]],
) -> dict[tuple[int, str], dict[str, torch.Tensor]]:
    overrides: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for key in keys:
        candidate_state = dict(accepted)
        candidate_state[key] = bundles[key]
        applicator.apply(candidate_state)
        layer, name = key
        overrides[key] = {
            name: get_linear_modules(model, layer)[name].weight.detach().clone()
        }
    applicator.apply(accepted)
    return overrides


def _checkpoint_with_hard_scale_compensation(
    checkpoint: dict[str, Any],
    selections: dict[tuple[int, str], RankedCandidates],
    applicator: HardScaleCompensatedApplicator,
    *,
    metadata: dict[str, Any],
    output: Path,
) -> None:
    compensations = applicator.selected_compensations(selections)
    apply_compensation_to_checkpoint(checkpoint, selections, compensations)
    checkpoint["hard_scale_compensated_assignment"] = metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, output)


@torch.inference_mode()
def score_cached_text_candidates(
    model: nn.Module,
    cache: PrefixTextCache,
    *,
    layer: int,
    candidate_overrides: dict[tuple[int, str], dict[str, torch.Tensor]],
    vocab_chunk_size: int,
    device: torch.device,
) -> dict[tuple[int, str], float]:
    keys = list(candidate_overrides)
    if not keys or any(key[0] != layer for key in keys):
        raise ValueError("text candidate scoring requires one non-empty layer")
    sums = torch.zeros(len(keys), dtype=torch.float64)
    example_count = 0
    for batch in cache.batches:
        if batch.hidden_states.shape[0] != 1:
            raise RuntimeError("candidate-batched text scoring requires cache batch 1")
        common = batch.hidden_states.to(device)
        for layer_index in range(cache.start_layer, layer):
            common = run_hidden_block_output(
                model,
                layer_index,
                get_linear_modules(model, layer_index),
                common,
                weights=None,
            )
        candidate_hidden = [
            _normalized_suffix_hidden(
                model,
                common,
                start_layer=layer,
                first_layer_weights=candidate_overrides[key],
                device=device,
            )
            for key in keys
        ]
        stacked_hidden = torch.cat(candidate_hidden, dim=0)
        teacher_indices = batch.teacher_indices.to(device).repeat(len(keys), 1, 1)
        teacher_probabilities = batch.teacher_probabilities.to(device).repeat(
            len(keys), 1, 1
        )
        losses = chunked_topk_teacher_cross_entropy(
            stacked_hidden[:, :-1],
            model.lm_head.weight,
            teacher_indices,
            teacher_probabilities,
            vocab_chunk_size=vocab_chunk_size,
        )
        sums.add_(losses.double().cpu())
        example_count += 1
        del common, candidate_hidden, stacked_hidden, losses
    return {key: float(sums[index] / example_count) for index, key in enumerate(keys)}


def build_deferred_text_teacher_split(
    model: nn.Module,
    checkpoint: dict[str, Any],
    applicator: SparseAssignmentApplicator,
    accepted: dict[tuple[int, str], RankedCandidates],
    *,
    model_path: str,
    token_cache: str,
    split: str,
    rows: list[int],
    sequence_length: int,
    teacher_topk: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor | int]]:
    """Materialize a held-out teacher split with one GPU-resident model at a time."""
    if split not in {"validation", "audit"} or not rows:
        raise ValueError("deferred teacher requires a non-empty held-out split")
    applicator.restore()
    model.to("cpu")
    torch.cuda.empty_cache()
    dense_teacher = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    dense_teacher.eval()
    dense_teacher.config.use_cache = False
    rows_by_split = {"train": [], "validation": [], "audit": []}
    rows_by_split[split] = rows
    examples = _build_text_guard_examples(
        dense_teacher,
        token_cache=token_cache,
        rows_by_split=rows_by_split,
        sequence_length=sequence_length,
        teacher_topk=teacher_topk,
        device=device,
    )[split]
    del dense_teacher
    torch.cuda.empty_cache()
    model.to(device)
    model.eval()
    model.config.use_cache = False
    _load_logical_weights(model, checkpoint, device=device)
    applicator.apply(accepted)
    return examples


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
    text_train_examples = _build_text_guard_examples(
        model,
        token_cache=args.text_token_cache,
        rows_by_split={
            "train": list(args.text_train_rows),
            "validation": [],
            "audit": [],
        },
        sequence_length=args.text_sequence_length,
        teacher_topk=args.text_teacher_topk,
        device=device,
    )["train"]

    _load_logical_weights(model, checkpoint, device=device)

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
    task_cache_parity = _cache_parity(
        model,
        task_cache,
        source_scores,
        examples["train"],
        count=args.cache_parity_examples,
        tolerance=args.cache_parity_tolerance,
        device=device,
    )
    if not task_cache_parity["passed"]:
        raise RuntimeError(f"task prefix cache failed parity: {task_cache_parity}")
    incumbent_task_objective = summarize_cached_task_scores(
        source_scores,
        examples["train"],
        teacher_kl_weight=args.teacher_kl_weight,
        teacher_temperature=args.teacher_temperature,
    )

    print("Collecting fixed four-view proposal gradients and curvature", flush=True)
    gradients, activation_second_moment = _proposal_gradients(
        model,
        examples["train"],
        text_train_examples,
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

    expected_text_cache_bytes = estimate_prefix_text_cache_bytes(
        sequences=len(text_train_examples),
        sequence_length=args.text_sequence_length,
        hidden_size=int(model.config.hidden_size),
        dtype_bytes=2,
    )
    maximum_text_cache_bytes = int(args.text_cache_max_gib * 1024**3)
    print(
        f"Building exact full text-training layer-{args.text_cache_start_layer} "
        f"prefix cache ({expected_text_cache_bytes / 1024**3:.3f} GiB)",
        flush=True,
    )
    text_cache = build_prefix_text_cache(
        model,
        text_train_examples,
        start_layer=args.text_cache_start_layer,
        batch_size=args.text_cache_batch_size,
        max_bytes=maximum_text_cache_bytes,
        device=device,
    )
    text_cache_parity_result = text_cache_parity(
        model,
        text_cache,
        count=args.text_cache_parity_examples,
        tolerance=args.text_cache_parity_tolerance,
        vocab_chunk_size=args.text_lm_head_vocab_chunk,
        device=device,
    )
    if not text_cache_parity_result["passed"]:
        raise RuntimeError(f"text prefix cache failed parity: {text_cache_parity_result}")
    incumbent_text_loss, _source_text_losses = score_cached_text_state(
        model,
        text_cache,
        vocab_chunk_size=args.text_lm_head_vocab_chunk,
        device=device,
    )

    applicator: SparseAssignmentApplicator | HardScaleCompensatedApplicator
    if args.hard_scale_compensation:
        applicator = HardScaleCompensatedApplicator(
            model, checkpoint, layers=list(args.layers), device=device
        )
    else:
        applicator = SparseAssignmentApplicator(
            model, checkpoint, layers=list(args.layers), device=device
        )
    accepted: dict[tuple[int, str], RankedCandidates] = {}
    layer_steps: list[dict[str, Any]] = []
    text_tolerance = incumbent_text_loss * args.text_train_max_relative_regression
    progress: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment": (
            "hard_scale_compensated_set_conditional_assignment_gain"
            if args.hard_scale_compensation
            else "text_constrained_set_conditional_assignment_gain"
        ),
        "completed": False,
        "arguments": vars(args),
        "data_contract": {
            "task_split_manifest": split_manifest,
            "formal_test_data_used": False,
            "text_train_rows": list(args.text_train_rows),
            "text_validation_rows": list(args.text_validation_rows),
            "text_audit_rows": list(args.text_audit_rows),
        },
        "task_cache_contract": {
            "start_layer": task_cache.start_layer,
            "batches": len(task_cache.batches),
            "examples": task_cache.example_count,
            "choices": task_cache.choice_count,
            "tokens": task_cache.token_count,
            "parity": task_cache_parity,
        },
        "text_cache_contract": {
            "start_layer": text_cache.start_layer,
            "batches": len(text_cache.batches),
            "examples": text_cache.example_count,
            "tokens": text_cache.token_count,
            "hidden_bytes": text_cache.hidden_bytes,
            "estimated_hidden_bytes": expected_text_cache_bytes,
            "parity": text_cache_parity_result,
            "objective": "exact full text-train top-k teacher cross entropy",
            "candidate_parallelism": "all task-improving Linear bundles share one chunked lm-head pass per text row",
        },
        "baseline_train_task_objective": incumbent_task_objective,
        "baseline_train_text_cross_entropy": incumbent_text_loss,
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
            try:
                applicator.apply(candidate_state)
            except CompensationInfeasibleError as error:
                applicator.apply(accepted)
                candidate_records.append(
                    {
                        "layer": layer,
                        "linear": name,
                        "bundle_size": args.bundle_size,
                        "conditioned_on": [
                            f"{accepted_key[0]}:{accepted_key[1]}"
                            for accepted_key in accepted
                        ],
                        "prediction": _selection_prediction({key: bundles[key]}),
                        "block_output_mse_vs_incumbent": None,
                        "train_task_objective": incumbent_task_objective,
                        "conditional_task_gain": 0.0,
                        "train_text_cross_entropy": None,
                        "scale_compensation": None,
                        "infeasible_reason": "scale_compensation_infeasible",
                        "infeasible_detail": str(error),
                    }
                )
                continue
            scores, block_mse = score_cached_task_cache(
                model,
                task_cache,
                device=device,
                probe_layer=layer,
                reference_outputs=reference_outputs,
            )
            task_objective = summarize_cached_task_scores(
                scores,
                examples["train"],
                teacher_kl_weight=args.teacher_kl_weight,
                teacher_temperature=args.teacher_temperature,
            )
            task_loss = float(task_objective["balanced_combined_loss"])
            candidate_records.append(
                {
                    "layer": layer,
                    "linear": name,
                    "bundle_size": args.bundle_size,
                    "conditioned_on": [f"{key[0]}:{key[1]}" for key in accepted],
                    "prediction": _selection_prediction({key: bundles[key]}),
                    "block_output_mse_vs_incumbent": block_mse,
                    "train_task_objective": task_objective,
                    "conditional_task_gain": float(
                        incumbent_task_objective["balanced_combined_loss"]
                    )
                    - task_loss,
                    "train_text_cross_entropy": None,
                    "scale_compensation": (
                        applicator.compensation_for(key, bundles[key]).summary()
                        if isinstance(applicator, HardScaleCompensatedApplicator)
                        else None
                    ),
                    "infeasible_reason": None,
                }
            )

        incumbent_task_loss = float(
            incumbent_task_objective["balanced_combined_loss"]
        )
        improving_keys = [
            (layer, str(record["linear"]))
            for record in candidate_records
            if float(record["train_task_objective"]["balanced_combined_loss"])
            + args.conditional_tolerance
            < incumbent_task_loss
        ]
        applicator.apply(accepted)
        if improving_keys:
            overrides = materialize_candidate_weight_overrides(
                model, applicator, accepted, bundles, improving_keys
            )
            text_losses = score_cached_text_candidates(
                model,
                text_cache,
                layer=layer,
                candidate_overrides=overrides,
                vocab_chunk_size=args.text_lm_head_vocab_chunk,
                device=device,
            )
            for record in candidate_records:
                key = (layer, str(record["linear"]))
                if key in text_losses:
                    record["train_text_cross_entropy"] = text_losses[key]
            del overrides

        decision = choose_text_constrained_bundle(
            incumbent_task_loss=incumbent_task_loss,
            incumbent_text_loss=incumbent_text_loss,
            candidates={
                str(record["linear"]): {
                    "task_loss": float(
                        record["train_task_objective"]["balanced_combined_loss"]
                    ),
                    "text_loss": record["train_text_cross_entropy"],
                    "infeasible_reason": record["infeasible_reason"],
                }
                for record in candidate_records
            },
            task_tolerance=args.conditional_tolerance,
            text_tolerance=text_tolerance,
        )
        if decision["accepted"]:
            selected_name = str(decision["key"])
            selected_record = next(
                record
                for record in candidate_records
                if record["linear"] == selected_name
            )
            accepted[(layer, selected_name)] = bundles[(layer, selected_name)]
            incumbent_task_objective = selected_record["train_task_objective"]
            incumbent_text_loss = float(selected_record["train_text_cross_entropy"])

        finite_control_records = [
            record
            for record in candidate_records
            if record["block_output_mse_vs_incumbent"] is not None
        ]
        curvature_costs = [
            float(record["prediction"]["curvature_cost_sum"])
            for record in finite_control_records
        ]
        block_mses = [
            float(record["block_output_mse_vs_incumbent"])
            for record in finite_control_records
        ]
        step = {
            "layer": layer,
            "incumbent_before": candidate_records[0]["conditioned_on"],
            "candidates": candidate_records,
            "decision": decision,
            "positive_control": {
                "quantity": "bundle curvature cost vs measured immediate block-output MSE",
                "spearman": (
                    spearman_rank_correlation(curvature_costs, block_mses)
                    if len(finite_control_records) >= 2
                    else None
                ),
                "count": len(finite_control_records),
            },
        }
        layer_steps.append(step)
        progress["layer_steps"] = layer_steps
        progress["accepted_bundles"] = [f"{key[0]}:{key[1]}" for key in accepted]
        progress["current_train_task_objective"] = incumbent_task_objective
        progress["current_train_text_cross_entropy"] = incumbent_text_loss
        _atomic_json(output_summary, progress)
        print(json.dumps({"text_constrained_layer": step}, ensure_ascii=False), flush=True)
        applicator.apply(accepted)
        del reference_outputs
        torch.cuda.empty_cache()

    applicator.apply(accepted)
    switched = sum(selection.vector_indices.numel() for selection in accepted.values())
    baseline_validation: dict[str, Any] | None = None
    selected_validation: dict[str, Any] | None = None
    baseline_audit: dict[str, Any] | None = None
    candidate_audit: dict[str, Any] | None = None
    task_gate_passed = False
    validation_gate_passed = False
    audit_gate_passed = False
    if switched:
        applicator.restore()
        baseline_validation = _evaluate_examples(
            model, examples["validation"], device=device
        )
        applicator.apply(accepted)
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
            validation_text_examples = build_deferred_text_teacher_split(
                model,
                checkpoint,
                applicator,
                accepted,
                model_path=args.model,
                token_cache=args.text_token_cache,
                split="validation",
                rows=list(args.text_validation_rows),
                sequence_length=args.text_sequence_length,
                teacher_topk=args.text_teacher_topk,
                device=device,
            )
            applicator.restore()
            baseline_validation["text_guard_cross_entropy"] = _evaluate_text_guard(
                model, validation_text_examples, device=device
            )
            applicator.apply(accepted)
            selected_validation["text_guard_cross_entropy"] = _evaluate_text_guard(
                model, validation_text_examples, device=device
            )
            validation_gate_passed = choose_loss_dominant_snapshot(
                baseline_validation,
                selected_validation,
                min_relative_loss_improvement=args.selection_min_relative_loss_improvement,
                max_macro_regression=args.selection_max_macro_regression,
                max_per_task_loss_regression=args.selection_max_per_task_loss_regression,
                max_text_loss_regression=args.text_max_relative_regression,
            )
        else:
            selected_validation["text_guard_cross_entropy"] = None

    if validation_gate_passed:
        audit_text_examples = build_deferred_text_teacher_split(
            model,
            checkpoint,
            applicator,
            accepted,
            model_path=args.model,
            token_cache=args.text_token_cache,
            split="audit",
            rows=list(args.text_audit_rows),
            sequence_length=args.text_sequence_length,
            teacher_topk=args.text_teacher_topk,
            device=device,
        )
        applicator.restore()
        baseline_audit = _evaluate_examples(model, examples["audit"], device=device)
        baseline_audit["text_guard_cross_entropy"] = _evaluate_text_guard(
            model, audit_text_examples, device=device
        )
        applicator.apply(accepted)
        candidate_audit = _evaluate_examples(model, examples["audit"], device=device)
        candidate_audit["text_guard_cross_entropy"] = _evaluate_text_guard(
            model, audit_text_examples, device=device
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
    checkpoint_contract: dict[str, Any] | None = None
    if audit_gate_passed:
        if not args.output_checkpoint:
            raise ValueError("accepted text-constrained endpoint requires checkpoint path")
        checkpoint_metadata = {
                "method": "exact_text_constrained_conditional_assignment_search",
                "bundle_size": args.bundle_size,
                "accepted_bundles": [f"{key[0]}:{key[1]}" for key in accepted],
                "train_task_objective": incumbent_task_objective,
                "train_text_cross_entropy": incumbent_text_loss,
                "validation": selected_validation,
                "audit": candidate_audit,
            }
        if isinstance(applicator, HardScaleCompensatedApplicator):
            checkpoint_metadata["method"] = (
                "hard_assignment_with_analytic_local_group_scale_compensation"
            )
            checkpoint_metadata["compensations"] = {
                f"{key[0]}:{key[1]}": compensation.summary()
                for key, compensation in applicator.selected_compensations(accepted).items()
            }
            _checkpoint_with_hard_scale_compensation(
                copy.deepcopy(checkpoint),
                accepted,
                applicator,
                metadata=checkpoint_metadata,
                output=Path(args.output_checkpoint),
            )
            fresh_checkpoint = torch.load(
                args.output_checkpoint,
                map_location="cpu",
                weights_only=not args.trusted_checkpoint,
            )
            try:
                checkpoint_contract = audit_fresh_compensated_checkpoint(
                    checkpoint,
                    fresh_checkpoint,
                    model,
                    accepted,
                    device=device,
                )
            except Exception:
                Path(args.output_checkpoint).unlink(missing_ok=True)
                raise
            finally:
                del fresh_checkpoint
        else:
            _checkpoint_with_selection(
                copy.deepcopy(checkpoint),
                accepted,
                metadata=checkpoint_metadata,
                output=Path(args.output_checkpoint),
            )
        checkpoint_written = True
    applicator.restore()

    peak_memory = int(torch.cuda.max_memory_allocated(device))
    summary = {
        **progress,
        "completed": True,
        "method": {
            "coordinate": (
                "one fixed hard-assignment bundle plus analytic affected-group scale projection per Linear"
                if args.hard_scale_compensation
                else "one fixed curvature-ranked assignment bundle per Linear"
            ),
            "search": "causal one-pass constrained discrete coordinate search",
            "feasible_set": "non-regressing exact full text-train top-k teacher cross entropy",
            "selection": "minimum task-train combined loss inside the text-feasible set",
            "text_metric": "top-k teacher cross entropy; its candidate difference equals top-k KL difference because teacher entropy is fixed",
            "text_train_sequences": len(text_train_examples),
            "text_train_sequence_length": args.text_sequence_length,
            "trained_parameters": 0,
            "scale_compensation": (
                "closed-form incumbent-hard-weight weighted least-squares on touched groups"
                if args.hard_scale_compensation
                else None
            ),
            "hard_assignment_precedes_scale_compensation": args.hard_scale_compensation,
            "logical_bitrate_changed": False,
        },
        "total_vectors": total_vectors,
        "accepted_bundles": [f"{key[0]}:{key[1]}" for key in accepted],
        "accepted_switches": int(switched),
        "accepted_switch_rate": int(switched) / total_vectors,
        "selected_train_task_objective": incumbent_task_objective,
        "selected_train_text_cross_entropy": incumbent_text_loss,
        "baseline_validation": baseline_validation,
        "selected_validation": selected_validation,
        "task_gate_passed": task_gate_passed,
        "validation_gate_passed": validation_gate_passed,
        "baseline_audit": baseline_audit,
        "candidate_audit": candidate_audit,
        "audit_gate_passed": audit_gate_passed,
        "checkpoint_written": checkpoint_written,
        "checkpoint_contract": checkpoint_contract,
        "output_checkpoint": args.output_checkpoint if checkpoint_written else None,
        "formal_test_data_used": False,
        "elapsed_seconds": time.time() - started,
        "gpu_peak_memory_bytes": peak_memory,
    }
    _atomic_json(output_summary, summary)
    print(json.dumps({"text_constrained_summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
