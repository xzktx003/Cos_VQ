"""Calibrate deployable Vector-GSQ group scales on task training prompts.

The discrete assignment and codebook stay fixed.  Only multiplicative deltas
for already-serialized input-group scales are learned, then folded back into
the same FP16 checkpoint fields, so the logical bitrate does not change.
Formal benchmark validation/test examples are never used for optimization:
train, selection, and final audit subsets are deterministic disjoint slices of
each lm-eval task's training split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.nowag_d6.diagnose_block_vector_gsq import (
    LINEAR_NAMES,
    get_linear_modules,
)
from experiments.nowag_d6.run_prefix_vector_gsq import reconstruct_logical_state


DEFAULT_TASKS = (
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "piqa",
    "winogrande",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=int, nargs="+", default=[28, 29, 30, 31])
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--train-per-task", type=int, default=8)
    parser.add_argument("--validation-per-task", type=int, default=4)
    parser.add_argument("--audit-per-task", type=int, default=4)
    parser.add_argument("--task-split-offset", type=int, default=0)
    parser.add_argument("--task-audit-offset", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--teacher-kl-weight", type=float, default=0.25)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--l2-weight", type=float, default=1e-4)
    parser.add_argument("--max-log-delta", type=float, default=0.08)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--text-token-cache", default="")
    parser.add_argument("--text-train-rows", type=int, nargs="+", default=None)
    parser.add_argument("--text-validation-rows", type=int, nargs="+", default=None)
    parser.add_argument("--text-audit-rows", type=int, nargs="+", default=None)
    parser.add_argument("--text-sequence-length", type=int, default=256)
    parser.add_argument("--text-teacher-topk", type=int, default=32)
    parser.add_argument("--text-guard-weight", type=float, default=0.0)
    parser.add_argument("--text-max-relative-regression", type=float, default=0.001)
    parser.add_argument("--selection-min-relative-loss-improvement", type=float, default=0.005)
    parser.add_argument("--selection-max-macro-regression", type=float, default=0.01)
    parser.add_argument("--selection-max-per-task-loss-regression", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trusted-checkpoint", action="store_true")
    return parser.parse_args()


def split_task_indices(
    document_count: int,
    *,
    train_count: int,
    validation_count: int,
    audit_count: int,
    seed: int,
    task_name: str,
    offset: int = 0,
    audit_offset: int | None = None,
) -> dict[str, list[int]]:
    if min(document_count, train_count, validation_count, audit_count, offset) < 0:
        raise ValueError("document and split counts plus offset must be non-negative")
    if audit_offset is not None and audit_offset < 0:
        raise ValueError("audit offset must be non-negative")
    requested = train_count + validation_count + audit_count
    if audit_offset is None and offset + requested > document_count:
        raise ValueError(
            f"task {task_name} has {document_count} training documents, "
            f"but requested window [{offset}, {offset + requested}) exceeds it"
        )
    train_validation_end = offset + train_count + validation_count
    if audit_offset is not None and train_validation_end > document_count:
        raise ValueError(
            f"task {task_name} train/validation window "
            f"[{offset}, {train_validation_end}) exceeds {document_count}"
        )
    audit_start = train_validation_end if audit_offset is None else audit_offset
    audit_end = audit_start + audit_count
    if audit_end > document_count:
        raise ValueError(
            f"task {task_name} audit window [{audit_start}, {audit_end}) "
            f"exceeds {document_count}"
        )
    train_validation_positions = set(range(offset, train_validation_end))
    audit_positions = set(range(audit_start, audit_end))
    if train_validation_positions & audit_positions:
        raise ValueError("task audit window overlaps train/validation window")
    task_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{task_name}".encode("utf-8")).digest()[:8],
        "big",
    )
    indices = list(range(document_count))
    random.Random(task_seed).shuffle(indices)
    train_start = offset
    train_end = train_start + train_count
    validation_end = train_end + validation_count
    return {
        "train": indices[train_start:train_end],
        "validation": indices[train_end:validation_end],
        "audit": indices[audit_start:audit_end],
    }


def gold_label(task_name: str, document: dict[str, Any]) -> int:
    if task_name in {"arc_easy", "arc_challenge"}:
        labels = [str(value) for value in document["choices"]["label"]]
        return labels.index(str(document["answerKey"]))
    if task_name == "hellaswag":
        return int(document.get("gold", document.get("label")))
    if task_name == "piqa":
        return int(document["label"])
    if task_name == "winogrande":
        return int(document["answer"]) - 1
    raise ValueError(f"no gold-label adapter for task {task_name}")


def encode_loglikelihood_request(
    tokenizer: Any,
    context: str,
    continuation: str,
    *,
    max_length: int,
) -> dict[str, Any]:
    if not isinstance(context, str) or not isinstance(continuation, str):
        raise TypeError("loglikelihood context and continuation must be strings")
    if not continuation:
        raise ValueError("loglikelihood continuation must be non-empty")
    if max_length < 2:
        raise ValueError("max_length must be at least two")

    context_ids = tokenizer.encode(context, add_special_tokens=False)
    joined_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
    if joined_ids[: len(context_ids)] == context_ids:
        continuation_ids = joined_ids[len(context_ids) :]
    else:
        continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if not continuation_ids:
        raise ValueError("tokenized continuation must be non-empty")
    if len(continuation_ids) >= max_length:
        raise ValueError("continuation alone does not fit within max_length")

    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        raise ValueError("tokenizer must define a BOS token")
    context_capacity = max_length - 1 - len(continuation_ids)
    retained_context = context_ids[-context_capacity:] if context_capacity else []
    input_ids = [int(bos_token_id), *retained_context, *continuation_ids]
    return {
        "input_ids": input_ids,
        "continuation_start": 1 + len(retained_context),
        "continuation_tokens": len(continuation_ids),
    }


class GroupScaleDelta(nn.Module):
    """Multiplicative row/input-group delta for one frozen Linear weight."""

    def __init__(self, *, rows: int, in_features: int, group_size: int) -> None:
        super().__init__()
        if min(rows, in_features, group_size) <= 0:
            raise ValueError("group-scale dimensions must be positive")
        self.in_features = int(in_features)
        self.group_size = int(group_size)
        groups = math.ceil(in_features / group_size)
        self.log_delta = nn.Parameter(torch.zeros(rows, groups, dtype=torch.float32))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        expanded = self.log_delta.exp().repeat_interleave(
            self.group_size, dim=1
        )[:, : self.in_features]
        return weight * expanded.to(device=weight.device, dtype=weight.dtype)


def fold_group_scale_delta(
    state: dict[str, Any], log_delta: torch.Tensor
) -> torch.Tensor:
    group_scale = state.get("group_scale")
    if not isinstance(group_scale, torch.Tensor):
        raise ValueError("logical state has no deployable group_scale tensor")
    if tuple(group_scale.shape) != tuple(log_delta.shape):
        raise ValueError(
            f"group scale shape {tuple(group_scale.shape)} does not match "
            f"delta shape {tuple(log_delta.shape)}"
        )
    folded = group_scale.float() * log_delta.detach().cpu().float().exp()
    if not bool(torch.isfinite(folded).all()):
        raise ValueError("folded group scales contain non-finite values")
    return folded.to(torch.float16)


def choose_validation_snapshot(
    best: dict[str, float], candidate: dict[str, float]
) -> bool:
    accuracy_delta = candidate["macro_accuracy"] - best["macro_accuracy"]
    if accuracy_delta > 1e-12:
        return True
    return (
        abs(accuracy_delta) <= 1e-12
        and candidate["balanced_loss"] < best["balanced_loss"] - 1e-12
    )


def choose_guarded_validation_snapshot(
    best: dict[str, float],
    candidate: dict[str, float],
    *,
    reference_text_loss: float,
    max_relative_regression: float,
) -> bool:
    if not math.isfinite(reference_text_loss) or reference_text_loss <= 0:
        raise ValueError("reference text loss must be finite and positive")
    if not math.isfinite(max_relative_regression) or max_relative_regression < 0:
        raise ValueError("text regression tolerance must be finite and non-negative")
    candidate_text_loss = float(candidate["text_guard_cross_entropy"])
    if candidate_text_loss > reference_text_loss * (1.0 + max_relative_regression):
        return False
    return choose_validation_snapshot(best, candidate)


def choose_loss_dominant_snapshot(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    min_relative_loss_improvement: float,
    max_macro_regression: float,
    max_per_task_loss_regression: float,
    max_text_loss_regression: float,
) -> bool:
    """Match assignment refinement's task/text checkpoint acceptance rule."""
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


def topk_teacher_cross_entropy(
    student_logits: torch.Tensor,
    teacher_indices: torch.Tensor,
    teacher_probabilities: torch.Tensor,
) -> torch.Tensor:
    if student_logits.ndim != 3 or student_logits.shape[0] != 1:
        raise ValueError("student logits must have shape [1, sequence, vocabulary]")
    if teacher_indices.shape != teacher_probabilities.shape:
        raise ValueError("teacher indices and probabilities must have matching shapes")
    if teacher_indices.ndim != 2 or teacher_indices.shape[0] != student_logits.shape[1]:
        raise ValueError("teacher tensors must cover every student sequence position")
    probabilities = teacher_probabilities.to(
        device=student_logits.device, dtype=torch.float32
    )
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("teacher probabilities contain non-finite values")
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    log_probabilities = F.log_softmax(student_logits.float(), dim=-1)[0]
    selected = log_probabilities.gather(
        1, teacher_indices.to(device=student_logits.device, dtype=torch.long)
    )
    return -(probabilities * selected).sum(dim=-1).mean()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _request_hash(task_name: str, document_index: int, requests: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{task_name}:{document_index}".encode("utf-8"))
    for request in requests:
        digest.update(repr(tuple(request.arguments)).encode("utf-8"))
    return digest.hexdigest()


def _build_examples(
    tokenizer: Any,
    *,
    task_names: list[str],
    train_per_task: int,
    validation_per_task: int,
    audit_per_task: int,
    seed: int,
    max_length: int,
    split_offset: int = 0,
    audit_offset: int | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    from lm_eval.tasks import TaskManager, get_task_dict

    task_dict = get_task_dict(task_names, task_manager=TaskManager())
    examples: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "audit": [],
    }
    split_manifest: dict[str, Any] = {}
    for task_name in task_names:
        task = task_dict[task_name]
        if not task.has_training_docs():
            raise ValueError(f"task {task_name} has no training split")
        documents = list(task.training_docs())
        split_indices = split_task_indices(
            len(documents),
            train_count=train_per_task,
            validation_count=validation_per_task,
            audit_count=audit_per_task,
            seed=seed,
            task_name=task_name,
            offset=split_offset,
            audit_offset=audit_offset,
        )
        split_manifest[task_name] = {
            "training_document_count": len(documents),
            "split_offset": split_offset,
            "audit_offset": audit_offset,
            "indices": split_indices,
        }
        for split_name, indices in split_indices.items():
            for document_index in indices:
                document = documents[document_index]
                context = task.fewshot_context(document, num_fewshot=0)
                requests = task.construct_requests(document, context)
                if not isinstance(requests, list):
                    requests = [requests]
                if not requests or any(
                    request.request_type != "loglikelihood" for request in requests
                ):
                    raise ValueError(
                        f"task {task_name} did not produce pure loglikelihood choices"
                    )
                encoded_choices = [
                    encode_loglikelihood_request(
                        tokenizer,
                        request.arguments[0],
                        request.arguments[1],
                        max_length=max_length,
                    )
                    for request in requests
                ]
                label = gold_label(task_name, document)
                if not 0 <= label < len(encoded_choices):
                    raise ValueError(
                        f"task {task_name} document {document_index} has invalid gold {label}"
                    )
                examples[split_name].append(
                    {
                        "task": task_name,
                        "document_index": document_index,
                        "gold": label,
                        "choices": encoded_choices,
                        "request_hash": _request_hash(
                            task_name, document_index, requests
                        ),
                    }
                )
    return examples, split_manifest


def _choice_score(
    model: nn.Module,
    encoded: dict[str, Any],
    *,
    device: torch.device,
) -> torch.Tensor:
    input_ids = torch.tensor(
        encoded["input_ids"], dtype=torch.long, device=device
    ).unsqueeze(0)
    start = int(encoded["continuation_start"])
    output = model(input_ids=input_ids, use_cache=False)
    logits = output.logits[0, start - 1 : -1].float()
    targets = input_ids[0, start:]
    if logits.shape[0] != targets.shape[0] or targets.numel() == 0:
        raise RuntimeError("continuation/logit alignment failed")
    token_scores = F.log_softmax(logits, dim=-1).gather(
        1, targets.unsqueeze(1)
    )[:, 0]
    return token_scores.mean()


@torch.inference_mode()
def _detached_scores(
    model: nn.Module,
    example: dict[str, Any],
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.stack(
        [_choice_score(model, choice, device=device) for choice in example["choices"]]
    ).detach()


def _teacher_scores(
    model: nn.Module,
    examples: dict[str, list[dict[str, Any]]],
    *,
    device: torch.device,
) -> None:
    for split_examples in examples.values():
        for example in split_examples:
            example["teacher_scores"] = _detached_scores(
                model, example, device=device
            ).cpu()


@torch.inference_mode()
def _build_text_guard_examples(
    model: nn.Module,
    *,
    token_cache: str,
    rows_by_split: dict[str, list[int]],
    sequence_length: int,
    teacher_topk: int,
    device: torch.device,
) -> dict[str, list[dict[str, torch.Tensor | int]]]:
    payload = torch.load(token_cache, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "input_ids" not in payload:
        raise ValueError("text token cache must contain input_ids")
    all_ids = payload["input_ids"]
    if all_ids.ndim != 2 or all_ids.shape[1] < sequence_length:
        raise ValueError("text token cache rows are shorter than requested sequence length")
    flat_rows = [row for rows in rows_by_split.values() for row in rows]
    if len(flat_rows) != len(set(flat_rows)):
        raise ValueError("text guard train/validation/audit rows must be disjoint")
    if min(flat_rows) < 0 or max(flat_rows) >= all_ids.shape[0]:
        raise ValueError("text guard row is outside the token cache")
    examples: dict[str, list[dict[str, torch.Tensor | int]]] = {
        "train": [],
        "validation": [],
        "audit": [],
    }
    for split, rows in rows_by_split.items():
        for row in rows:
            input_ids = all_ids[row, :sequence_length].clone()
            logits = model(
                input_ids=input_ids.unsqueeze(0).to(device), use_cache=False
            ).logits[:, :-1].float()
            top_values, top_indices = logits.topk(teacher_topk, dim=-1)
            teacher_probabilities = top_values.softmax(dim=-1)[0]
            examples[split].append(
                {
                    "row": int(row),
                    "input_ids": input_ids,
                    "teacher_indices": top_indices[0].to(torch.int32).cpu(),
                    "teacher_probabilities": teacher_probabilities.to(torch.float16).cpu(),
                }
            )
    return examples


def _text_guard_loss(
    model: nn.Module,
    example: dict[str, torch.Tensor | int],
    *,
    device: torch.device,
) -> torch.Tensor:
    input_ids = example["input_ids"]
    assert isinstance(input_ids, torch.Tensor)
    output = model(input_ids=input_ids.unsqueeze(0).to(device), use_cache=False)
    indices = example["teacher_indices"]
    probabilities = example["teacher_probabilities"]
    assert isinstance(indices, torch.Tensor) and isinstance(probabilities, torch.Tensor)
    return topk_teacher_cross_entropy(
        output.logits[:, :-1], indices.to(device), probabilities.to(device)
    )


@torch.inference_mode()
def _evaluate_text_guard(
    model: nn.Module,
    examples: list[dict[str, torch.Tensor | int]],
    *,
    device: torch.device,
) -> float:
    if not examples:
        raise ValueError("text guard evaluation requires at least one example")
    losses = [float(_text_guard_loss(model, example, device=device)) for example in examples]
    return sum(losses) / len(losses)


def _evaluate_examples(
    model: nn.Module,
    examples: list[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    by_task: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    with torch.inference_mode():
        for example in examples:
            scores = _detached_scores(model, example, device=device)
            target = torch.tensor([example["gold"]], device=device)
            loss = float(F.cross_entropy(scores.unsqueeze(0), target))
            correct = int(scores.argmax()) == int(example["gold"])
            by_task[example["task"]].append((loss, correct))
    per_task = {
        task: {
            "count": len(records),
            "accuracy": sum(correct for _loss, correct in records) / len(records),
            "loss": sum(loss for loss, _correct in records) / len(records),
        }
        for task, records in sorted(by_task.items())
    }
    return {
        "macro_accuracy": sum(item["accuracy"] for item in per_task.values())
        / len(per_task),
        "balanced_loss": sum(item["loss"] for item in per_task.values())
        / len(per_task),
        "per_task": per_task,
    }


def _training_coefficients(
    scores: torch.Tensor,
    *,
    gold: int,
    teacher_scores: torch.Tensor,
    teacher_kl_weight: float,
    teacher_temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    probabilities = scores.softmax(dim=0)
    coefficients = probabilities.clone()
    coefficients[gold] -= 1.0
    supervised_loss = F.cross_entropy(
        scores.unsqueeze(0), torch.tensor([gold], device=scores.device)
    )
    teacher_probabilities = (teacher_scores / teacher_temperature).softmax(dim=0)
    student_log_probabilities = (scores / teacher_temperature).log_softmax(dim=0)
    teacher_log_probabilities = (teacher_scores / teacher_temperature).log_softmax(dim=0)
    teacher_kl = (
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities)
    ).sum() * (teacher_temperature**2)
    coefficients = coefficients + (
        teacher_kl_weight
        * teacher_temperature
        * ((scores / teacher_temperature).softmax(dim=0) - teacher_probabilities)
    )
    return coefficients.detach(), {
        "supervised_loss": float(supervised_loss),
        "teacher_kl": float(teacher_kl),
        "combined_loss": float(supervised_loss + teacher_kl_weight * teacher_kl),
    }


def _load_logical_weights(
    model: nn.Module,
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
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


def _install_group_scale_deltas(
    model: nn.Module,
    checkpoint: dict[str, Any],
    *,
    layers: list[int],
) -> dict[tuple[int, str], GroupScaleDelta]:
    installed: dict[tuple[int, str], GroupScaleDelta] = {}
    for layer in layers:
        modules = get_linear_modules(model, layer)
        layer_state = checkpoint["layers"][str(layer)]
        for name in LINEAR_NAMES:
            state = layer_state[name]
            if "group_scale" not in state or "group_size" not in state:
                raise ValueError(f"layer {layer} {name} has no group-scale metadata")
            module = modules[name]
            delta = GroupScaleDelta(
                rows=module.out_features,
                in_features=module.in_features,
                group_size=int(state["group_size"]),
            ).to(module.weight.device)
            if tuple(delta.log_delta.shape) != tuple(state["group_scale"].shape):
                raise ValueError(f"layer {layer} {name} group-scale shape mismatch")
            parametrize.register_parametrization(module, "weight", delta)
            installed[(layer, name)] = delta
    return installed


def _snapshot_deltas(
    installed: dict[tuple[int, str], GroupScaleDelta]
) -> dict[tuple[int, str], torch.Tensor]:
    return {
        key: module.log_delta.detach().cpu().clone()
        for key, module in installed.items()
    }


def _restore_deltas(
    installed: dict[tuple[int, str], GroupScaleDelta],
    snapshot: dict[tuple[int, str], torch.Tensor],
) -> None:
    with torch.no_grad():
        for key, module in installed.items():
            module.log_delta.copy_(
                snapshot[key].to(module.log_delta.device, module.log_delta.dtype)
            )


def _train_one_example(
    model: nn.Module,
    example: dict[str, Any],
    *,
    device: torch.device,
    teacher_kl_weight: float,
    teacher_temperature: float,
    loss_scale: float,
) -> dict[str, float]:
    scores = _detached_scores(model, example, device=device)
    teacher_scores = example["teacher_scores"].to(device)
    coefficients, losses = _training_coefficients(
        scores,
        gold=int(example["gold"]),
        teacher_scores=teacher_scores,
        teacher_kl_weight=teacher_kl_weight,
        teacher_temperature=teacher_temperature,
    )
    for coefficient, choice in zip(coefficients, example["choices"]):
        score = _choice_score(model, choice, device=device)
        (score * coefficient * loss_scale).backward()
    return losses


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.train_per_task,
        args.validation_per_task,
        args.audit_per_task,
        args.epochs,
        args.gradient_accumulation,
        args.max_length,
    )
    if min(positive) <= 0:
        raise ValueError("split, epoch, accumulation, and length values must be positive")
    if args.task_split_offset < 0:
        raise ValueError("task split offset must be non-negative")
    if args.task_audit_offset is not None and args.task_audit_offset < 0:
        raise ValueError("task audit offset must be non-negative")
    finite_positive = (args.learning_rate, args.teacher_temperature, args.max_log_delta)
    if not all(math.isfinite(value) and value > 0 for value in finite_positive):
        raise ValueError("learning rate, temperature, and max delta must be finite positive")
    if not math.isfinite(args.teacher_kl_weight) or args.teacher_kl_weight < 0:
        raise ValueError("teacher KL weight must be finite and non-negative")
    if not math.isfinite(args.l2_weight) or args.l2_weight < 0:
        raise ValueError("L2 weight must be finite and non-negative")
    if len(set(args.layers)) != len(args.layers):
        raise ValueError("calibrated layers must be unique")
    if set(args.tasks) != set(DEFAULT_TASKS):
        raise ValueError(f"tasks must be exactly {list(DEFAULT_TASKS)}")
    text_rows = (
        args.text_train_rows,
        args.text_validation_rows,
        args.text_audit_rows,
    )
    text_guard_enabled = bool(args.text_token_cache)
    if text_guard_enabled != all(rows is not None and len(rows) > 0 for rows in text_rows):
        raise ValueError(
            "text token cache and non-empty train/validation/audit rows must be supplied together"
        )
    if args.text_guard_weight < 0 or not math.isfinite(args.text_guard_weight):
        raise ValueError("text guard weight must be finite and non-negative")
    if text_guard_enabled and args.text_guard_weight <= 0:
        raise ValueError("enabled text guard requires a positive weight")
    if min(args.text_sequence_length, args.text_teacher_topk) <= 0:
        raise ValueError("text sequence length and teacher top-k must be positive")
    if (
        not math.isfinite(args.text_max_relative_regression)
        or args.text_max_relative_regression < 0
    ):
        raise ValueError("text regression tolerance must be finite and non-negative")
    selection_tolerances = (
        args.selection_min_relative_loss_improvement,
        args.selection_max_macro_regression,
        args.selection_max_per_task_loss_regression,
    )
    if not all(
        math.isfinite(value) and 0 <= value <= 1
        for value in selection_tolerances
    ):
        raise ValueError("selection tolerances must be finite values in [0, 1]")


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
    print("Collecting dense-teacher multiple-choice scores", flush=True)
    _teacher_scores(model, examples, device=device)
    text_guard_examples = None
    if args.text_token_cache:
        print("Collecting dense-teacher long-text top-k distributions", flush=True)
        text_guard_examples = _build_text_guard_examples(
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

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _load_logical_weights(model, checkpoint, device=device)
    installed = _install_group_scale_deltas(
        model, checkpoint, layers=list(args.layers)
    )
    trainable = [module.log_delta for module in installed.values()]
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)

    baseline_validation = _evaluate_examples(
        model, examples["validation"], device=device
    )
    baseline_audit = _evaluate_examples(model, examples["audit"], device=device)
    baseline_text_validation = None
    baseline_text_audit = None
    if text_guard_examples is not None:
        baseline_text_validation = _evaluate_text_guard(
            model, text_guard_examples["validation"], device=device
        )
        baseline_text_audit = _evaluate_text_guard(
            model, text_guard_examples["audit"], device=device
        )
        baseline_validation["text_guard_cross_entropy"] = baseline_text_validation
        baseline_audit["text_guard_cross_entropy"] = baseline_text_audit
    best_validation = baseline_validation
    best_epoch = 0
    anchor_snapshot = _snapshot_deltas(installed)
    best_snapshot = anchor_snapshot
    history = [{"epoch": 0, "validation": baseline_validation}]
    generator = random.Random(args.seed)
    text_train_position = 0

    for epoch in range(1, args.epochs + 1):
        order = list(examples["train"])
        generator.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        loss_records = []
        for position, example in enumerate(order, start=1):
            loss_records.append(
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
                position % args.gradient_accumulation == 0 or position == len(order)
            )
            if boundary:
                if text_guard_examples is not None:
                    text_example = text_guard_examples["train"][
                        text_train_position % len(text_guard_examples["train"])
                    ]
                    text_train_position += 1
                    (
                        args.text_guard_weight
                        * _text_guard_loss(model, text_example, device=device)
                    ).backward()
                if args.l2_weight:
                    regularizer = torch.stack(
                        [parameter.square().mean() for parameter in trainable]
                    ).mean()
                    (args.l2_weight * regularizer).backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                with torch.no_grad():
                    for parameter in trainable:
                        parameter.clamp_(-args.max_log_delta, args.max_log_delta)
                optimizer.zero_grad(set_to_none=True)
        validation = _evaluate_examples(
            model, examples["validation"], device=device
        )
        if text_guard_examples is not None:
            validation["text_guard_cross_entropy"] = _evaluate_text_guard(
                model, text_guard_examples["validation"], device=device
            )
        epoch_record = {
            "epoch": epoch,
            "mean_train_combined_loss": sum(
                record["combined_loss"] for record in loss_records
            )
            / len(loss_records),
            "validation": validation,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
        if text_guard_examples is not None:
            eligible_against_anchor = choose_loss_dominant_snapshot(
                baseline_validation,
                validation,
                min_relative_loss_improvement=(
                    args.selection_min_relative_loss_improvement
                ),
                max_macro_regression=args.selection_max_macro_regression,
                max_per_task_loss_regression=(
                    args.selection_max_per_task_loss_regression
                ),
                max_text_loss_regression=args.text_max_relative_regression,
            )
            snapshot_selected = eligible_against_anchor and (
                best_epoch == 0
                or validation["balanced_loss"]
                < best_validation["balanced_loss"] - 1e-12
            )
        else:
            snapshot_selected = choose_validation_snapshot(best_validation, validation)
        if snapshot_selected:
            best_validation = validation
            best_epoch = epoch
            best_snapshot = _snapshot_deltas(installed)

    _restore_deltas(installed, best_snapshot)
    selected_validation = _evaluate_examples(
        model, examples["validation"], device=device
    )
    candidate_audit = _evaluate_examples(model, examples["audit"], device=device)
    if text_guard_examples is not None:
        selected_validation["text_guard_cross_entropy"] = _evaluate_text_guard(
            model, text_guard_examples["validation"], device=device
        )
        candidate_audit["text_guard_cross_entropy"] = _evaluate_text_guard(
            model, text_guard_examples["audit"], device=device
        )

    audit_gate_passed: bool | None = None
    final_audit = candidate_audit
    if text_guard_examples is not None:
        audit_gate_passed = best_epoch > 0 and choose_loss_dominant_snapshot(
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
        if not audit_gate_passed:
            _restore_deltas(installed, anchor_snapshot)
            best_epoch = 0
            best_snapshot = anchor_snapshot
            best_validation = baseline_validation
            selected_validation = baseline_validation
            final_audit = baseline_audit

    for (layer, name), delta in installed.items():
        state = checkpoint["layers"][str(layer)][name]
        state["group_scale"] = fold_group_scale_delta(
            state, delta.log_delta.detach()
        )
    checkpoint["task_group_scale_calibration"] = {
        "method": "training_split_mc_supervision_with_dense_teacher_kl",
        "layers": list(args.layers),
        "tasks": list(args.tasks),
        "seed": args.seed,
        "task_split_offset": args.task_split_offset,
        "best_epoch": best_epoch,
        "bitrate_changed": False,
        "selection": "task_and_text_loss_dominance_with_independent_audit",
        "audit_gate_passed": audit_gate_passed,
    }
    output_checkpoint = Path(args.output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = output_checkpoint.with_suffix(
        output_checkpoint.suffix + ".tmp"
    )
    torch.save(checkpoint, temporary_checkpoint)
    os.replace(temporary_checkpoint, output_checkpoint)

    selected_deltas = torch.cat(
        [value.reshape(-1) for value in best_snapshot.values()]
    )
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment": "llama_vector_gsq_task_group_scale_calibration",
        "completed": True,
        "arguments": vars(args),
        "data_contract": {
            "source": "lm_eval_training_splits_only",
            "formal_validation_or_test_used_for_tuning": False,
            "split_manifest": split_manifest,
            "request_hashes": {
                split: [example["request_hash"] for example in split_examples]
                for split, split_examples in examples.items()
            },
            "text_guard": (
                {
                    "token_cache_identity": torch.load(
                        args.text_token_cache, map_location="cpu", weights_only=False
                    ).get("identity"),
                    "rows": {
                        "train": list(args.text_train_rows),
                        "validation": list(args.text_validation_rows),
                        "audit": list(args.text_audit_rows),
                    },
                    "sequence_length": args.text_sequence_length,
                    "teacher_topk": args.text_teacher_topk,
                    "formal_test_data": False,
                }
                if text_guard_examples is not None
                else None
            ),
        },
        "baseline_validation": baseline_validation,
        "baseline_audit": baseline_audit,
        "selected_validation": selected_validation,
        "candidate_audit": candidate_audit,
        "final_audit": final_audit,
        "audit_gate_passed": audit_gate_passed,
        "best_epoch": best_epoch,
        "history": history,
        "delta": {
            "count": selected_deltas.numel(),
            "min": float(selected_deltas.min()),
            "max": float(selected_deltas.max()),
            "mean_abs": float(selected_deltas.abs().mean()),
            "nonzero": int(torch.count_nonzero(selected_deltas)),
        },
        "checkpoint_contract": {
            "assignments_changed": False,
            "codebooks_changed": False,
            "group_scale_dtype": "float16",
            "logical_bitrate_changed": False,
        },
        "elapsed_seconds": time.time() - started,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    _atomic_json(Path(args.output_summary), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
