"""Shared validation and result schema for Vector-GSQ/QTIP comparisons."""

from __future__ import annotations

import re
from pathlib import PurePath
from typing import Any, Mapping


DEFAULT_TASKS = (
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "lambada_openai",
    "piqa",
    "winogrande",
)


def infer_model_family(config: object) -> str:
    """Return the supported dense decoder family represented by a HF config."""
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = [
        str(value).lower()
        for value in (getattr(config, "architectures", None) or [])
    ]
    if model_type == "llama" or "llamaforcausallm" in architectures:
        return "llama"
    if model_type == "qwen3" or "qwen3forcausallm" in architectures:
        return "qwen3"
    raise ValueError(
        "Unsupported model architecture for Vector-GSQ: "
        f"model_type={model_type!r}, architectures={architectures!r}; "
        "supported families are dense LLaMA and Qwen3"
    )


def normalize_model_name(value: str) -> str:
    """Normalize local paths and Hub identifiers to a comparable model slug."""
    leaf = PurePath(str(value).rstrip("/")).name.lower()
    return re.sub(r"[^a-z0-9]+", "_", leaf).strip("_")


def _model_identity(value: str) -> str:
    """Compare the full model slug while ignoring separator placement."""
    return normalize_model_name(value).replace("_", "")


def validate_token_cache_identity(identity: object, model: str) -> None:
    """Reject calibration tokens produced by a different tokenizer/model."""
    if not isinstance(identity, str):
        raise ValueError("Token cache is missing its tokenizer identity")
    fields = identity.split("|")
    if len(fields) < 2 or not fields[1]:
        raise ValueError(f"Malformed token cache identity: {identity!r}")
    cached_model = normalize_model_name(fields[1])
    selected_model = normalize_model_name(model)
    if cached_model != selected_model:
        raise ValueError(
            "Token cache model mismatch: "
            f"cache={cached_model}, selected={selected_model}. "
            "Build a tokenizer-specific cache for the selected model."
        )


def _require_completed(payload: Mapping[str, Any], label: str) -> None:
    if payload.get("completed") is not True:
        raise ValueError(f"{label} result is not completed")


def _task_names(payload: Mapping[str, Any], label: str) -> list[str]:
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError(f"{label} result has no evaluation mapping")
    tasks = evaluation.get("tasks")
    if not isinstance(tasks, Mapping):
        raise ValueError(f"{label} result has no per-task results")
    names = list(tasks)
    if set(names) != set(DEFAULT_TASKS):
        raise ValueError(
            f"Task set mismatch for {label}: got {sorted(names)}, "
            f"expected {sorted(DEFAULT_TASKS)}"
        )
    return names


def _macro_average(payload: Mapping[str, Any], label: str) -> float:
    evaluation = payload["evaluation"]
    value = float(evaluation["macro_average"])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} macro accuracy is outside [0, 1]: {value}")
    return value


def _validate_protocol(
    vector_ppl: Mapping[str, Any],
    vector_accuracy: Mapping[str, Any],
    qtip: Mapping[str, Any],
) -> tuple[str, list[str]]:
    vector_model = normalize_model_name(vector_ppl["arguments"]["model"])
    accuracy_model = normalize_model_name(vector_accuracy["arguments"]["model"])
    qtip_model = normalize_model_name(qtip["model"]["base_model"])
    model_identities = {
        _model_identity(vector_ppl["arguments"]["model"]),
        _model_identity(vector_accuracy["arguments"]["model"]),
        _model_identity(qtip["model"]["base_model"]),
    }
    if len(model_identities) != 1:
        raise ValueError(
            "Base model mismatch: "
            f"Vector-PPL={vector_model}, Vector-accuracy={accuracy_model}, "
            f"QTIP={qtip_model}"
        )

    vector_seqlen = int(vector_ppl["arguments"]["ppl_seqlen"])
    qtip_seqlen = int(qtip["arguments"]["ppl_seqlen"])
    vector_split = str(
        vector_ppl["arguments"].get(
            "ppl_split",
            vector_ppl.get("ppl_split", "test"),
        )
    )
    qtip_split = str(qtip["arguments"].get("ppl_split", "test"))
    if vector_seqlen != qtip_seqlen or vector_split != qtip_split:
        raise ValueError(
            "PPL protocol mismatch: "
            f"Vector=({vector_split}, {vector_seqlen}), "
            f"QTIP=({qtip_split}, {qtip_seqlen})"
        )

    vector_tasks = _task_names(vector_accuracy, "Vector-GSQ")
    qtip_tasks = _task_names(qtip, "QTIP")
    if set(vector_tasks) != set(qtip_tasks):
        raise ValueError(
            f"Task set mismatch: Vector-GSQ={vector_tasks}, QTIP={qtip_tasks}"
        )
    return vector_model, list(DEFAULT_TASKS)


def build_comparison_summary(
    vector_ppl: Mapping[str, Any],
    vector_accuracy: Mapping[str, Any],
    qtip: Mapping[str, Any],
    *,
    max_bpp_gap: float = 0.10,
) -> dict[str, Any]:
    """Validate a matched protocol and calculate signed Vector-GSQ deltas."""
    if max_bpp_gap < 0:
        raise ValueError("max_bpp_gap must be non-negative")
    for payload, label in (
        (vector_ppl, "Vector-GSQ PPL"),
        (vector_accuracy, "Vector-GSQ accuracy"),
        (qtip, "QTIP"),
    ):
        _require_completed(payload, label)

    model, tasks = _validate_protocol(vector_ppl, vector_accuracy, qtip)
    vector_bpp = float(vector_ppl["logical_effective_bpp"])
    qtip_bpp = float(qtip["arguments"]["effective_bpp"])
    absolute_bpp_gap = abs(vector_bpp - qtip_bpp)
    vector_ppl_value = float(vector_ppl["ppl"])
    qtip_ppl_value = float(qtip["ppl"]["value"])
    vector_accuracy_value = _macro_average(vector_accuracy, "Vector-GSQ")
    qtip_accuracy_value = _macro_average(qtip, "QTIP")

    per_task = {}
    for task in tasks:
        vector_value = float(vector_accuracy["evaluation"]["tasks"][task]["value"])
        qtip_value = float(qtip["evaluation"]["tasks"][task]["value"])
        per_task[task] = {
            "vector_gsq": vector_value,
            "qtip": qtip_value,
            "vector_gsq_minus_qtip": vector_value - qtip_value,
        }

    ppl_delta = vector_ppl_value - qtip_ppl_value
    accuracy_delta = vector_accuracy_value - qtip_accuracy_value
    rate_comparable = absolute_bpp_gap <= max_bpp_gap
    if not rate_comparable:
        verdict = "not_rate_comparable"
    elif ppl_delta < 0 and accuracy_delta > 0:
        verdict = "vector_gsq_better_on_both_quality_metrics"
    elif ppl_delta > 0 and accuracy_delta < 0:
        verdict = "qtip_better_on_both_quality_metrics"
    elif ppl_delta == 0 and accuracy_delta == 0:
        verdict = "quality_metrics_tied"
    else:
        verdict = "mixed_quality_result"

    return {
        "completed": True,
        "protocol": {
            "model": model,
            "ppl_dataset": "wikitext2",
            "ppl_split": vector_ppl["arguments"].get("ppl_split", "test"),
            "ppl_seqlen": int(vector_ppl["arguments"]["ppl_seqlen"]),
            "tasks": tasks,
        },
        "rate": {
            "vector_gsq_effective_bpp": vector_bpp,
            "qtip_effective_bpp": qtip_bpp,
            "absolute_bpp_gap": absolute_bpp_gap,
            "max_bpp_gap": float(max_bpp_gap),
            "comparable": rate_comparable,
        },
        "ppl": {
            "vector_gsq": vector_ppl_value,
            "qtip": qtip_ppl_value,
            "vector_gsq_minus_qtip": ppl_delta,
        },
        "accuracy": {
            "vector_gsq_macro_average": vector_accuracy_value,
            "qtip_macro_average": qtip_accuracy_value,
            "vector_gsq_minus_qtip": accuracy_delta,
            "per_task": per_task,
        },
        "verdict": verdict,
    }
