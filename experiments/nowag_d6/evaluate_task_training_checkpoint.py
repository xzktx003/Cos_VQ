"""Evaluate a logical checkpoint on a held-out slice of task training splits."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.nowag_d6.calibrate_task_group_scales import (
    DEFAULT_TASKS,
    _build_examples,
    _evaluate_examples,
    _load_logical_weights,
)


def candidate_passes_internal_gate(
    base: dict[str, float], candidate: dict[str, float]
) -> bool:
    return (
        candidate["macro_accuracy"] >= base["macro_accuracy"] - 1e-12
        and candidate["balanced_loss"] < base["balanced_loss"] - 1e-12
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--selection-per-task", type=int, default=64)
    parser.add_argument("--selection-seed", type=int, default=991)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--trusted-checkpoint", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if min(args.selection_per_task, args.max_length) <= 0:
        raise ValueError("selection count and max length must be positive")
    started = time.time()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    examples, manifest = _build_examples(
        tokenizer,
        task_names=list(DEFAULT_TASKS),
        train_per_task=0,
        validation_per_task=args.selection_per_task,
        audit_per_task=0,
        seed=args.selection_seed,
        max_length=args.max_length,
    )
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=not args.trusted_checkpoint,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    _load_logical_weights(model, checkpoint, device=device)
    metrics = _evaluate_examples(model, examples["validation"], device=device)
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment": "task_training_split_checkpoint_selection",
        "completed": True,
        "arguments": vars(args),
        "formal_validation_or_test_used": False,
        "metrics": metrics,
        "split_manifest": manifest,
        "request_hashes": [
            example["request_hash"] for example in examples["validation"]
        ],
        "elapsed_seconds": time.time() - started,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    _atomic_json(Path(args.output), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
