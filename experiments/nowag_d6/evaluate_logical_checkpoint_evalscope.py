"""Evaluate a reconstructed logical Vector-GSQ checkpoint with EvalScope."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from evalscope.config import TaskConfig
from evalscope.constants import EvalType, HubType
from evalscope.models.modelscope import ModelScopeAPI
from evalscope.run import run_task
from evalscope.api.model import GenerateConfig

from experiments.remote_runner.evaluate_logical_checkpoint_lm_eval import (
    reconstruct_checkpoint,
)


SUPPORTED_DATASETS = ("arc", "hellaswag", "piqa", "winogrande")


def parse_evalscope_datasets(value: str) -> list[str]:
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    unsupported = sorted(set(datasets) - set(SUPPORTED_DATASETS))
    if unsupported:
        raise ValueError(
            f"Datasets {unsupported} are not supported by EvalScope native "
            f"in this environment; supported={list(SUPPORTED_DATASETS)}"
        )
    if not datasets:
        raise ValueError("At least one EvalScope dataset is required")
    return datasets


def build_evalscope_dataset_args(datasets: list[str]) -> dict[str, dict[str, Any]]:
    result = {name: {"few_shot_num": 0} for name in datasets}
    if "arc" in result:
        result["arc"]["subset_list"] = ["ARC-Easy", "ARC-Challenge"]
    return result


def json_default(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-id", default="vector-gsq-d6")
    parser.add_argument("--datasets", default=",".join(SUPPORTED_DATASETS))
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument(
        "--dataset-hub",
        choices=(HubType.HUGGINGFACE, HubType.MODELSCOPE),
        default=HubType.HUGGINGFACE,
    )
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    try:
        datasets = parse_evalscope_datasets(args.datasets)
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        generation_config = GenerateConfig(
            do_sample=False,
            temperature=0.0,
            max_tokens=args.max_new_tokens,
        )
        model_api = ModelScopeAPI(
            model_name=args.model_id,
            model_path=args.model,
            device_map=args.device,
            precision="torch.bfloat16",
            local_files_only=True,
            enable_thinking=False,
            config=generation_config,
        )
        model_api.model.eval()
        for parameter in model_api.model.parameters():
            parameter.requires_grad_(False)
        checkpoint_audit = reconstruct_checkpoint(
            model_api.model,
            checkpoint,
            device,
        )
        atomic_json(output_dir / "vector_checkpoint_audit.json", checkpoint_audit)

        task_kwargs: dict[str, Any] = {}
        if args.dataset_dir:
            task_kwargs["dataset_dir"] = args.dataset_dir
        task_config = TaskConfig(
            model=model_api,
            model_id=args.model_id,
            datasets=datasets,
            dataset_args=build_evalscope_dataset_args(datasets),
            dataset_hub=args.dataset_hub,
            eval_type=EvalType.CHECKPOINT,
            eval_batch_size=args.eval_batch_size,
            limit=args.limit,
            generation_config=generation_config.model_dump(exclude_none=True),
            work_dir=str(output_dir),
            no_timestamp=True,
            seed=args.seed,
            **task_kwargs,
        )
        evalscope_result = run_task(task_config)
        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "experiment": "logical_vector_gsq_evalscope_native_zero_shot",
            "evalscope_version": __import__("evalscope").__version__,
            "arguments": vars(args),
            "datasets": datasets,
            "dataset_args": build_evalscope_dataset_args(datasets),
            "checkpoint_audit": checkpoint_audit,
            "evalscope_result": evalscope_result,
            "elapsed_seconds": time.time() - started,
            "gpu_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "completed": True,
        }
        atomic_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))
    except Exception as error:
        failure = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "arguments": vars(args),
            "elapsed_seconds": time.time() - started,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "completed": False,
        }
        atomic_json(output_dir / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
