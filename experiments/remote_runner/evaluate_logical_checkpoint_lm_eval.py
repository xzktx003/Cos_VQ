"""Evaluate a logical Vector-GSQ checkpoint with lm-evaluation-harness.

The base Hugging Face model stays the source for embeddings, norms, and
``lm_head``.  Only the seven Linear modules stored for every Transformer block
are reconstructed from the logical checkpoint before zero-shot evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from experiments.llama_qtip.evaluate_qtip_checkpoint import evaluate_wikitext2
from experiments.nowag_d6.logical_state import (
    LINEAR_NAMES,
    get_linear_modules,
    logical_state_bits,
    residual_metadata,
)
from experiments.nowag_d6.v19_model_placement import (
    place_evaluation_model,
    placement_summary,
    reconstruct_on_module,
)


DEFAULT_TASKS = (
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "lambada_openai",
    "piqa",
    "winogrande",
)

SUPPORTED_EXTRA_TASKS = ("boolq", "mmlu", "ifeval")
MMLU_PREFIX = "mmlu_"

PRIMARY_METRICS = {
    "arc_challenge": ("acc_norm,none", "acc,none"),
    "arc_easy": ("acc_norm,none", "acc,none"),
    "hellaswag": ("acc_norm,none", "acc,none"),
    "lambada_openai": ("acc,none", "acc_norm,none"),
    "piqa": ("acc_norm,none", "acc,none"),
    "winogrande": ("acc,none", "acc_norm,none"),
    "boolq": ("acc,none", "acc_norm,none"),
    "mmlu": ("acc,none", "acc_norm,none"),
    "ifeval": (
        "prompt_level_strict_acc,none",
        "inst_level_strict_acc,none",
        "prompt_level_loose_acc,none",
        "inst_level_loose_acc,none",
    ),
}

BINARY_PAIRED_TASKS = set(DEFAULT_TASKS) | {"boolq"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split-gpus", action="store_true", help="Keep complete decoder layers resident across all visible GPUs")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--limit", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wikitext2-revision", default="main")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help=(
            "Forward apply_chat_template=True to lm_eval. Use this for "
            "instruction-following tasks such as IFEval."
        ),
    )
    parser.add_argument(
        "--matched-ppl",
        action="store_true",
        help=(
            "Evaluate WikiText2 test perplexity with the same implementation "
            "used by the matched QTIP evaluator before zero-shot tasks."
        ),
    )
    parser.add_argument(
        "--clean-c4",
        action="store_true",
        help=(
            "Evaluate clean C4 validation shard 00001 with 256 deterministic "
            "token spans of length 2048 (seed 19)."
        ),
    )
    parser.add_argument(
        "--clean-c4-cache-dir",
        default="",
        help="Tokenizer-specific clean C4 token-cache directory.",
    )
    parser.add_argument("--clean-c4-batch-size", type=int, default=1)
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help=(
            "Persist lm_eval per-sample outputs and a compact stable-key artifact "
            "for paired checkpoint analysis."
        ),
    )
    parser.add_argument(
        "--trusted-checkpoint",
        action="store_true",
        help=(
            "Allow weights_only=False only when an older PyTorch cannot safely "
            "decode a locally produced, checksum-verified checkpoint."
        ),
    )
    return parser.parse_args()


def metric_candidates(task: str) -> tuple[str, ...]:
    if task in PRIMARY_METRICS:
        return PRIMARY_METRICS[task]
    if task.startswith(MMLU_PREFIX):
        return PRIMARY_METRICS["mmlu"]
    raise ValueError(f"Unsupported task name: {task}")


def validate_tasks(tasks: list[str]) -> None:
    supported = set(DEFAULT_TASKS) | set(SUPPORTED_EXTRA_TASKS)
    unknown_tasks = sorted(
        task
        for task in set(tasks)
        if task not in supported and not task.startswith(MMLU_PREFIX)
    )
    if unknown_tasks:
        raise ValueError(f"Unsupported task names: {unknown_tasks}")


def select_primary_metric(task: str, metrics: dict[str, Any]) -> str:
    primary_key = next(
        (key for key in metric_candidates(task) if key in metrics),
        None,
    )
    if primary_key is None:
        raise ValueError(
            f"Task {task} lacks expected metrics {metric_candidates(task)}"
        )
    return primary_key


def supports_paired_samples(task: str) -> bool:
    return task in BINARY_PAIRED_TASKS or (
        task.startswith(MMLU_PREFIX) and not task.endswith("_generative")
    )


def _summary_payload(task: str, metrics: dict[str, Any]) -> dict[str, Any]:
    primary_key = select_primary_metric(task, metrics)
    return {
        "metric": primary_key,
        "value": float(metrics[primary_key]),
        "stderr": metrics.get(primary_key.replace(",none", "_stderr,none")),
    }


def aggregate_mmlu_subtasks(result_mapping: dict[str, Any]) -> dict[str, Any]:
    values = []
    subtasks = []
    for task, metrics in sorted(result_mapping.items()):
        if not task.startswith(MMLU_PREFIX) or not isinstance(metrics, dict):
            continue
        primary_key = next(
            (key for key in metric_candidates(task) if key in metrics),
            None,
        )
        if primary_key is None:
            continue
        subtasks.append(task)
        values.append(float(metrics[primary_key]))
    if not values:
        raise ValueError("lm_eval results are missing task mmlu")
    return {
        "metric": "acc,none",
        "value": sum(values) / len(values),
        "stderr": None,
        "subtask_count": len(values),
        "subtasks": subtasks,
    }


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def reconstruct_checkpoint(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    layers = checkpoint.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("Logical checkpoint is missing the layers mapping")

    expected_layers = len(model.model.layers)
    layer_ids = sorted(int(key) for key in layers)
    if layer_ids != list(range(expected_layers)):
        raise ValueError(
            f"Checkpoint covers layers {layer_ids}, expected 0..{expected_layers - 1}"
        )

    total_bits = 0
    total_weights = 0
    per_linear = []
    for layer_index in layer_ids:
        layer_state = layers[str(layer_index)]
        if set(layer_state) != set(LINEAR_NAMES):
            raise ValueError(
                f"Layer {layer_index} Linear coverage mismatch: "
                f"{sorted(layer_state)}"
            )
        modules = get_linear_modules(model, layer_index)
        for name in LINEAR_NAMES:
            state = layer_state[name]
            assignments = state["assignments"].long()
            codebook_size = int(state["codebook"].shape[0])
            if int(assignments.min()) < 0 or int(assignments.max()) >= codebook_size:
                raise ValueError(
                    f"Out-of-range assignment in layer {layer_index} {name}"
                )
            metadata_keys = ["codebook", "row_norm", "col_norm"]
            if "group_scale" in state:
                metadata_keys.append("group_scale")
            if not all(
                bool(torch.isfinite(state[key].float()).all())
                for key in metadata_keys
            ):
                raise ValueError(f"Non-finite metadata in layer {layer_index} {name}")

            module = modules[name]
            weight = reconstruct_on_module(module, state)
            if not torch.equal(module.weight.detach(), weight.to(module.weight.dtype)):
                raise AssertionError(
                    f"Fresh reconstruction mismatch in layer {layer_index} {name}"
                )

            assignment_bits = int(state["assignment_bits"])
            bits = logical_state_bits(state)
            total_bits += bits
            total_weights += module.weight.numel()
            per_linear.append(
                {
                    "layer": layer_index,
                    "name": name,
                    "shape": list(module.weight.shape),
                    "codebook_size": codebook_size,
                    "assignment_bits": assignment_bits,
                    **residual_metadata(state),
                    "used_codewords": int(torch.unique(assignments).numel()),
                    "assignment_min": int(assignments.min()),
                    "assignment_max": int(assignments.max()),
                    "group_scale_count": int(
                        state["group_scale"].numel()
                        if "group_scale" in state
                        else 0
                    ),
                    "group_size": int(state["group_size"])
                    if "group_size" in state
                    else None,
                }
            )

    return {
        "quantized_layers": expected_layers,
        "quantized_linears": len(per_linear),
        "expected_linears": expected_layers * len(LINEAR_NAMES),
        "linear_names": list(LINEAR_NAMES),
        "logical_effective_bpp": total_bits / max(total_weights, 1),
        "residual_parameters": sum(item["residual_parameters"] for item in per_linear),
        "residual_bits": sum(item["residual_bits"] for item in per_linear),
        "q_norm_k_norm_float_count": sum(
            1
            for name, _ in model.named_parameters()
            if name.endswith("q_norm.weight") or name.endswith("k_norm.weight")
        ),
        "lm_head_in_logical_checkpoint": any(
            "lm_head" in name for layer in layers.values() for name in layer
        ),
        "per_linear": per_linear,
    }


def extract_summary(
    raw_results: dict[str, Any],
    tasks: list[str],
) -> dict[str, Any]:
    result_mapping = raw_results.get("results", {})
    task_metrics = {}
    primary_values = []
    for task in tasks:
        metrics = result_mapping.get(task)
        if metrics is None:
            if task == "mmlu":
                payload = aggregate_mmlu_subtasks(result_mapping)
            else:
                raise ValueError(f"lm_eval results are missing task {task}")
        else:
            payload = _summary_payload(task, metrics)
        primary_values.append(float(payload["value"]))
        task_metrics[task] = payload
    return {
        "tasks": task_metrics,
        "macro_average": sum(primary_values) / max(len(primary_values), 1),
    }


def extract_paired_samples(
    raw_results: dict[str, Any],
    tasks: list[str],
) -> dict[str, Any]:
    """Build a compact, deterministic artifact for paired accuracy analysis.

    The four hashes plus ``doc_id`` identify the exact evaluation item and
    prompt contract.  Only the binary primary accuracy score is retained; full
    documents and responses remain available in ``lm_eval_results.json``.
    """

    sample_mapping = raw_results.get("samples")
    if not isinstance(sample_mapping, dict):
        raise ValueError("lm_eval results do not contain per-task samples")

    task_payloads = {}
    skipped_tasks = {}
    for task in tasks:
        if not supports_paired_samples(task):
            if task == "mmlu":
                skipped_tasks[task] = (
                    "paired samples require concrete binary-accuracy task samples"
                )
            else:
                skipped_tasks[task] = "paired samples require binary accuracy scores"
            continue
        samples = sample_mapping.get(task)
        if not isinstance(samples, list):
            raise ValueError(f"lm_eval samples are missing task {task}")

        metric = metric_candidates(task)[0].split(",", 1)[0]
        records = []
        seen_keys = set()
        for sample in samples:
            # lm_eval 0.4.4 omits this field for the default filter.
            if sample.get("filter", "none") != "none":
                continue
            missing = [
                key
                for key in (
                    "doc_id",
                    "doc_hash",
                    "prompt_hash",
                    "target_hash",
                    metric,
                )
                if key not in sample
            ]
            if missing:
                raise ValueError(
                    f"Task {task} sample is missing required fields {missing}"
                )
            score = float(sample[metric])
            if score not in (0.0, 1.0):
                raise ValueError(
                    f"Task {task} expected binary accuracy score, got {score}"
                )
            record = {
                "doc_id": int(sample["doc_id"]),
                "doc_hash": str(sample["doc_hash"]),
                "prompt_hash": str(sample["prompt_hash"]),
                "target_hash": str(sample["target_hash"]),
                "score": score,
            }
            pairing_key = tuple(record[key] for key in record if key != "score")
            if pairing_key in seen_keys:
                raise ValueError(f"Task {task} has duplicate pairing key {pairing_key}")
            seen_keys.add(pairing_key)
            records.append(record)

        records.sort(
            key=lambda item: (
                item["doc_id"],
                item["doc_hash"],
                item["prompt_hash"],
                item["target_hash"],
            )
        )
        task_payloads[task] = {
            "metric": metric,
            "sample_count": len(records),
            "samples": records,
        }

    return {
        "schema_version": 1,
        "pairing_key": ["doc_id", "doc_hash", "prompt_hash", "target_hash"],
        "tasks": task_payloads,
        "skipped_tasks": skipped_tasks,
    }


def write_matched_ppl(
    model: torch.nn.Module,
    tokenizer,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    ppl, sequence_count = evaluate_wikitext2(
        model,
        tokenizer,
        seqlen=2048,
        batch_size=1,
        revision=getattr(args, "wikitext2_revision", "main"),
    )
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "checkpoint": args.checkpoint,
        "ppl": float(ppl),
        "sequence_count": int(sequence_count),
        "seqlen": 2048,
        "dataset": "wikitext2_test",
        "dataset_revision": getattr(args, "wikitext2_revision", "main"),
        "ce_implementation": (
            "experiments.llama_qtip.evaluate_qtip_checkpoint.evaluate_wikitext2"
        ),
        "completed": True,
    }
    atomic_json(output_dir / "matched_ppl.json", payload)
    return payload


def write_clean_c4_ppl(
    model: torch.nn.Module,
    tokenizer,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    # This optional metric is deliberately isolated from the published
    # WikiText-2 + six-task V19+E2E evaluation path.
    from experiments.supplement_eval.clean_c4_ppl import (
        evaluate_loaded_model_on_clean_c4,
        load_or_build_clean_c4_cache,
    )

    if not args.clean_c4_cache_dir:
        raise ValueError("--clean-c4-cache-dir is required with --clean-c4")
    cache = load_or_build_clean_c4_cache(
        tokenizer,
        args.clean_c4_cache_dir,
    )
    payload = evaluate_loaded_model_on_clean_c4(
        model,
        cache,
        batch_size=args.clean_c4_batch_size,
        device=args.device,
    )
    payload.update(
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": args.model,
            "checkpoint": args.checkpoint,
        }
    )
    atomic_json(output_dir / "clean_c4_ppl.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    if args.clean_c4:
        raise RuntimeError(
            "Clean-C4 evaluation is an optional supplement not included in this standalone package"
        )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    validate_tasks(tasks)

    try:
        torch.manual_seed(args.seed)
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)

        try:
            checkpoint = torch.load(
                args.checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        except pickle.UnpicklingError:
            if not args.trusted_checkpoint:
                raise
            checkpoint = torch.load(
                args.checkpoint,
                map_location="cpu",
                weights_only=False,
            )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
            local_files_only=True,
        )
        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        if config.model_type in {"qwen3", "llama"}:
            from transformers import modeling_utils
            if getattr(modeling_utils, 'ALL_PARALLEL_STYLES', ()) is None:
                from experiments.qwen_qtip.qtip_runtime import install_tp_plan_compat
                install_tp_plan_compat(config)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = place_evaluation_model(model, device, split_gpus=args.split_gpus).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        checkpoint_audit = reconstruct_checkpoint(model, checkpoint, device)
        matched_ppl = None
        if args.matched_ppl:
            matched_ppl = write_matched_ppl(model, tokenizer, args, output_dir)
        clean_c4_ppl = None
        if args.clean_c4:
            clean_c4_ppl = write_clean_c4_ppl(
                model,
                tokenizer,
                args,
                output_dir,
            )

        harness_model = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            device=args.device,
            trust_remote_code=True,
        )
        raw_results = evaluator.simple_evaluate(
            model=harness_model,
            tasks=tasks,
            num_fewshot=0,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            device=args.device,
            limit=args.limit,
            bootstrap_iters=args.bootstrap_iters,
            log_samples=args.log_samples,
            apply_chat_template=args.apply_chat_template,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )
        if raw_results is None:
            raise RuntimeError("lm_eval returned no results")

        raw_path = output_dir / "lm_eval_results.json"
        atomic_json(raw_path, raw_results)
        paired_path = None
        paired_counts = None
        if args.log_samples:
            paired_path = output_dir / "paired_samples.json"
            paired_samples = extract_paired_samples(raw_results, tasks)
            atomic_json(paired_path, paired_samples)
            paired_counts = {
                task: payload["sample_count"]
                for task, payload in paired_samples["tasks"].items()
            }
        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "experiment": "logical_vector_gsq_lm_eval_zero_shot",
            "arguments": vars(args),
            "checkpoint_audit": checkpoint_audit,
            "model_placement": placement_summary(model),
            "evaluation": extract_summary(raw_results, tasks),
            "elapsed_seconds": time.time() - started,
            "gpu_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "raw_results": str(raw_path),
            "matched_ppl": (
                str(output_dir / "matched_ppl.json") if matched_ppl else None
            ),
            "clean_c4_ppl": (
                str(output_dir / "clean_c4_ppl.json") if clean_c4_ppl else None
            ),
            "paired_samples": str(paired_path) if paired_path else None,
            "paired_sample_counts": paired_counts,
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
