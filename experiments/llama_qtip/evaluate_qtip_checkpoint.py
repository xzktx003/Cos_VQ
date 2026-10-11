"""Evaluate a Hugging Face QTIP checkpoint with the Vector-GSQ protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoConfig, AutoTokenizer

from experiments.llama_qtip.protocol import DEFAULT_TASKS, infer_model_family
from experiments.llama_qtip.qtip_manifest_compat import (
    install_qtip_manifest_fallbacks,
)


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

SUPPORTED_EXTRA_TASKS = ("boolq", "mmlu", "ifeval")
MMLU_PREFIX = "mmlu_"


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


def metric_candidates(task: str) -> tuple[str, ...]:
    if task in PRIMARY_METRICS:
        return PRIMARY_METRICS[task]
    if task.startswith(MMLU_PREFIX):
        return PRIMARY_METRICS["mmlu"]
    raise ValueError(f"Unsupported task name: {task}")


def validate_tasks(tasks: list[str]) -> None:
    supported = set(DEFAULT_TASKS) | set(SUPPORTED_EXTRA_TASKS)
    unknown = sorted(
        task
        for task in set(tasks)
        if task not in supported and not task.startswith(MMLU_PREFIX)
    )
    if unknown:
        raise ValueError(f"Unsupported task names: {unknown}")


def _task_summary(task: str, metrics: dict[str, Any]) -> dict[str, Any]:
    primary_key = next(
        (key for key in metric_candidates(task) if key in metrics),
        None,
    )
    if primary_key is None:
        raise ValueError(
            f"Task {task} lacks expected metrics {metric_candidates(task)}"
        )
    return {
        "metric": primary_key,
        "value": float(metrics[primary_key]),
        "stderr": metrics.get(primary_key.replace(",none", "_stderr,none")),
    }


def _mmlu_summary(result_mapping: dict[str, Any]) -> dict[str, Any]:
    values = []
    subtasks = []
    for task, metrics in sorted(result_mapping.items()):
        if not task.startswith(MMLU_PREFIX) or not isinstance(metrics, dict):
            continue
        try:
            payload = _task_summary(task, metrics)
        except ValueError:
            continue
        subtasks.append(task)
        values.append(float(payload["value"]))
    if not values:
        raise ValueError("lm_eval results are missing task mmlu")
    return {
        "metric": "acc,none",
        "value": sum(values) / len(values),
        "stderr": None,
        "subtask_count": len(values),
        "subtasks": subtasks,
    }


def extract_summary(raw_results: dict[str, Any], tasks: list[str]) -> dict[str, Any]:
    result_mapping = raw_results.get("results", {})
    task_metrics = {}
    primary_values = []
    for task in tasks:
        metrics = result_mapping.get(task)
        if metrics is None:
            payload = _mmlu_summary(result_mapping) if task == "mmlu" else None
            if payload is None:
                raise ValueError(f"lm_eval results are missing task {task}")
        else:
            payload = _task_summary(task, metrics)
        value = float(payload["value"])
        primary_values.append(value)
        task_metrics[task] = payload
    return {
        "tasks": task_metrics,
        "macro_average": sum(primary_values) / len(primary_values),
    }


def validate_quality_sanity(ppl: float | None, evaluation: dict[str, Any]) -> None:
    """Reject completed-looking results produced by a corrupted QTIP decode."""
    if ppl is not None and (
        not math.isfinite(ppl) or ppl <= 1.0 or ppl >= 1_000.0
    ):
        raise RuntimeError(
            f"QTIP decode produced implausible WikiText2 perplexity: {ppl}"
        )
    macro_average = float(evaluation["macro_average"])
    if not math.isfinite(macro_average) or not 0.0 <= macro_average <= 1.0:
        raise RuntimeError(
            f"QTIP decode produced invalid task macro average: {macro_average}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtip-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument(
        "--tokenizer-path",
        default="",
        help="Optional local tokenizer path for gated LLaMA base models.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--effective-bpp", type=float, required=True)
    parser.add_argument("--ppl-seqlen", type=int, default=2048)
    parser.add_argument("--ppl-batch-size", type=int, default=1)
    parser.add_argument("--ppl-split", choices=("test",), default="test")
    parser.add_argument("--wikitext2-revision", default="main")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument(
        "--skip-wikitext2",
        action="store_true",
        help="Skip the already-completed WikiText2 pass for supplement-only jobs.",
    )
    parser.add_argument(
        "--clean-c4",
        action="store_true",
        help="Evaluate clean C4 validation shard 1 PPL in the loaded-model process.",
    )
    parser.add_argument(
        "--clean-c4-cache-dir",
        default="",
        help="Tokenizer-specific clean C4 token cache directory; defaults under output-dir.",
    )
    parser.add_argument("--clean-c4-batch-size", type=int, default=1)
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply the tokenizer chat template for instruction-following tasks.",
    )
    parser.add_argument("--max-mem-ratio", type=float, default=0.7)
    parser.add_argument(
        "--manifest-model",
        action="store_true",
        help="Materialize QTIP weights when its codebook lacks a fast kernel.",
    )
    return parser.parse_args()


def install_transformers_qtip_compat(config: Any | None = None) -> dict[str, str]:
    """Repair Transformers compatibility needed by QTIP's vendored model."""
    import transformers.utils as transformers_utils

    compatibility: dict[str, str] = {}
    if hasattr(transformers_utils, "is_quanto_available"):
        compatibility["is_quanto_available"] = "native"
    else:
        replacement = getattr(
            transformers_utils,
            "is_optimum_quanto_available",
            lambda: False,
        )
        transformers_utils.is_quanto_available = replacement
        compatibility["is_quanto_available"] = "is_optimum_quanto_available"

    if config is not None:
        import transformers.modeling_utils as modeling_utils

        if getattr(modeling_utils, "ALL_PARALLEL_STYLES", ()) is None:
            from experiments.qwen_qtip.qtip_runtime import install_tp_plan_compat

            install_tp_plan_compat(config)
            compatibility["all_parallel_styles"] = "repaired_from_model_config"
        else:
            compatibility["all_parallel_styles"] = "native"
    return compatibility


def import_qtip(
    qtip_repo: Path,
    *,
    manifest_model: bool,
    model_family: str,
    model_config: Any,
):
    """Import QTIP only after validating the explicitly selected checkout."""
    qtip_repo = qtip_repo.resolve()
    required = qtip_repo / "lib/utils/unsafe_import.py"
    if not required.is_file():
        raise FileNotFoundError(f"Not a QTIP checkout: missing {required}")
    for import_root in (qtip_repo, qtip_repo / "qtip-kernels"):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    transformers_compatibility = install_transformers_qtip_compat(model_config)
    compatibility = (
        install_qtip_manifest_fallbacks() if manifest_model else {"mode": "native"}
    )
    compatibility["transformers"] = transformers_compatibility
    from lib.linear import QuantizedLinear  # type: ignore[import-not-found]
    if model_family == "llama":
        from lib.utils.unsafe_import import (  # type: ignore[import-not-found]
            model_from_hf_path,
        )
    elif model_family == "qwen3":
        from experiments.qwen_qtip.qtip_runtime import load_qwen3_qtip_model

        def model_from_hf_path(path, max_mem_ratio=0.7, device_map=None):
            del max_mem_ratio
            return load_qwen3_qtip_model(
                path,
                qtip_repo=qtip_repo,
                device_map=device_map,
            )
    else:
        raise ValueError(f"Unsupported QTIP model family: {model_family}")

    return model_from_hf_path, QuantizedLinear, compatibility


def model_input_device(model: torch.nn.Module) -> torch.device:
    embedding = model.get_input_embeddings()
    device = getattr(getattr(embedding, "weight", None), "device", None)
    if isinstance(device, torch.device) and device.type != "meta":
        return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Could not determine QTIP model input device")


@torch.no_grad()
def evaluate_wikitext2(
    model: torch.nn.Module,
    tokenizer,
    *,
    seqlen: int,
    batch_size: int,
    revision: str = "main",
) -> tuple[float, int]:
    """Evaluate contiguous WikiText2 test tokens at the prescribed length."""
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
        revision=revision,
    )
    encoded = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    token_ids = encoded.input_ids
    sequence_count = token_ids.numel() // seqlen
    if sequence_count == 0:
        raise ValueError(f"WikiText2 contains fewer than {seqlen} tokens")
    token_ids = token_ids[:, : sequence_count * seqlen].reshape(sequence_count, seqlen)

    input_device = model_input_device(model)
    total_nll = 0.0
    total_tokens = 0
    for offset in range(0, sequence_count, batch_size):
        batch = token_ids[offset : offset + batch_size].to(input_device)
        logits = model(
            input_ids=batch,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
        ).logits[:, :-1, :]
        labels = batch[:, 1:].to(logits.device)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels.reshape(-1),
            reduction="sum",
        )
        total_nll += float(loss)
        total_tokens += labels.numel()
    return math.exp(total_nll / total_tokens), sequence_count


def write_clean_c4_ppl(
    model: torch.nn.Module,
    tokenizer,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    # Clean-C4 is an optional supplement, not part of the V19+E2E endpoint.
    # Keep its package dependency out of the standard WikiText-2 + lm-eval path.
    from experiments.supplement_eval import clean_c4_ppl

    cache_dir = (
        Path(args.clean_c4_cache_dir)
        if args.clean_c4_cache_dir
        else output_dir / "clean_c4_cache"
    )
    cache = clean_c4_ppl.load_or_build_clean_c4_cache(
        tokenizer,
        cache_dir,
        sample_count=clean_c4_ppl.DEFAULT_SAMPLE_COUNT,
        seqlen=args.ppl_seqlen,
        seed=clean_c4_ppl.DEFAULT_SEED,
    )
    payload = clean_c4_ppl.evaluate_loaded_model_on_clean_c4(
        model,
        cache,
        batch_size=args.clean_c4_batch_size,
        device=model_input_device(model),
    )
    output_path = output_dir / "clean_c4_ppl.json"
    atomic_json(output_path, payload)
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
    if args.effective_bpp <= 0:
        raise ValueError("--effective-bpp must be positive")

    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        checkpoint_config = AutoConfig.from_pretrained(args.hf_path)
        checkpoint_family = infer_model_family(checkpoint_config)
        model_from_hf_path, quantized_linear_cls, runtime_compatibility = import_qtip(
            Path(args.qtip_repo),
            manifest_model=args.manifest_model,
            model_family=checkpoint_family,
            model_config=checkpoint_config,
        )
        model, base_model = model_from_hf_path(
            args.hf_path,
            max_mem_ratio=args.max_mem_ratio,
            device_map="balanced",
        )
        model.eval()
        model_family = infer_model_family(model.config)
        if model_family != checkpoint_family:
            raise ValueError(
                "QTIP loader changed model family: "
                f"checkpoint={checkpoint_family}, loaded={model_family}"
            )
        if args.manifest_model:
            if runtime_compatibility["qtip_kernels"] != "native_extension":
                raise RuntimeError(
                    "QTIP manifestation requires its native CUDA decoder; "
                    "the kernel-packed trellis is not equivalent to the "
                    "standard Python unpacking layout"
                )
            manifested_linear_count = 0
            for module in model.modules():
                if isinstance(module, quantized_linear_cls):
                    if not module.has_kernel:
                        raise RuntimeError(
                            "The selected QTIP checkpoint contains a linear "
                            "unsupported by the compiled CUDA decoder"
                        )
                    module.mode = "train-fixW"
                    manifested_linear_count += 1
            runtime_compatibility["trellis_layout"] = "native_cuda_kernel_packed"
            runtime_compatibility["manifested_linear_count"] = (
                manifested_linear_count
            )

        tokenizer_source = args.tokenizer_path or base_model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        ppl = None
        sequence_count = 0
        if not args.skip_wikitext2:
            ppl, sequence_count = evaluate_wikitext2(
                model,
                tokenizer,
                seqlen=args.ppl_seqlen,
                batch_size=args.ppl_batch_size,
                revision=args.wikitext2_revision,
            )
        clean_c4_result = None
        if args.clean_c4:
            clean_c4_result = write_clean_c4_ppl(model, tokenizer, args, output_dir)
        harness_model = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
        )
        raw_results = evaluator.simple_evaluate(
            model=harness_model,
            tasks=tasks,
            num_fewshot=0,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            limit=args.limit,
            bootstrap_iters=args.bootstrap_iters,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
            apply_chat_template=args.apply_chat_template,
        )
        if raw_results is None:
            raise RuntimeError("lm_eval returned no results")
        evaluation_summary = extract_summary(raw_results, tasks)
        validate_quality_sanity(ppl, evaluation_summary)
        raw_path = output_dir / "lm_eval_results.json"
        atomic_json(raw_path, raw_results)

        result: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "experiment": "qtip_llama_matched_protocol_evaluation",
            "arguments": vars(args),
            "model": {
                "qtip_hf_path": args.hf_path,
                "base_model": base_model,
                "tokenizer_path": tokenizer_source,
                "family": model_family,
                "class": type(model).__name__,
                "source_repo": str(Path(args.qtip_repo).resolve()),
                "runtime_compatibility": runtime_compatibility,
            },
            "ppl": (
                None
                if ppl is None
                else {
                    "dataset": "wikitext2",
                    "split": args.ppl_split,
                    "dataset_revision": args.wikitext2_revision,
                    "seqlen": args.ppl_seqlen,
                    "sequence_count": sequence_count,
                    "value": ppl,
                }
            ),
            "evaluation": evaluation_summary,
            "raw_results": str(raw_path),
            "clean_c4_ppl": (
                str(output_dir / "clean_c4_ppl.json")
                if clean_c4_result is not None
                else None
            ),
            "elapsed_seconds": time.time() - started,
            "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "completed": True,
        }
        atomic_json(output_dir / "summary.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=json_default))
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
