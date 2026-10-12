"""Run the preregistered six-model V19+E2E reproduction pipeline.

The default invocation runs all models sequentially and writes every generated
cache, checkpoint, log, JSON result, and Markdown report below this package.
Use ``--dry-run`` to inspect the full command plan without creating files,
downloading models, or starting any training/evaluation process.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Iterable


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "configs" / "reproduction" / "v19_e2e_six_models.json"
RESULTS_ROOT = PACKAGE_ROOT / "experiments" / "results" / "v19_e2e"
REPORT_ROOT = PACKAGE_ROOT / "report"
MODEL_CHOICES = (
    "qwen3-4b",
    "qwen3-8b",
    "qwen3-14b",
    "qwen3-32b",
    "llama2-7b",
    "llama3.1-8b-instruct",
)


@dataclass(frozen=True)
class Command:
    name: str
    argv: list[str]


@dataclass(frozen=True)
class Artifacts:
    run_dir: Path
    rp1t_raw: Path
    tokens: Path
    fineweb: Path
    matched: Path
    e2e_tokens: Path
    baseline_raw: Path
    baseline: Path
    scale_dir: Path
    standard_dir: Path
    wide_dir: Path
    v19_selection: Path

    def stage_checkpoint(self, directory: Path, run_stamp: str) -> Path:
        return directory / f"{run_stamp}_checkpoint.pt"

    def e2e_trial_dir(self, branch: str, learning_rate: float) -> Path:
        return self.run_dir / "training" / branch / learning_rate_label(learning_rate)

    def e2e_selection(self, branch: str) -> Path:
        return self.run_dir / "selections" / f"{branch}_e2e.json"

    def final_eval_dir(self, endpoint: str) -> Path:
        return self.run_dir / "evaluation" / endpoint


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "v19-e2e-reproduction-v1":
        raise ValueError(f"Unsupported reproduction config schema: {path}")
    models = payload.get("models")
    if not isinstance(models, dict) or set(models) != set(MODEL_CHOICES):
        raise ValueError("The reproduction config must define exactly the six published models")
    return payload


def learning_rate_label(learning_rate: float) -> str:
    rendered = f"{learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"lr_{rendered}"


def make_artifacts(run_dir: Path) -> Artifacts:
    data_dir = run_dir / "data"
    return Artifacts(
        run_dir=run_dir,
        rp1t_raw=data_dir / "redpajama_raw.pt",
        tokens=data_dir / "token_cache.pt",
        fineweb=data_dir / "fineweb_reference.pt",
        matched=data_dir / "matched_fineweb_c4.pt",
        e2e_tokens=data_dir / "e2e_token_cache.pt",
        baseline_raw=run_dir / "checkpoints" / "baseline_vector_gsq.pt",
        baseline=run_dir / "checkpoints" / "baseline.pt",
        scale_dir=run_dir / "v19" / "scale_only",
        standard_dir=run_dir / "v19" / "standard",
        wide_dir=run_dir / "v19" / "wide",
        v19_selection=run_dir / "selections" / "v19.json",
    )


def module_command(python: str, module: str, *arguments: object) -> Command:
    return Command(module.rsplit(".", 1)[-1], [python, "-m", module, *map(str, arguments)])


def cache_commands(
    *,
    python: str,
    model: dict[str, Any],
    protocol: dict[str, Any],
    model_path: str,
    reference_model_path: str | None,
    artifacts: Artifacts,
) -> list[Command]:
    if model["cache_kind"] == "redpajama":
        data = protocol["datasets"]["redpajama"]
        return [
            module_command(
                python,
                "experiments.qwen_qtip.build_calibration_cache",
                "--tokenizer-model",
                model_path,
                "--output",
                artifacts.rp1t_raw,
                "--hessian-devset-size",
                data["hessian_sequences"],
                "--finetune-devset-size",
                model["finetune_sequences"],
                "--ctx-size",
                data["context_length"],
                "--seed",
                data["seed"],
                "--dataset-revision",
                data["revision"],
            ),
            module_command(
                python,
                "scripts.prepare_qtip_baseline_cache",
                "--source",
                artifacts.rp1t_raw,
                "--model",
                model_path,
                "--output",
                artifacts.tokens,
            ),
        ]

    if not reference_model_path:
        raise ValueError("Llama3.1 requires its published reference tokenizer snapshot")
    fineweb = protocol["datasets"]["fineweb_edu"]
    guard = protocol["datasets"]["c4_guard"]
    return [
        module_command(
            python,
            "experiments.llama_qtip.build_llama_token_cache",
            "--model",
            reference_model_path,
            "--output",
            artifacts.fineweb,
            "--sequences",
            fineweb["sequences"],
            "--sequence-length",
            fineweb["context_length"],
            "--seed",
            fineweb["seed"],
            "--shuffle-seed",
            fineweb["shuffle_seed"],
            "--shuffle-buffer-size",
            fineweb["shuffle_buffer_size"],
            "--dataset-revision",
            fineweb["revision"],
        ),
        module_command(
            python,
            "experiments.llama_qtip.build_matched_llama_assignment_cache",
            "--reference-model",
            reference_model_path,
            "--target-model",
            model_path,
            "--fineweb-cache",
            artifacts.fineweb,
            "--output",
            artifacts.matched,
            "--guard-sequences",
            guard["sequences"],
            "--sequence-length",
            guard["context_length"],
            "--guard-seed",
            guard["shuffle_seed"],
            "--c4-source",
            "download",
            "--c4-revision",
            guard["revision"],
        ),
        module_command(
            python,
            "experiments.nowag_d6.v19_prepare_e2e_cache",
            "--model",
            model_path,
            "--source-cache",
            artifacts.matched,
            "--output",
            artifacts.e2e_tokens,
            "--train-sequences",
            model["e2e_train_sequences"],
            "--validation-start",
            model["e2e_validation_start"],
            "--validation-sequences",
            model["e2e_validation_sequences"],
        ),
    ]


def baseline_command(
    *,
    python: str,
    model: dict[str, Any],
    protocol: dict[str, Any],
    model_path: str,
    token_cache: Path,
    artifacts: Artifacts,
) -> Command:
    settings = {**protocol["vector_baseline"], **model["baseline"]}
    arguments: list[object] = [
        "--model",
        model_path,
        "--token-cache",
        token_cache,
        "--device",
        "cuda:0",
        "--nowag-repo",
        PACKAGE_ROOT / "third_party" / "NoWag",
        "--output",
        artifacts.run_dir / "baseline" / "quantization.json",
        "--checkpoint",
        artifacts.baseline_raw,
        "--stage",
        "gsq",
        "--seed",
        protocol["seed"],
        "--max-layers",
        model["layers"],
        "--ppl-depths",
        "",
        "--ppl-seqlen",
        protocol["wiki_text_sequence_length"],
        "--calib-sequences",
        settings["calib_sequences"],
        "--calib-sequence-length",
        settings["calib_sequence_length"],
        "--gptq-token-offset",
        settings["gptq_token_offset"],
        "--train-sequences",
        settings["train_sequences"],
        "--train-sequence-length",
        settings["train_sequence_length"],
        "--validation-sequences",
        settings["validation_sequences"],
        "--validation-token-offset",
        settings["validation_token_offset"],
        "--gptq-row-chunk",
        settings["gptq_row_chunk"],
        "--gptq-candidates",
        settings["gptq_candidates"],
        "--gptq-concurrency",
        settings["gptq_concurrency"],
        "--local-candidates",
        settings["local_candidates"],
        "--steps",
        settings["steps"],
        "--vector-dim",
        protocol["vector_dim"],
        "--bits-per-weight",
        protocol["bits_per_weight"],
        "--lloyd-iters",
        settings["lloyd_iters"],
    ]
    if settings.get("gsq_objective"):
        arguments.extend(("--gsq-objective", settings["gsq_objective"]))
    if settings.get("layerwise_cpu_offload"):
        arguments.extend(("--layerwise-cpu-offload", "--use-hidden-cache"))
        arguments.extend(("--gpu-reservation-mib", settings["gpu_reservation_mib"]))
    if settings.get("use_hidden_cache"):
        arguments.append("--use-hidden-cache")
    if settings.get("stream_dense_targets"):
        arguments.append("--stream-dense-targets")
    if settings.get("staged_joint_group_scales"):
        arguments.append("--staged-joint-group-scales")
    if settings.get("scale_group_size"):
        arguments.extend(("--scale-finetune-mode", "group"))
        arguments.extend(("--scale-group-size", settings["scale_group_size"]))
    arguments.extend(("--train-batch-size", settings["train_batch_size"]))
    arguments.extend(("--train-microbatch-size", settings["train_microbatch_size"]))
    return module_command(python, "experiments.nowag_d6.run_prefix_vector_gsq", *arguments)


def llama_task_assignment_command(
    *,
    python: str,
    model: dict[str, Any],
    model_path: str,
    token_cache: Path,
    artifacts: Artifacts,
    source_checkpoint: Path,
) -> Command:
    settings = model["baseline"]["task_assignment"]
    arguments: list[object] = [
        "--model",
        model_path,
        "--checkpoint",
        source_checkpoint,
        "--output-checkpoint",
        artifacts.baseline,
        "--output-summary",
        artifacts.run_dir / "baseline" / "task_assignment.json",
        "--device",
        "cuda:0",
        "--layers",
        *settings["layers"],
        "--tasks",
        *settings["tasks"],
        "--optimization-objective",
        settings["optimization_objective"],
        "--train-per-task",
        settings["train_per_task"],
        "--validation-per-task",
        settings["validation_per_task"],
        "--audit-per-task",
        settings["audit_per_task"],
        "--task-split-offset",
        settings["task_split_offset"],
        "--epochs",
        settings["epochs"],
        "--learning-rate",
        settings["learning_rate"],
        "--gradient-accumulation",
        settings["gradient_accumulation"],
        "--teacher-kl-weight",
        settings["teacher_kl_weight"],
        "--teacher-temperature",
        settings["teacher_temperature"],
        "--switch-penalty",
        settings["switch_penalty"],
        "--max-switch-rate",
        settings["max_switch_rate"],
        "--initial-logit-margin",
        settings["initial_logit_margin"],
        "--initial-temperature",
        settings["initial_temperature"],
        "--final-temperature",
        settings["final_temperature"],
        "--initial-kappa",
        settings["initial_kappa"],
        "--final-kappa",
        settings["final_kappa"],
        "--local-candidates",
        settings["local_candidates"],
        "--candidate-chunk",
        settings["candidate_chunk"],
        "--expectation-chunk",
        settings["expectation_chunk"],
        "--max-length",
        settings["max_length"],
        "--text-token-cache",
        token_cache,
        "--text-train-rows",
        *range(*settings["text_train_rows"]),
        "--text-validation-rows",
        *range(*settings["text_validation_rows"]),
        "--text-audit-rows",
        *range(*settings["text_audit_rows"]),
        "--text-sequence-length",
        settings["text_sequence_length"],
        "--text-teacher-topk",
        settings["text_teacher_topk"],
        "--text-guard-weight",
        settings["text_guard_weight"],
        "--text-max-relative-regression",
        settings["text_max_relative_regression"],
        "--selection-min-relative-loss-improvement",
        settings["selection_min_relative_loss_improvement"],
        "--selection-max-macro-regression",
        settings["selection_max_macro_regression"],
        "--selection-max-per-task-loss-regression",
        settings["selection_max_per_task_loss_regression"],
        "--projection-stages",
        settings["projection_stages"],
        "--seed",
        settings["seed"],
        "--trusted-checkpoint",
    ]
    return module_command(
        python,
        "experiments.nowag_d6.calibrate_task_local_assignments",
        *arguments,
    )


def v19_command(
    *,
    python: str,
    model: dict[str, Any],
    protocol: dict[str, Any],
    model_path: str,
    token_cache: Path,
    source_checkpoint: Path | str,
    output_dir: Path,
    run_stamp: str,
    mode: str,
    active_positions: int,
    candidates_per_ranking: int,
) -> Command:
    settings = protocol["v19"]
    arguments: list[object] = [
        "--model",
        model_path,
        "--checkpoint",
        source_checkpoint,
        "--token-cache",
        token_cache,
        "--output-dir",
        output_dir,
        "--run-stamp",
        run_stamp,
        "--mode",
        mode,
        "--passes",
        settings["passes"],
        "--majorizer",
        settings["majorizer"],
        "--scale-layout",
        model["scale_layout"],
        "--calibration-sequences",
        settings["calibration_sequences"],
        "--context-length",
        settings["context_length"],
        "--calibration-start",
        settings["calibration_start"],
        "--development-start",
        model["v19_development_start"],
        "--development-sequences",
        model["v19_development_sequences"],
        "--active-positions",
        active_positions,
        "--candidates-per-ranking",
        candidates_per_ranking,
        "--groups-per-row",
        model["groups_per_row"],
        "--row-batch-size",
        settings["row_batch_size"],
        "--eta-relative",
        settings["eta_relative"],
        "--max-layers",
        0,
    ]
    return module_command(python, "experiments.nowag_d6.v19_model_driver", *arguments)


def v19_validation_command(
    *,
    python: str,
    model: dict[str, Any],
    protocol: dict[str, Any],
    model_path: str,
    token_cache: Path,
    artifacts: Artifacts,
    run_stamp: str,
    device: str,
    split_gpus: bool,
) -> Command:
    checkpoints = [
        artifacts.baseline,
        artifacts.stage_checkpoint(artifacts.scale_dir, run_stamp),
        artifacts.stage_checkpoint(artifacts.standard_dir, run_stamp),
        artifacts.stage_checkpoint(artifacts.wide_dir, run_stamp),
    ]
    labels = ["baseline", "scale", "v_standard", "v_wide"]
    settings = protocol["v19"]
    arguments: list[object] = [
        "--model",
        model_path,
        "--checkpoints",
        *checkpoints,
        "--labels",
        *labels,
        "--select-labels",
        "v_standard",
        "v_wide",
        "--token-cache",
        token_cache,
        "--start",
        model["v19_development_start"],
        "--sequences",
        model["v19_development_sequences"],
        "--exclude-sequences",
        model["v19_exclude_sequences"],
        "--context-length",
        settings["context_length"],
        "--output",
        artifacts.v19_selection,
        "--device",
        device,
    ]
    if split_gpus:
        arguments.append("--split-gpus")
    return module_command(python, "experiments.nowag_d6.v19_validation_score", *arguments)


def e2e_trial_command(
    *,
    python: str,
    model: dict[str, Any],
    protocol: dict[str, Any],
    model_path: str,
    source_checkpoint: Path | str,
    token_cache: Path,
    output_dir: Path,
    run_stamp: str,
    learning_rate: float,
) -> Command:
    arguments: list[object] = [
        "--model",
        model_path,
        "--checkpoint",
        source_checkpoint,
        "--token-cache",
        token_cache,
        "--dataset-label",
        "FineWeb-Edu" if model["cache_kind"] == "fineweb_c4_guard" else "RedPajama-1T",
        "--output-dir",
        output_dir,
        "--run-stamp",
        run_stamp,
        "--train-sequences",
        model["e2e_train_sequences"],
        "--validation-sequences",
        model["e2e_validation_sequences"],
        "--context-length",
        protocol["e2e_context_length"],
        "--epochs",
        protocol["e2e_epochs"],
        "--batch-size",
        protocol["e2e_batch_size"],
        "--learning-rate",
        learning_rate,
        "--seed",
        protocol["seed"],
        "--validation-interval",
        model["e2e_validation_interval"],
    ]
    if model["tune_group_scales"]:
        arguments.append("--tune-group-scales")
    return module_command(python, "experiments.nowag_d6.finetune_hard_codebook_e2e", *arguments)


def e2e_selection_command(
    *,
    python: str,
    artifacts: Artifacts,
    branch: str,
    run_stamp: str,
    learning_rates: Iterable[float],
) -> Command:
    directories = [artifacts.e2e_trial_dir(branch, lr) for lr in learning_rates]
    progress = [directory / f"{run_stamp}_progress.json" for directory in directories]
    labels = [learning_rate_label(lr) for lr in learning_rates]
    return module_command(
        python,
        "experiments.nowag_d6.v19_e2e_selection",
        "--progress",
        *progress,
        "--labels",
        *labels,
        "--output",
        artifacts.e2e_selection(branch),
    )


def evaluation_command(
    *,
    python: str,
    model_path: str,
    checkpoint: Path | str,
    output_dir: Path,
    protocol: dict[str, Any],
    device: str,
    split_gpus: bool,
) -> Command:
    arguments: list[object] = [
        "--model",
        model_path,
        "--checkpoint",
        checkpoint,
        "--output-dir",
        output_dir,
        "--device",
        device,
        "--wikitext2-revision",
        protocol["datasets"]["wikitext2"]["revision"],
        "--batch-size",
        protocol["lm_eval_batch_size"],
        "--tasks",
        ",".join(protocol["lm_eval_tasks"]),
        "--seed",
        protocol["lm_eval_seed"],
        "--bootstrap-iters",
        1000,
        "--matched-ppl",
        "--log-samples",
        "--trusted-checkpoint",
    ]
    if split_gpus:
        arguments.append("--split-gpus")
    return module_command(
        python,
        "experiments.remote_runner.evaluate_logical_checkpoint_lm_eval",
        *arguments,
    )


def model_plan(
    *,
    config: dict[str, Any],
    model_key: str,
    run_stamp: str,
    model_path: str,
    reference_model_path: str | None,
    run_dir: Path,
    python: str,
    device: str,
    split_gpus: bool,
) -> list[Command]:
    model = config["models"][model_key]
    protocol = config["protocol"]
    artifacts = make_artifacts(run_dir)
    quant_cache = artifacts.tokens if model["cache_kind"] == "redpajama" else artifacts.matched
    e2e_cache = artifacts.tokens if model["cache_kind"] == "redpajama" else artifacts.e2e_tokens
    commands = cache_commands(
        python=python,
        model=model,
        protocol=protocol,
        model_path=model_path,
        reference_model_path=reference_model_path,
        artifacts=artifacts,
    )
    commands.append(
        baseline_command(
            python=python,
            model=model,
            protocol=protocol,
            model_path=model_path,
            token_cache=quant_cache,
            artifacts=artifacts,
        )
    )
    baseline_checkpoint: Path | str = artifacts.baseline_raw
    if model["cache_kind"] == "fineweb_c4_guard":
        commands.append(
            llama_task_assignment_command(
                python=python,
                model=model,
                model_path=model_path,
                token_cache=quant_cache,
                artifacts=artifacts,
                source_checkpoint=artifacts.baseline_raw,
            )
        )
        baseline_checkpoint = artifacts.baseline
    v19 = protocol["v19"]
    commands.extend(
        [
            v19_command(
                python=python,
                model=model,
                protocol=protocol,
                model_path=model_path,
                token_cache=quant_cache,
                source_checkpoint=baseline_checkpoint,
                output_dir=artifacts.scale_dir,
                run_stamp=run_stamp,
                mode="scale_only",
                active_positions=v19["scale_only"]["active_positions"],
                candidates_per_ranking=v19["scale_only"]["candidates_per_ranking"],
            ),
            v19_command(
                python=python,
                model=model,
                protocol=protocol,
                model_path=model_path,
                token_cache=quant_cache,
                source_checkpoint=baseline_checkpoint,
                output_dir=artifacts.standard_dir,
                run_stamp=run_stamp,
                mode="interval",
                active_positions=v19["standard"]["active_positions"],
                candidates_per_ranking=v19["standard"]["candidates_per_ranking"],
            ),
            v19_command(
                python=python,
                model=model,
                protocol=protocol,
                model_path=model_path,
                token_cache=quant_cache,
                source_checkpoint=baseline_checkpoint,
                output_dir=artifacts.wide_dir,
                run_stamp=run_stamp,
                mode="interval",
                active_positions=v19["wide"]["active_positions"],
                candidates_per_ranking=v19["wide"]["candidates_per_ranking"],
            ),
            v19_validation_command(
                python=python,
                model=model,
                protocol=protocol,
                model_path=model_path,
                token_cache=quant_cache,
                artifacts=artifacts,
                run_stamp=run_stamp,
                device=device,
                split_gpus=split_gpus,
            ),
        ]
    )

    for branch in e2e_branches(model):
        source: Path | str
        if branch == "baseline":
            source = baseline_checkpoint
        elif branch == "scale":
            source = artifacts.stage_checkpoint(artifacts.scale_dir, run_stamp)
        else:
            source = f"<selected-V19-checkpoint:{artifacts.v19_selection}#selected.checkpoint>"
        for learning_rate in protocol["e2e_learning_rates"]:
            commands.append(
                e2e_trial_command(
                    python=python,
                    model=model,
                    protocol=protocol,
                    model_path=model_path,
                    source_checkpoint=source,
                    token_cache=e2e_cache,
                    output_dir=artifacts.e2e_trial_dir(branch, learning_rate),
                    run_stamp=run_stamp,
                    learning_rate=learning_rate,
                )
            )
        commands.append(
            e2e_selection_command(
                python=python,
                artifacts=artifacts,
                branch=branch,
                run_stamp=run_stamp,
                learning_rates=protocol["e2e_learning_rates"],
            )
        )

    endpoint_checkpoints: dict[str, Path | str] = {
        "baseline": baseline_checkpoint,
        "scale": artifacts.stage_checkpoint(artifacts.scale_dir, run_stamp),
        "v19": f"<selected-V19-checkpoint:{artifacts.v19_selection}#selected.checkpoint>",
    }
    for branch in e2e_branches(model):
        endpoint_checkpoints[f"{branch}_e2e"] = (
            f"<selected-E2E-checkpoint:{artifacts.e2e_selection(branch)}#selected.checkpoint>"
        )
    for endpoint, checkpoint in endpoint_checkpoints.items():
        commands.append(
            evaluation_command(
                python=python,
                model_path=model_path,
                checkpoint=checkpoint,
                output_dir=artifacts.final_eval_dir(endpoint),
                protocol=protocol,
                device=device,
                split_gpus=split_gpus,
            )
        )
    return commands


def e2e_branches(model: dict[str, Any]) -> list[str]:
    branches = ["baseline", "v19"]
    if model["scale_e2e"]:
        branches.append("scale")
    return branches


def display_command(command: Command) -> str:
    argv = list(command.argv)
    # The Llama3.1 baseline passes explicit disjoint row IDs to preserve the
    # paper protocol. Compress these long integer runs only in preview output.
    for flag, end_marker in (
        ("--text-train-rows", "--text-validation-rows"),
        ("--text-validation-rows", "--text-audit-rows"),
        ("--text-audit-rows", "--text-sequence-length"),
    ):
        if flag in argv and end_marker in argv:
            begin = argv.index(flag) + 1
            end = argv.index(end_marker)
            values = argv[begin:end]
            if values and all(str(value).isdigit() for value in values):
                argv[begin:end] = [f"{values[0]}..{values[-1]}"]
    return shlex.join(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def resolve_model_snapshot(
    model: dict[str, Any], *, override: str | None, dry_run: bool
) -> str:
    if override:
        return str(Path(override).expanduser().resolve())
    if dry_run:
        return f"HF_SNAPSHOT({model['repo_id']}@{model['revision']})"
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model["repo_id"], revision=model["revision"])


def resolve_dataset_revisions(
    config: dict[str, Any],
    model_keys: Iterable[str],
    overrides: dict[str, str | None],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Freeze mutable Hub dataset branches to immutable commit SHAs per run."""
    resolved_config = copy.deepcopy(config)
    needed: set[str] = set()
    needed.add("wikitext2")
    for model_key in model_keys:
        kind = resolved_config["models"][model_key]["cache_kind"]
        if kind == "redpajama":
            needed.add("redpajama")
        else:
            needed.update(("fineweb_edu", "c4_guard"))
    revisions: dict[str, str] = {}
    for key in sorted(needed):
        dataset = resolved_config["protocol"]["datasets"][key]
        requested = overrides.get(key) or dataset["revision"]
        if re.fullmatch(r"[0-9a-f]{40}", requested):
            revision = requested
        else:
            from huggingface_hub import HfApi

            revision = HfApi().dataset_info(
                dataset["repo_id"], revision=requested
            ).sha
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise RuntimeError(f"Could not resolve immutable dataset revision for {key}")
        dataset["requested_revision"] = requested
        dataset["revision"] = revision
        revisions[key] = revision
    return resolved_config, revisions


def output_root_path(value: str | None) -> Path:
    root = Path(value).expanduser() if value else RESULTS_ROOT
    if not root.is_absolute():
        root = PACKAGE_ROOT / root
    root = root.resolve()
    allowed_root = (PACKAGE_ROOT / "experiments" / "results").resolve()
    if root != allowed_root and allowed_root not in root.parents:
        raise ValueError(f"Run outputs must stay under {allowed_root}")
    return root


class RunManager:
    def __init__(
        self,
        campaign_dir: Path,
        *,
        python: str,
        cuda_visible_devices: str | None,
    ) -> None:
        self.campaign_dir = campaign_dir
        self.python = python
        self.cuda_visible_devices = cuda_visible_devices
        self.manifest_path = campaign_dir / "campaign_manifest.json"
        self.manifest: dict[str, Any] = {
            "schema": "v19-e2e-run-manifest-v1",
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
            "python": python,
            "cuda_visible_devices": cuda_visible_devices,
            "steps": [],
            "models": [],
        }
        self.write_manifest()

    def write_manifest(self) -> None:
        write_json_atomic(self.manifest_path, self.manifest)

    def execute(self, model_key: str, run_dir: Path, command: Command) -> None:
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        index = len(self.manifest["steps"]) + 1
        log_path = log_dir / f"{index:03d}_{re.sub(r'[^a-z0-9]+', '_', command.name.lower()).strip('_')}.log"
        started_at = datetime.now().astimezone().isoformat()
        record = {
            "model": model_key,
            "name": command.name,
            "argv": command.argv,
            "log": str(log_path.relative_to(self.campaign_dir)),
            "started_at": started_at,
            "status": "running",
        }
        self.manifest["steps"].append(record)
        self.write_manifest()
        environment = os.environ.copy()
        if self.cuda_visible_devices:
            environment["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                completed = subprocess.run(
                    command.argv,
                    cwd=PACKAGE_ROOT,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        except OSError as error:
            record.update(status="failed", error=str(error), completed_at=datetime.now().astimezone().isoformat())
            self.write_manifest()
            raise RuntimeError(f"Could not start {command.name}; see {log_path}") from error
        record.update(
            status="completed" if completed.returncode == 0 else "failed",
            returncode=completed.returncode,
            completed_at=datetime.now().astimezone().isoformat(),
        )
        self.write_manifest()
        if completed.returncode != 0:
            raise RuntimeError(
                f"Step {command.name} failed with exit code {completed.returncode}; "
                f"see {log_path}"
            )


def selection_checkpoint(path: Path, selector: str = "selected") -> Path:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("completed") is not True:
        raise ValueError(f"Selection has not completed: {path}")
    selected = result.get(selector)
    checkpoint = selected.get("checkpoint") if isinstance(selected, dict) else None
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
        raise ValueError(f"Selection does not name an existing checkpoint: {path}")
    return Path(checkpoint).resolve()


def selected_v19_checkpoint(path: Path) -> Path:
    return selection_checkpoint(path, selector="selected")


def run_one_model(
    *,
    manager: RunManager,
    config: dict[str, Any],
    model_key: str,
    run_stamp: str,
    campaign_dir: Path,
    python: str,
    device: str,
    split_gpus: bool,
    model_override: str | None,
    reference_override: str | None,
) -> dict[str, Any]:
    model = config["models"][model_key]
    protocol = config["protocol"]
    run_dir = campaign_dir / model_key
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts = make_artifacts(run_dir)
    record: dict[str, Any] = {"model": model_key, "status": "running"}
    manager.manifest["models"].append(record)
    manager.write_manifest()

    try:
        model_path = resolve_model_snapshot(model, override=model_override, dry_run=False)
        if not (Path(model_path) / "config.json").is_file():
            raise FileNotFoundError(f"Model snapshot lacks config.json: {model_path}")
        reference_model_path = None
        if model.get("reference_repo_id"):
            reference_spec = {
                "repo_id": model["reference_repo_id"],
                "revision": model["reference_revision"],
            }
            reference_model_path = resolve_model_snapshot(
                reference_spec,
                override=reference_override,
                dry_run=False,
            )
            if not (Path(reference_model_path) / "tokenizer.json").is_file():
                raise FileNotFoundError(
                    "Reference model snapshot lacks tokenizer.json: "
                    f"{reference_model_path}"
                )
        elif reference_override:
            raise ValueError("--reference-model-path is only valid for Llama3.1-8B-Instruct")

        record.update(
            model_snapshot=str(model_path),
            model_snapshot_revision=Path(model_path).name if not model_override else None,
            reference_snapshot=str(reference_model_path) if reference_model_path else None,
        )
        manager.write_manifest()
        quant_cache = artifacts.tokens if model["cache_kind"] == "redpajama" else artifacts.matched
        e2e_cache = artifacts.tokens if model["cache_kind"] == "redpajama" else artifacts.e2e_tokens

        for command in cache_commands(
            python=python,
            model=model,
            protocol=protocol,
            model_path=model_path,
            reference_model_path=reference_model_path,
            artifacts=artifacts,
        ):
            manager.execute(model_key, run_dir, command)

        baseline = baseline_command(
            python=python,
            model=model,
            protocol=protocol,
            model_path=model_path,
            token_cache=quant_cache,
            artifacts=artifacts,
        )
        manager.execute(model_key, run_dir, baseline)
        baseline_checkpoint = artifacts.baseline_raw
        if model["cache_kind"] == "fineweb_c4_guard":
            assignment = llama_task_assignment_command(
                python=python,
                model=model,
                model_path=model_path,
                token_cache=quant_cache,
                artifacts=artifacts,
                source_checkpoint=artifacts.baseline_raw,
            )
            manager.execute(model_key, run_dir, assignment)
            baseline_checkpoint = artifacts.baseline

        v19_commands = [
            v19_command(
                python=python,
                model=model,
                protocol=protocol,
                model_path=model_path,
                token_cache=quant_cache,
                source_checkpoint=baseline_checkpoint,
                output_dir=artifacts.scale_dir,
                run_stamp=run_stamp,
                mode="scale_only",
                active_positions=protocol["v19"]["scale_only"]["active_positions"],
                candidates_per_ranking=protocol["v19"]["scale_only"]["candidates_per_ranking"],
            ),
            v19_command(
                python=python,
                model=model,
                protocol=protocol,
                model_path=model_path,
                token_cache=quant_cache,
                source_checkpoint=baseline_checkpoint,
                output_dir=artifacts.standard_dir,
                run_stamp=run_stamp,
                mode="interval",
                active_positions=protocol["v19"]["standard"]["active_positions"],
                candidates_per_ranking=protocol["v19"]["standard"]["candidates_per_ranking"],
            ),
            v19_command(
                python=python,
                model=model,
                protocol=protocol,
                model_path=model_path,
                token_cache=quant_cache,
                source_checkpoint=baseline_checkpoint,
                output_dir=artifacts.wide_dir,
                run_stamp=run_stamp,
                mode="interval",
                active_positions=protocol["v19"]["wide"]["active_positions"],
                candidates_per_ranking=protocol["v19"]["wide"]["candidates_per_ranking"],
            ),
        ]
        for command in v19_commands:
            manager.execute(model_key, run_dir, command)
        score_command = v19_validation_command(
            python=python,
            model=model,
            protocol=protocol,
            model_path=model_path,
            token_cache=quant_cache,
            artifacts=artifacts,
            run_stamp=run_stamp,
            device=device,
            split_gpus=split_gpus,
        )
        manager.execute(model_key, run_dir, score_command)
        v19_checkpoint = selected_v19_checkpoint(artifacts.v19_selection)
        record["selected_v19_checkpoint"] = str(v19_checkpoint)
        manager.write_manifest()

        selected_endpoints: dict[str, Path] = {
            "baseline": baseline_checkpoint,
            "scale": artifacts.stage_checkpoint(artifacts.scale_dir, run_stamp),
            "v19": v19_checkpoint,
        }
        for branch in e2e_branches(model):
            source_checkpoint = {
                "baseline": baseline_checkpoint,
                "scale": selected_endpoints["scale"],
                "v19": v19_checkpoint,
            }[branch]
            for learning_rate in protocol["e2e_learning_rates"]:
                command = e2e_trial_command(
                    python=python,
                    model=model,
                    protocol=protocol,
                    model_path=model_path,
                    source_checkpoint=source_checkpoint,
                    token_cache=e2e_cache,
                    output_dir=artifacts.e2e_trial_dir(branch, learning_rate),
                    run_stamp=run_stamp,
                    learning_rate=learning_rate,
                )
                manager.execute(model_key, run_dir, command)
            selection = e2e_selection_command(
                python=python,
                artifacts=artifacts,
                branch=branch,
                run_stamp=run_stamp,
                learning_rates=protocol["e2e_learning_rates"],
            )
            manager.execute(model_key, run_dir, selection)
            selected_endpoints[f"{branch}_e2e"] = selection_checkpoint(
                artifacts.e2e_selection(branch)
            )

        endpoint_outputs: dict[str, Path] = {}
        for endpoint, checkpoint in selected_endpoints.items():
            output_dir = artifacts.final_eval_dir(endpoint)
            command = evaluation_command(
                python=python,
                model_path=model_path,
                checkpoint=checkpoint,
                output_dir=output_dir,
                protocol=protocol,
                device=device,
                split_gpus=split_gpus,
            )
            manager.execute(model_key, run_dir, command)
            endpoint_outputs[endpoint] = output_dir / "summary.json"

        record.update(
            status="completed",
            completed_at=datetime.now().astimezone().isoformat(),
            checkpoints={key: str(value) for key, value in selected_endpoints.items()},
            summaries={key: str(value) for key, value in endpoint_outputs.items()},
        )
        manager.write_manifest()
        return record
    except BaseException as error:
        record.update(
            status="failed",
            completed_at=datetime.now().astimezone().isoformat(),
            error=f"{type(error).__name__}: {error}",
        )
        manager.write_manifest()
        raise


def metric_from_summary(path: Path) -> tuple[float | None, float | None]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    evaluation = summary.get("evaluation", {})
    macro = evaluation.get("macro_average")
    ppl = None
    matched_path = summary.get("matched_ppl")
    if matched_path:
        matched = json.loads(Path(matched_path).read_text(encoding="utf-8"))
        raw_ppl = matched.get("ppl", matched.get("perplexity"))
        if raw_ppl is None and isinstance(matched.get("wikitext2"), dict):
            raw_ppl = matched["wikitext2"].get("ppl")
        ppl = float(raw_ppl) if raw_ppl is not None else None
    return ppl, float(macro) if macro is not None else None


def write_run_report(
    *,
    config: dict[str, Any],
    campaign_dir: Path,
    run_stamp: str,
    completed_models: list[dict[str, Any]],
) -> Path:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_ROOT / f"{run_stamp}_v19_e2e_reproduction.md"
    lines = [
        f"# {run_stamp} V19+E2E six-model reproduction report",
        "",
        "This report is generated automatically from the run manifest and evaluation summaries.",
        "It records the status and metrics produced by this run. Incomplete endpoints have no inferred metrics.",
        "",
        "## Objective and method",
        "",
        "For each of the six preregistered models, build the Vector-GSQ baseline, scale-only endpoint,",
        "and standard/wide V19 candidates. Select the V19 endpoint using held-out development NLL,",
        "then fine-tune the baseline, V19, and applicable scale endpoints with fixed discrete assignments",
        "and select the learning rate. Evaluate WikiText-2 test and six zero-shot lm-eval tasks after selection.",
        "",
        "## Frozen protocol",
        "",
        f"- V19: 128 calibration sequences of 2048 tokens, {config['protocol']['v19']['majorizer']},",
        f"  standard={config['protocol']['v19']['standard']['active_positions']}/"
        f"{config['protocol']['v19']['standard']['candidates_per_ranking']}, wide="
        f"{config['protocol']['v19']['wide']['active_positions']}/"
        f"{config['protocol']['v19']['wide']['candidates_per_ranking']}; scale-only uses the standard budget.",
        f"- E2E: {config['protocol']['e2e_epochs']} epochs, batch "
        f"{config['protocol']['e2e_batch_size']}, context "
        f"{config['protocol']['e2e_context_length']}, candidate learning rates "
        f"{', '.join(map(str, config['protocol']['e2e_learning_rates']))}, seed {config['protocol']['seed']}.",
        f"- Final evaluation: WikiText-2 test sequence length {config['protocol']['wiki_text_sequence_length']}; "
        f"lm-eval tasks `{', '.join(config['protocol']['lm_eval_tasks'])}`.",
        "- Full logs and JSON are saved separately for each branch. Model weights and caches are excluded from Git.",
        f"- Resolved dataset commits for this run: `{json.dumps(config['protocol'].get('resolved_dataset_revisions', {}), ensure_ascii=False)}`.",
        "",
        "## Results",
        "",
        "| Model | Endpoint | WikiText-2 test PPL (lower is better) | Six-task macro accuracy (higher is better) | Evaluation summary |",
        "|---|---|---:|---:|---|",
    ]
    for model_record in completed_models:
        for endpoint, summary_value in model_record.get("summaries", {}).items():
            summary_path = Path(summary_value)
            try:
                ppl, macro = metric_from_summary(summary_path)
                ppl_text = f"{ppl:.6f}" if ppl is not None else "Not recorded"
                macro_text = f"{macro:.6f}" if macro is not None else "Not recorded"
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                ppl_text = macro_text = f"Read failed: {error}"
            relative = summary_path.relative_to(PACKAGE_ROOT)
            lines.append(
                f"| {model_record['model']} | {endpoint} | {ppl_text} | {macro_text} | `{relative}` |"
            )
    lines.extend(
        [
            "",
            "## Run artifacts",
            "",
            f"- Campaign manifest: `{(campaign_dir / 'campaign_manifest.json').relative_to(PACKAGE_ROOT)}`",
            f"- Run directory: `{campaign_dir.relative_to(PACKAGE_ROOT)}`",
            "- The preregistration record, model cards, dataset cards, and license reviews remain separate records.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", *MODEL_CHOICES), default="all")
    parser.add_argument("--model-path", help="Optional local snapshot; only valid for one selected model")
    parser.add_argument("--reference-model-path", help="Optional local Llama-3 tokenizer snapshot")
    parser.add_argument("--run-stamp", default=datetime.now().astimezone().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", help="Optional result root, restricted to experiments/results")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used for pipeline steps")
    parser.add_argument("--device", default="cuda:0", help="Logical CUDA device used by single-device stages")
    parser.add_argument("--cuda-visible-devices", help="Optional CUDA_VISIBLE_DEVICES passed to all stages")
    parser.add_argument("--redpajama-revision", help="Hub branch/tag/commit; mutable branches are frozen to a SHA before download")
    parser.add_argument("--fineweb-revision", help="Hub branch/tag/commit for FineWeb-Edu")
    parser.add_argument("--c4-revision", help="Hub branch/tag/commit for the C4 guard split")
    parser.add_argument("--wikitext-revision", help="Hub branch/tag/commit for WikiText-2 test")
    parser.add_argument("--split-gpus", action="store_true", help="Split final/dev scoring across exactly two visible GPUs")
    parser.add_argument("--dry-run", action="store_true", help="Print the pipeline plan without creating files or launching processes")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"\d{8}_\d{6}", args.run_stamp):
        raise ValueError("--run-stamp must use YYYYMMDD_HHMMSS")
    if args.model == "all" and args.model_path:
        raise ValueError("--model-path is only valid when selecting one model")
    if args.model == "all" and args.reference_model_path:
        raise ValueError("--reference-model-path is only valid when selecting one model")
    if args.reference_model_path and args.model != "llama3.1-8b-instruct":
        raise ValueError("--reference-model-path is only valid for Llama3.1-8B-Instruct")
    if args.device != "cuda:0":
        raise ValueError(
            "Training stages use logical cuda:0; select its physical GPU with "
            "--cuda-visible-devices"
        )
    if args.cuda_visible_devices:
        if not re.fullmatch(r"\d+(?:,\d+)*", args.cuda_visible_devices):
            raise ValueError("--cuda-visible-devices must be comma-separated physical GPU indices")
        if len(set(args.cuda_visible_devices.split(","))) != len(args.cuda_visible_devices.split(",")):
            raise ValueError("--cuda-visible-devices cannot contain duplicate indices")
    if args.split_gpus:
        visible = args.cuda_visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or len(visible.split(",")) != 2:
            raise ValueError("--split-gpus requires exactly two visible GPUs; set --cuda-visible-devices")


def dry_run(args: argparse.Namespace, config: dict[str, Any]) -> None:
    models = MODEL_CHOICES if args.model == "all" else (args.model,)
    output_root = output_root_path(args.output_root)
    campaign_dir = output_root / args.run_stamp
    for model_key in models:
        model = config["models"][model_key]
        model_path = (
            str(Path(args.model_path).expanduser().resolve())
            if args.model_path
            else f"HF_SNAPSHOT({model['repo_id']}@{model['revision']})"
        )
        reference_path = None
        if model.get("reference_repo_id"):
            reference_path = (
                str(Path(args.reference_model_path).expanduser().resolve())
                if args.reference_model_path
                else f"HF_SNAPSHOT({model['reference_repo_id']}@{model['reference_revision']})"
            )
        commands = model_plan(
            config=config,
            model_key=model_key,
            run_stamp=args.run_stamp,
            model_path=model_path,
            reference_model_path=reference_path,
            run_dir=campaign_dir / model_key,
            python=args.python,
            device=args.device,
            split_gpus=args.split_gpus,
        )
        print(f"[{model_key}] {model['repo_id']} @ {model['revision']}")
        for index, command in enumerate(commands, start=1):
            print(f"  {index:02d}. {command.name}: {display_command(command)}")
    print("DRY RUN ONLY: no model download, data download, training, evaluation, or output files were created.")


def run_campaign(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    models = list(MODEL_CHOICES if args.model == "all" else (args.model,))
    output_root = output_root_path(args.output_root)
    campaign_dir = output_root / args.run_stamp
    if campaign_dir.exists() or campaign_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {campaign_dir}")
    campaign_dir.mkdir(parents=True, exist_ok=False)
    cuda_visible = args.cuda_visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES")
    manager = RunManager(
        campaign_dir,
        python=args.python,
        cuda_visible_devices=cuda_visible,
    )
    manager.manifest.update(
        run_stamp=args.run_stamp,
        config_path=str(CONFIG_PATH.relative_to(PACKAGE_ROOT)),
        config_sha256=sha256_file(CONFIG_PATH),
        requested_models=models,
        device=args.device,
        split_gpus=args.split_gpus,
        protocol=config["protocol"],
    )
    manager.write_manifest()
    try:
        config, data_revisions = resolve_dataset_revisions(
            config,
            models,
            {
                "redpajama": args.redpajama_revision,
                "fineweb_edu": args.fineweb_revision,
                "c4_guard": args.c4_revision,
                "wikitext2": args.wikitext_revision,
            },
        )
        config["protocol"]["resolved_dataset_revisions"] = data_revisions
        manager.manifest.update(
            requested_dataset_revisions={
                "redpajama": args.redpajama_revision,
                "fineweb_edu": args.fineweb_revision,
                "c4_guard": args.c4_revision,
                "wikitext2": args.wikitext_revision,
            },
            resolved_dataset_revisions=data_revisions,
            protocol=config["protocol"],
        )
        manager.write_manifest()
        for model_key in models:
            override = args.model_path if args.model == model_key else None
            reference_override = args.reference_model_path if args.model == model_key else None
            run_one_model(
                manager=manager,
                config=config,
                model_key=model_key,
                run_stamp=args.run_stamp,
                campaign_dir=campaign_dir,
                python=args.python,
                device=args.device,
                split_gpus=args.split_gpus,
                model_override=override,
                reference_override=reference_override,
            )
        manager.manifest.update(
            status="completed",
            completed_at=datetime.now().astimezone().isoformat(),
        )
        manager.write_manifest()
        report = write_run_report(
            config=config,
            campaign_dir=campaign_dir,
            run_stamp=args.run_stamp,
            completed_models=manager.manifest["models"],
        )
        manager.manifest["report"] = str(report.relative_to(PACKAGE_ROOT))
        manager.write_manifest()
        return report
    except BaseException as error:
        manager.manifest.update(
            status="failed",
            completed_at=datetime.now().astimezone().isoformat(),
            error=f"{type(error).__name__}: {error}",
        )
        manager.write_manifest()
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    config = load_config()
    if args.dry_run:
        dry_run(args, config)
        return 0
    report = run_campaign(args, config)
    print(f"Completed. Reproduction report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
