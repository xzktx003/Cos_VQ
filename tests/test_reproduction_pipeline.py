from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys

import pytest

from scripts import reproduce_v19_e2e as reproduction


STAMP = "20300102_030405"


def _plan(model_key: str, tmp_path: Path):
    config = reproduction.load_config()
    model = config["models"][model_key]
    reference = "REFERENCE_MODEL" if model.get("reference_repo_id") else None
    commands = reproduction.model_plan(
        config=config,
        model_key=model_key,
        run_stamp=STAMP,
        model_path="TARGET_MODEL",
        reference_model_path=reference,
        run_dir=tmp_path / model_key,
        python="python",
        device="cuda:0",
        split_gpus=False,
    )
    return config, model, commands


def test_six_model_plan_keeps_baseline_v19_selection_and_e2e_endpoints(tmp_path):
    config, model, commands = _plan("qwen3-4b", tmp_path)
    by_name = {}
    for command in commands:
        by_name.setdefault(command.name, []).append(command.argv)

    assert len(config["models"]) == 6
    baseline = by_name["run_prefix_vector_gsq"][0]
    assert baseline[baseline.index("--stage") + 1] == "gsq"
    assert baseline[baseline.index("--max-layers") + 1] == str(model["layers"])
    assert baseline[baseline.index("--gptq-token-offset") + 1] == "4224"
    v19 = by_name["v19_model_driver"]
    assert [args[args.index("--mode") + 1] for args in v19] == [
        "scale_only",
        "interval",
        "interval",
    ]
    assert [args[args.index("--active-positions") + 1] for args in v19] == [
        "8",
        "8",
        "16",
    ]
    trials = by_name["finetune_hard_codebook_e2e"]
    assert len(trials) == 4
    assert all("--tune-group-scales" not in args for args in trials)
    evaluations = by_name["evaluate_logical_checkpoint_lm_eval"]
    assert len(evaluations) == 5
    assert all("--matched-ppl" in args and "--log-samples" in args for args in evaluations)
    assert all(
        args[args.index("--wikitext2-revision") + 1]
        == config["protocol"]["datasets"]["wikitext2"]["revision"]
        for args in evaluations
    )
    assert all(
        args[args.index("--tasks") + 1] == ",".join(config["protocol"]["lm_eval_tasks"])
        for args in evaluations
    )


def test_llama31_plan_includes_public_cache_guard_group160_and_selected_scale_e2e(tmp_path):
    _config, model, commands = _plan("llama3.1-8b-instruct", tmp_path)
    by_name = {}
    for command in commands:
        by_name.setdefault(command.name, []).append(command.argv)

    matched_cache = by_name["build_matched_llama_assignment_cache"][0]
    assert matched_cache[matched_cache.index("--c4-source") + 1] == "download"
    assert matched_cache[matched_cache.index("--guard-sequences") + 1] == "640"
    baseline = by_name["run_prefix_vector_gsq"][0]
    assert baseline[baseline.index("--gsq-objective") + 1] == "staged"
    assert "--staged-joint-group-scales" in baseline
    assert baseline[baseline.index("--scale-group-size") + 1] == "160"
    assignment = by_name["calibrate_task_local_assignments"][0]
    train_rows = assignment[
        assignment.index("--text-train-rows") + 1 : assignment.index("--text-validation-rows")
    ]
    validation_rows = assignment[
        assignment.index("--text-validation-rows") + 1 : assignment.index("--text-audit-rows")
    ]
    audit_rows = assignment[
        assignment.index("--text-audit-rows") + 1 : assignment.index("--text-sequence-length")
    ]
    assert train_rows[0] == "0" and train_rows[-1] == "4095"
    assert validation_rows[0] == "4736" and validation_rows[-1] == "4863"
    assert audit_rows[0] == "4864" and audit_rows[-1] == "5375"
    assert model["scale_e2e"] is True
    assert len(by_name["finetune_hard_codebook_e2e"]) == 6
    assert all("--tune-group-scales" in args for args in by_name["finetune_hard_codebook_e2e"])


def test_dry_run_prints_plan_without_creating_outputs_or_invoking_processes(monkeypatch):
    output_root = reproduction.RESULTS_ROOT / STAMP
    assert not output_root.exists()
    monkeypatch.setattr(
        reproduction.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("dry-run must never start a child process"),
    )
    output = StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    assert reproduction.main(["--model", "qwen3-4b", "--run-stamp", STAMP, "--dry-run"]) == 0
    assert "DRY RUN ONLY" in output.getvalue()
    assert "--matched-ppl" in output.getvalue()
    assert not output_root.exists()


def test_pinned_dataset_revision_is_recorded_without_hub_resolution(monkeypatch):
    config = reproduction.load_config()
    pinned = "a" * 40
    config["protocol"]["datasets"]["redpajama"]["revision"] = pinned
    config["protocol"]["datasets"]["wikitext2"]["revision"] = pinned
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    resolved, revisions = reproduction.resolve_dataset_revisions(
        config,
        ["qwen3-4b"],
        {
            "redpajama": None,
            "fineweb_edu": None,
            "c4_guard": None,
            "wikitext2": None,
        },
    )
    assert revisions == {"redpajama": pinned, "wikitext2": pinned}
    assert resolved["protocol"]["datasets"]["redpajama"]["requested_revision"] == pinned
    assert resolved["protocol"]["datasets"]["redpajama"]["revision"] == pinned
    assert resolved["protocol"]["datasets"]["wikitext2"]["revision"] == pinned


def test_all_pipeline_modules_are_bundled(tmp_path):
    config = reproduction.load_config()
    for model_key in reproduction.MODEL_CHOICES:
        _config, model, commands = _plan(model_key, tmp_path)
        for command in commands:
            module_index = command.argv.index("-m") + 1
            module_path = reproduction.PACKAGE_ROOT.joinpath(
                *command.argv[module_index].split(".")
            )
            assert module_path.with_suffix(".py").is_file() or (
                module_path / "__init__.py"
            ).is_file(), (model_key, command.name, command.argv[module_index])


def test_every_published_model_has_a_shell_accuracy_launcher():
    launcher_dir = reproduction.PACKAGE_ROOT / "scripts" / "reproduce_accuracy"
    launchers = {
        "qwen3-4b": "qwen3_4b.sh",
        "qwen3-8b": "qwen3_8b.sh",
        "qwen3-14b": "qwen3_14b.sh",
        "qwen3-32b": "qwen3_32b.sh",
        "llama2-7b": "llama2_7b.sh",
        "llama3.1-8b-instruct": "llama3_1_8b_instruct.sh",
    }
    assert set(launchers) == set(reproduction.MODEL_CHOICES)
    for model, filename in launchers.items():
        text = (launcher_dir / filename).read_text(encoding="utf-8")
        assert "_run_model.sh" in text
        assert model in text


def test_public_reproduction_documentation_is_bundled():
    required = (
        reproduction.PACKAGE_ROOT / "README.md",
        reproduction.PACKAGE_ROOT / "docs" / "V19_E2E_REPRODUCTION.md",
        reproduction.PACKAGE_ROOT / "docs" / "ARTIFACTS_AND_RESULTS.md",
        reproduction.PACKAGE_ROOT / "scripts" / "reproduce_accuracy" / "README.md",
        reproduction.CONFIG_PATH,
    )
    assert all(path.is_file() for path in required)


def test_output_root_cannot_escape_package_results():
    with pytest.raises(ValueError, match="must stay under"):
        reproduction.output_root_path("../../outside")


@pytest.mark.parametrize("devices", [None, "0", "0,1,2", "0,0"])
def test_split_gpu_mode_requires_exactly_two_unique_devices(devices, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    argv = ["--model", "qwen3-4b", "--split-gpus"]
    if devices is not None:
        argv.extend(("--cuda-visible-devices", devices))
    args = reproduction.parse_args(argv)
    with pytest.raises(ValueError, match="visible GPU|duplicate"):
        reproduction.validate_args(args)
