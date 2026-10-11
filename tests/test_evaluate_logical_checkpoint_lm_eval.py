import pytest
import torch

from experiments.remote_runner.evaluate_logical_checkpoint_lm_eval import (
    extract_paired_samples,
    extract_summary,
)


def test_checkpoint_audit_counts_residual_and_reconstructs_each_linear(monkeypatch):
    from experiments.remote_runner import evaluate_logical_checkpoint_lm_eval as evaluator
    from experiments.nowag_d6.logical_state import LINEAR_NAMES

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    modules = {name: torch.nn.Linear(3, 2, bias=False) for name in LINEAR_NAMES}
    monkeypatch.setattr(evaluator, 'get_linear_modules', lambda *_: modules)
    state = {
        'shape': torch.tensor([2, 3]), 'vector_dim': torch.tensor(1),
        'assignments': torch.zeros(6, dtype=torch.int32),
        'assignment_bits': torch.tensor(1),
        'codebook': torch.ones(2, 1, dtype=torch.float16),
        'row_norm': torch.ones(2, dtype=torch.float16),
        'col_norm': torch.ones(3, dtype=torch.float16),
        'residual_A': torch.ones(1, 3, dtype=torch.float16),
        'residual_B': torch.ones(2, 1, dtype=torch.float16),
    }
    checkpoint = {'layers': {'0': {name: dict(state) for name in LINEAR_NAMES}}}
    audit = evaluator.reconstruct_checkpoint(model, checkpoint, torch.device('cpu'))
    assert audit['residual_parameters'] == 7 * 5
    assert audit['residual_bits'] == 7 * 5 * 16
    assert audit['logical_effective_bpp'] == (6 + (7 + 5) * 16) / 6
    assert all(item['residual_rank'] == 1 for item in audit['per_linear'])
    for module in modules.values():
        torch.testing.assert_close(module.weight, torch.full((2, 3), 2.))

    del checkpoint['layers']['0'][LINEAR_NAMES[0]]['residual_B']
    with pytest.raises(ValueError, match='residual'):
        evaluator.reconstruct_checkpoint(model, checkpoint, torch.device('cpu'))


def test_optional_clean_c4_fails_before_creating_output(tmp_path, monkeypatch):
    from experiments.remote_runner import evaluate_logical_checkpoint_lm_eval as evaluator

    output_dir = tmp_path / "clean-c4"
    monkeypatch.setattr(
        evaluator,
        "parse_args",
        lambda: type("Args", (), {"clean_c4": True, "output_dir": str(output_dir)})(),
    )
    with pytest.raises(RuntimeError, match="not included in this standalone package"):
        evaluator.main()
    assert not output_dir.exists()


def _sample(doc_id, metric, value, suffix=""):
    return {
        "doc_id": doc_id,
        "filter": "none",
        "doc_hash": f"doc-{doc_id}{suffix}",
        "prompt_hash": f"prompt-{doc_id}{suffix}",
        "target_hash": f"target-{doc_id}{suffix}",
        metric: value,
    }


def test_extract_paired_samples_keeps_stable_keys_and_primary_scores():
    raw = {
        "samples": {
            "arc_challenge": [
                _sample(1, "acc_norm", 1.0),
                _sample(0, "acc_norm", 0.0),
            ],
            "lambada_openai": [
                _sample(0, "acc", 1.0),
            ],
        }
    }

    result = extract_paired_samples(
        raw,
        ["arc_challenge", "lambada_openai"],
    )

    assert result["schema_version"] == 1
    assert result["tasks"]["arc_challenge"]["metric"] == "acc_norm"
    assert result["tasks"]["arc_challenge"]["sample_count"] == 2
    assert result["tasks"]["arc_challenge"]["samples"] == [
        {
            "doc_id": 0,
            "doc_hash": "doc-0",
            "prompt_hash": "prompt-0",
            "target_hash": "target-0",
            "score": 0.0,
        },
        {
            "doc_id": 1,
            "doc_hash": "doc-1",
            "prompt_hash": "prompt-1",
            "target_hash": "target-1",
            "score": 1.0,
        },
    ]
    assert result["tasks"]["lambada_openai"]["metric"] == "acc"


def test_extract_paired_samples_rejects_missing_task_samples():
    with pytest.raises(ValueError, match="missing task arc_easy"):
        extract_paired_samples({"samples": {}}, ["arc_easy"])


def test_extract_paired_samples_accepts_lm_eval_044_default_filter_schema():
    legacy = _sample(0, 'acc_norm', 1.0)
    del legacy['filter']  # Real lm_eval 0.4.4 results omit the default filter.
    explicit_other = _sample(1, 'acc_norm', 0.0)
    explicit_other['filter'] = 'custom'
    raw = {'samples': {'arc_easy': [legacy, explicit_other]}}
    result = extract_paired_samples(raw, ['arc_easy'])['tasks']['arc_easy']
    assert result['sample_count'] == 1
    assert result['samples'][0]['score'] == 1.0


def test_extract_paired_samples_rejects_duplicate_pairing_keys():
    duplicate = _sample(0, "acc_norm", 1.0)
    raw = {"samples": {"arc_challenge": [duplicate, dict(duplicate)]}}

    with pytest.raises(ValueError, match="duplicate pairing key"):
        extract_paired_samples(raw, ["arc_challenge"])


def test_extract_paired_samples_rejects_non_binary_accuracy_score():
    raw = {
        "samples": {
            "winogrande": [_sample(0, "acc", 0.25)],
        }
    }

    with pytest.raises(ValueError, match="binary accuracy score"):
        extract_paired_samples(raw, ["winogrande"])


def test_extended_tasks_extract_summary_handles_boolq_mmlu_and_ifeval():
    raw = {
        "results": {
            "boolq": {"acc,none": 0.73},
            "mmlu_anatomy": {"acc,none": 0.61},
            "ifeval": {
                "prompt_level_strict_acc,none": 0.42,
                "inst_level_strict_acc,none": 0.51,
            },
        }
    }

    summary = extract_summary(raw, ["boolq", "mmlu_anatomy", "ifeval"])

    assert summary["tasks"]["boolq"]["metric"] == "acc,none"
    assert summary["tasks"]["boolq"]["value"] == 0.73
    assert summary["tasks"]["mmlu_anatomy"]["metric"] == "acc,none"
    assert summary["tasks"]["mmlu_anatomy"]["value"] == 0.61
    assert summary["tasks"]["ifeval"]["metric"] == "prompt_level_strict_acc,none"
    assert summary["tasks"]["ifeval"]["value"] == 0.42
    assert summary["macro_average"] == pytest.approx((0.73 + 0.61 + 0.42) / 3)


def test_mmlu_group_summary_aggregates_subtasks_when_group_metric_is_absent():
    raw = {
        "results": {
            "mmlu_abstract_algebra": {"acc,none": 0.25},
            "mmlu_anatomy": {"acc,none": 0.75},
            "boolq": {"acc,none": 0.8},
        }
    }

    summary = extract_summary(raw, ["mmlu", "boolq"])

    assert summary["tasks"]["mmlu"]["metric"] == "acc,none"
    assert summary["tasks"]["mmlu"]["value"] == pytest.approx(0.5)
    assert summary["tasks"]["mmlu"]["subtask_count"] == 2
    assert summary["tasks"]["mmlu"]["subtasks"] == [
        "mmlu_abstract_algebra",
        "mmlu_anatomy",
    ]
    assert summary["macro_average"] == pytest.approx((0.5 + 0.8) / 2)


def test_extract_paired_samples_skips_unsupported_group_and_generative_tasks():
    raw = {
        "samples": {
            "boolq": [_sample(0, "acc", 1.0)],
            "ifeval": [
                _sample(0, "prompt_level_strict_acc", 0.5),
            ],
        }
    }

    result = extract_paired_samples(raw, ["boolq", "mmlu", "ifeval"])

    assert result["tasks"]["boolq"]["sample_count"] == 1
    assert result["skipped_tasks"] == {
        "mmlu": "paired samples require concrete binary-accuracy task samples",
        "ifeval": "paired samples require binary accuracy scores",
    }


def test_extract_paired_samples_allows_binary_mmlu_subtasks():
    raw = {
        "samples": {
            "mmlu_anatomy": [
                _sample(0, "acc", 1.0),
                _sample(1, "acc", 0.0),
            ],
        }
    }

    result = extract_paired_samples(raw, ["mmlu_anatomy"])

    assert result["tasks"]["mmlu_anatomy"]["metric"] == "acc"
    assert result["tasks"]["mmlu_anatomy"]["sample_count"] == 2
    assert result["skipped_tasks"] == {}


def test_parse_args_records_apply_chat_template(monkeypatch):
    from experiments.remote_runner import evaluate_logical_checkpoint_lm_eval as evaluator

    arguments = [
        "evaluator",
        "--model",
        "m",
        "--checkpoint",
        "c",
        "--output-dir",
        "o",
        "--tasks",
        "ifeval",
        "--apply-chat-template",
    ]
    monkeypatch.setattr("sys.argv", arguments)

    args = evaluator.parse_args()

    assert args.tasks == "ifeval"
    assert args.apply_chat_template is True


def test_parse_args_records_clean_c4_protocol(monkeypatch):
    from experiments.remote_runner import evaluate_logical_checkpoint_lm_eval as evaluator

    arguments = [
        "evaluator",
        "--model",
        "m",
        "--checkpoint",
        "c",
        "--output-dir",
        "o",
        "--clean-c4",
        "--clean-c4-cache-dir",
        "cache",
        "--clean-c4-batch-size",
        "2",
    ]
    monkeypatch.setattr("sys.argv", arguments)

    args = evaluator.parse_args()

    assert args.clean_c4 is True
    assert args.clean_c4_cache_dir == "cache"
    assert args.clean_c4_batch_size == 2
