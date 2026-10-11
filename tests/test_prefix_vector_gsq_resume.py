from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import LlamaConfig

from experiments.nowag_d6.diagnose_block_vector_gsq import LINEAR_NAMES
from experiments.nowag_d6 import run_prefix_vector_gsq as runner


class _RecordingDecoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_mask: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        **_kwargs: object,
    ) -> tuple[torch.Tensor]:
        self.attention_mask = attention_mask
        return (hidden_states,)


class _DummyRotary(nn.Module):
    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (hidden_states.shape[0], hidden_states.shape[1], 2)
        return (
            torch.ones(shape, dtype=hidden_states.dtype),
            torch.zeros(shape, dtype=hidden_states.dtype),
        )


def _logical_state(*, group_scale: bool = True) -> dict[str, torch.Tensor]:
    state = {
        "assignments": torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        "codebook": torch.arange(8, dtype=torch.float16).reshape(4, 2),
        "row_norm": torch.tensor([1.0, 2.0], dtype=torch.float16),
        "col_norm": torch.tensor([1.0, 1.5, 2.0, 2.5], dtype=torch.float16),
        "shape": torch.tensor([2, 4], dtype=torch.int32),
        "vector_dim": torch.tensor(2, dtype=torch.int32),
        "assignment_bits": torch.tensor(2, dtype=torch.int32),
    }
    if group_scale:
        state["group_scale"] = torch.ones(2, 2, dtype=torch.float16)
        state["group_size"] = torch.tensor(2, dtype=torch.int32)
    return state


def _checkpoint(layer_count: int = 2) -> dict[str, object]:
    return {
        "metadata": {
            "model": "/models/llama",
            "model_family": "llama",
            "stage": "gsq",
            "seed": 0,
            "vector_dim": 2,
            "bits_per_weight": 1.0,
            "gsq_objective": "staged",
            "train_sequences": 16,
            "validation_sequences": 4,
            "calib_sequences": 8,
        },
        "layers": {
            str(index): {name: _logical_state() for name in LINEAR_NAMES}
            for index in range(layer_count)
        },
    }


def _result(layer_count: int = 2) -> dict[str, object]:
    return {
        "model_family": "llama",
        "layers": [{"layer": index} for index in range(layer_count)],
        "prefix_ppl": {},
    }


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        model="/models/llama",
        stage="gsq",
        seed=0,
        vector_dim=2,
        bits_per_weight=1.0,
        gsq_objective="staged",
        train_sequences=16,
        validation_sequences=4,
        calib_sequences=8,
        max_layers=4,
    )


def test_resume_budget_includes_preserved_residual_factors() -> None:
    checkpoint = _checkpoint()
    for layer in checkpoint['layers'].values():
        for state in layer.values():
            state['residual_A'] = torch.ones(1, 4, dtype=torch.float16)
            state['residual_B'] = torch.ones(2, 1, dtype=torch.float16)
    baseline = runner.validate_resume_artifacts(_checkpoint(), _result(), _args(), 'llama')
    actual = runner.validate_resume_artifacts(checkpoint, _result(), _args(), 'llama')
    assert actual.total_logical_bits == baseline.total_logical_bits + 2 * 7 * 6 * 16
    assert actual.total_quantized_weights == baseline.total_quantized_weights
    layer = checkpoint['layers']['0']
    modules = {name: torch.nn.Linear(4, 2, bias=False) for name in LINEAR_NAMES}
    runner.restore_logical_layer_weights(modules, layer, torch.device('cpu'))
    baseline_layer = _checkpoint()['layers']['0']
    for name in LINEAR_NAMES:
        expected = runner.reconstruct_logical_state(baseline_layer[name], torch.device('cpu')) + 1
        torch.testing.assert_close(modules[name].weight, expected)
        assert 'residual_A' in layer[name] and 'residual_B' in layer[name]


def test_hidden_block_replay_preserves_causal_attention() -> None:
    config = LlamaConfig(
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
    )
    config._attn_implementation = "eager"
    block = _RecordingDecoderBlock()
    model = SimpleNamespace(
        config=config,
        model=SimpleNamespace(layers=[block], rotary_emb=_DummyRotary()),
    )
    hidden = torch.zeros(1, 4, 4)

    output = runner.run_hidden_block_output(model, 0, {}, hidden, weights=None)

    assert torch.equal(output, hidden)
    assert block.attention_mask is not None
    assert block.attention_mask.shape == (1, 1, 4, 4)
    assert block.attention_mask[0, 0, 3, 0] == 0
    assert block.attention_mask[0, 0, 0, 1] < -1e20


def test_resume_artifacts_require_contiguous_matching_completed_layers() -> None:
    checkpoint = _checkpoint()
    result = _result()

    summary = runner.validate_resume_artifacts(
        checkpoint,
        result,
        _args(),
        model_family="llama",
    )

    assert summary.completed_layers == 2
    assert summary.total_quantized_weights == 2 * len(LINEAR_NAMES) * 8
    expected_state_bits = 4 * 2 + 4 * 2 * 16 + (2 + 4) * 16 + 2 * 2 * 16
    assert summary.total_logical_bits == (2 * len(LINEAR_NAMES) * expected_state_bits)

    missing = deepcopy(checkpoint)
    missing["layers"].pop("0")
    with pytest.raises(ValueError, match="contiguous"):
        runner.validate_resume_artifacts(
            missing,
            result,
            _args(),
            model_family="llama",
        )

    mismatched_result = _result(layer_count=1)
    with pytest.raises(ValueError, match="disagree"):
        runner.validate_resume_artifacts(
            checkpoint,
            mismatched_result,
            _args(),
            model_family="llama",
        )


def test_resume_artifacts_reject_method_configuration_changes() -> None:
    args = _args()
    args.vector_dim = 6
    with pytest.raises(ValueError, match="vector_dim"):
        runner.validate_resume_artifacts(
            _checkpoint(),
            _result(),
            args,
            model_family="llama",
        )


def test_restore_logical_layer_copies_exact_deployable_weights() -> None:
    modules = {name: torch.nn.Linear(4, 2, bias=False) for name in LINEAR_NAMES}
    checkpoint_layer = {name: _logical_state() for name in LINEAR_NAMES}

    runner.restore_logical_layer_weights(
        modules,
        checkpoint_layer,
        torch.device("cpu"),
    )

    for name, module in modules.items():
        expected = runner.reconstruct_logical_state(
            checkpoint_layer[name],
            torch.device("cpu"),
        ).to(module.weight.dtype)
        assert torch.equal(module.weight, expected)

    missing = dict(checkpoint_layer)
    missing.pop(LINEAR_NAMES[-1])
    with pytest.raises(ValueError, match="Linear set"):
        runner.restore_logical_layer_weights(
            modules,
            missing,
            torch.device("cpu"),
        )


def test_streamed_stage_target_uses_dense_teacher_without_cache(monkeypatch) -> None:
    hidden = torch.randn(2, 3, 4)
    expected = hidden + 7
    calls = []

    def fake_stage_output(
        model,
        layer_index,
        modules,
        hidden_states,
        weights,
        objective,
    ):
        calls.append((layer_index, weights, objective))
        return hidden_states + 7

    monkeypatch.setattr(runner, "run_hidden_stage_output", fake_stage_output)
    target = runner.stage_training_target_batch(
        model=object(),
        layer_index=3,
        modules={},
        hidden_batch=hidden,
        cached_targets=torch.empty(0),
        batch_indices=torch.tensor([0, 1]),
        objective="attention",
        stream_dense_targets=True,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(target, expected.float())
    assert calls == [(3, None, "attention")]
    assert not target.requires_grad


def test_cached_stage_target_does_not_recompute_teacher(monkeypatch) -> None:
    cached = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("dense teacher should not be recomputed")

    monkeypatch.setattr(runner, "run_hidden_stage_output", fail_if_called)
    target = runner.stage_training_target_batch(
        model=object(),
        layer_index=0,
        modules={},
        hidden_batch=torch.empty(1, 3, 4),
        cached_targets=cached,
        batch_indices=torch.tensor([1]),
        objective="block",
        stream_dense_targets=False,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(target, cached[1:2].float())

