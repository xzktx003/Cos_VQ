from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.qwen_qtip.qwen3_qtip_adapter import (
    QWEN3_QTIP_LINEAR_SPECS,
    replace_qwen3_linears,
    validate_qwen3_qtip_config,
)
from experiments.qwen_qtip.qtip_artifacts import audit_qtip_layer_artifacts


class _DummyLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False)


class _DummyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = _DummyLinear(64, 96)
        self.k_proj = _DummyLinear(64, 32)
        self.v_proj = _DummyLinear(64, 32)
        self.o_proj = _DummyLinear(96, 64)


class _DummyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = _DummyLinear(64, 128)
        self.up_proj = _DummyLinear(64, 128)
        self.down_proj = _DummyLinear(128, 64)


class _DummyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _DummyAttention()
        self.mlp = _DummyMlp()


class _DummyModel(nn.Module):
    def __init__(self, layers: int = 2) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_DummyLayer() for _ in range(layers)])


class _RecordingQuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        td_x,
        td_y,
        L,
        K,
        V,
        tlut_bits,
        decode_mode,
        *,
        bias=False,
        dtype=torch.float16,
    ) -> None:
        super().__init__()
        self.args = {
            "in_features": in_features,
            "out_features": out_features,
            "td_x": td_x,
            "td_y": td_y,
            "L": L,
            "K": K,
            "V": V,
            "tlut_bits": tlut_bits,
            "decode_mode": decode_mode,
            "bias": bias,
            "dtype": dtype,
        }


def _config(*, skip_list=None):
    return SimpleNamespace(
        model_type="qwen3",
        quip_params={
            "td_x": 16,
            "td_y": 16,
            "L": 16,
            "K": 2,
            "V": 2,
            "tlut_bits": 9,
            "decode_mode": "quantlut_sym",
            "skip_list": skip_list,
        },
    )


def test_qwen3_qtip_adapter_covers_exactly_seven_decoder_linears() -> None:
    assert tuple(spec.save_name for spec in QWEN3_QTIP_LINEAR_SPECS) == (
        "q",
        "k",
        "v",
        "o",
        "up",
        "gate",
        "down",
    )


def test_validate_qwen3_qtip_config_rejects_wrong_family() -> None:
    config = _config()
    config.model_type = "llama"

    with pytest.raises(ValueError, match="model_type='qwen3'"):
        validate_qwen3_qtip_config(config)


def test_validate_qwen3_qtip_config_requires_all_quantizer_fields() -> None:
    config = _config()
    del config.quip_params["decode_mode"]

    with pytest.raises(ValueError, match="decode_mode"):
        validate_qwen3_qtip_config(config)


def test_replace_qwen3_linears_preserves_shapes_dtype_and_skip_list() -> None:
    model = _DummyModel(layers=2).to(torch.bfloat16)
    config = _config(skip_list=["1_gate"])

    replaced = replace_qwen3_linears(
        model,
        config,
        quantized_linear_cls=_RecordingQuantizedLinear,
    )

    assert len(replaced) == 13
    assert "model.layers.1.mlp.gate_proj" not in replaced
    assert isinstance(model.model.layers[1].mlp.gate_proj, _DummyLinear)

    q_proj = model.model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, _RecordingQuantizedLinear)
    assert q_proj.args["in_features"] == 64
    assert q_proj.args["out_features"] == 96
    assert q_proj.args["dtype"] == torch.bfloat16
    assert q_proj.args["bias"] is False


def test_replace_qwen3_linears_rejects_biased_projection() -> None:
    model = _DummyModel(layers=1)
    model.model.layers[0].self_attn.q_proj = nn.Linear(64, 96, bias=True)

    with pytest.raises(ValueError, match="bias-free"):
        replace_qwen3_linears(
            model,
            _config(),
            quantized_linear_cls=_RecordingQuantizedLinear,
        )


def test_qtip_artifact_audit_requires_gate_projection(tmp_path) -> None:
    for suffix in ("q", "k", "v", "o", "up", "down", "layernorm"):
        (tmp_path / f"0_{suffix}.pt").touch()

    with pytest.raises(FileNotFoundError, match="0_gate.pt"):
        audit_qtip_layer_artifacts(tmp_path, expected_layers=1)

    (tmp_path / "0_gate.pt").touch()
    audit = audit_qtip_layer_artifacts(tmp_path, expected_layers=1)

    assert audit["quantized_layers"] == 1
    assert audit["quantized_linears"] == 7
    assert len(audit["artifacts"]) == 8


def test_upstream_qtip_resume_check_includes_gate_projection() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "QTIP_official_20260828/quantize_llama/quantize_finetune_llama.py"
    ).read_text(encoding="utf-8")

    assert "['q', 'k', 'v', 'o', 'up', 'gate', 'down', 'layernorm']" in source
