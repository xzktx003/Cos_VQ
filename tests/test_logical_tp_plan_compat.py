from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, modeling_utils

from experiments.qwen_qtip import qtip_runtime


def test_missing_registry_supports_real_tiny_llama_on_cpu(monkeypatch):
    config = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    )
    monkeypatch.setattr(modeling_utils, "ALL_PARALLEL_STYLES", None)
    qtip_runtime.install_tp_plan_compat(config)
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3]])).logits
    assert logits.shape == (1, 3, 32)
    assert torch.isfinite(logits).all()
    assert next(model.parameters()).device.type == "cpu"


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_unknown_styles_fail_closed(monkeypatch, model_type):
    monkeypatch.setattr(modeling_utils, "ALL_PARALLEL_STYLES", None)
    config = SimpleNamespace(model_type=model_type, base_model_tp_plan={"x": "unknown"})
    with pytest.raises(ValueError, match="Unsupported"):
        qtip_runtime.install_tp_plan_compat(config)
    assert modeling_utils.ALL_PARALLEL_STYLES is None


def test_existing_registry_and_qwen_only_api_are_preserved(monkeypatch):
    registry = {"existing"}
    monkeypatch.setattr(modeling_utils, "ALL_PARALLEL_STYLES", registry)
    qtip_runtime.install_tp_plan_compat(LlamaConfig())
    assert modeling_utils.ALL_PARALLEL_STYLES is registry
    with pytest.raises(ValueError, match="restricted to Qwen3"):
        qtip_runtime.install_qwen3_tp_plan_compat(LlamaConfig())


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="restricted"):
        qtip_runtime.install_tp_plan_compat(SimpleNamespace(model_type="unknown"))
