from types import SimpleNamespace

import torch
import pytest
from torch import nn

from src.models.qwen3 import Qwen3Wrapper


class _ToyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 3, bias=False)

    def forward(self, x):
        return self.proj(x)


class _ToyQwen(Qwen3Wrapper):
    def __init__(self):
        self.current_layer_idx = 0
        self.layer_prefix = "model.layers"
        self.layer = SimpleNamespace(mlp=_ToyMlp())
        self.loss_fn = nn.MSELoss()
        self.dense_mlp_inputs_precomputed = True

    def get_layer_module(self, idx):
        assert idx == 0
        return self.layer

    def get_mlp_output(self, batch):
        return batch + self.layer.mlp(batch)


def test_precomputed_dense_mlp_path_skips_attention_and_backpropagates():
    wrapper = _ToyQwen()
    batch = torch.randn(2, 4, 3)
    quantized = torch.randn(3, 3, requires_grad=True)

    expected_fp = wrapper.get_mlp_output(batch).detach()
    expected_q = batch + torch.nn.functional.linear(batch, quantized)
    expected_loss = torch.nn.functional.mse_loss(expected_q, expected_fp)

    loss = wrapper.calculate_mse(
        batch,
        {"model.layers.0.mlp.proj": quantized},
        self_attn=False,
        accumulation_steps=1,
    )

    assert loss == pytest.approx(expected_loss.item())
    assert quantized.grad is not None
    assert torch.isfinite(quantized.grad).all()


def test_precomputed_dense_mlp_validation_does_not_create_gradients():
    wrapper = _ToyQwen()
    batch = torch.randn(2, 4, 3)
    quantized = torch.randn(3, 3, requires_grad=True)

    wrapper.calculate_mse(
        batch,
        {"model.layers.0.mlp.proj": quantized},
        self_attn=False,
        validation=True,
    )

    assert quantized.grad is None


def test_precomputed_dense_mlp_can_defer_scalar_sync():
    wrapper = _ToyQwen()
    batch = torch.randn(2, 4, 3)
    quantized = torch.randn(3, 3, requires_grad=True)

    loss = wrapper.calculate_mse(
        batch,
        {"model.layers.0.mlp.proj": quantized},
        self_attn=False,
        defer_item=True,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not loss.requires_grad
