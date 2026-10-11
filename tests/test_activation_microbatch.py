"""Regression tests for activation propagation memory bounds.

The full calibration batch can be much larger than a decoder layer can safely
forward at once.  These tests construct a wrapper without loading a model and
verify that every activation-propagation path honors the dedicated device
microbatch instead of ``data.batch_size``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.base import BaseModelWrapper  # noqa: E402


class _RecordingLayer(torch.nn.Module):
    def __init__(self, seen_batch_sizes: list[int]):
        super().__init__()
        self.seen_batch_sizes = seen_batch_sizes

    def forward(self, x, **_kwargs):
        self.seen_batch_sizes.append(x.shape[0])
        return (x + 1,)


class _OomAboveTwoLayer(torch.nn.Module):
    def __init__(self, seen_batch_sizes: list[int]):
        super().__init__()
        self.seen_batch_sizes = seen_batch_sizes

    def forward(self, x, **_kwargs):
        self.seen_batch_sizes.append(x.shape[0])
        if x.shape[0] > 2:
            raise RuntimeError("CUDA out of memory")
        return (x + 1,)


class _TestWrapper(BaseModelWrapper):
    def __init__(self):
        # Deliberately skip BaseModelWrapper.__init__: the regression only
        # exercises batching and must not download or allocate a real model.
        self.batch_size = 64
        self.activation_microbatch_size = 2
        self.device = "cpu"
        self.current_layer_idx = 0
        self.kwargs = {}
        self.layer_batch_sizes: list[int] = []
        self.mlp_input_batch_sizes: list[int] = []
        self.mlp_output_batch_sizes: list[int] = []
        self.layer = _RecordingLayer(self.layer_batch_sizes)

    def get_mlp_input(self, layer_input):
        self.mlp_input_batch_sizes.append(layer_input.shape[0])
        return layer_input + 2

    def get_mlp_output(self, mlp_input_batch):
        self.mlp_output_batch_sizes.append(mlp_input_batch.shape[0])
        return mlp_input_batch + 3

    def get_layer_module(self, _idx):
        return self.layer

    def move_embed_to(self, _device):
        pass

    def move_output_heads_to(self, _device):
        pass

    def _layer_prefixes(self, _layer_name):
        return {}

    def ppl_evaluation(self, read_from_disk=-1):
        return float(read_from_disk)


def _inputs():
    return {"input": torch.zeros(5, 3, 4)}


def test_layer_activation_propagation_uses_dedicated_microbatch():
    wrapper = _TestWrapper()
    data = _inputs()

    wrapper.get_layer_activations(data)

    assert wrapper.layer_batch_sizes == [2, 2, 1]
    assert torch.equal(data["input"], torch.ones_like(data["input"]))


def test_mlp_activation_propagation_uses_dedicated_microbatch():
    wrapper = _TestWrapper()
    mlp_inputs = _inputs()
    mlp_outputs = _inputs()

    wrapper.get_mlp_input_all(mlp_inputs)
    wrapper.get_mlp_output_all(mlp_outputs)

    assert wrapper.mlp_input_batch_sizes == [2, 2, 1]
    assert wrapper.mlp_output_batch_sizes == [2, 2, 1]
    assert torch.equal(mlp_inputs["input"], torch.full_like(mlp_inputs["input"], 2))
    assert torch.equal(mlp_outputs["input"], torch.full_like(mlp_outputs["input"], 3))


def test_activation_oom_persistently_halves_microbatch_and_retries_slice():
    wrapper = _TestWrapper()
    wrapper.activation_microbatch_size = 8
    wrapper.activation_oom_microbatch_backoff = True
    wrapper.min_activation_microbatch_size = 2
    wrapper.layer_batch_sizes = []
    wrapper.layer = _OomAboveTwoLayer(wrapper.layer_batch_sizes)
    data = _inputs()

    wrapper.get_layer_activations(data)

    assert wrapper.layer_batch_sizes == [5, 4, 2, 2, 1]
    assert wrapper.activation_microbatch_size == 2
    assert torch.equal(data["input"], torch.ones_like(data["input"]))


def main():
    test_layer_activation_propagation_uses_dedicated_microbatch()
    test_mlp_activation_propagation_uses_dedicated_microbatch()
    test_activation_oom_persistently_halves_microbatch_and_retries_slice()
    print("OK: activation propagation honors the dedicated microbatch")


if __name__ == "__main__":
    main()
