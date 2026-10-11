from argparse import Namespace
from pathlib import Path

from torch import nn

from experiments.nowag_d6.run_prefix_vector_gsq import (
    build_model_load_kwargs,
    move_layer_for_quantization,
    offload_layer_after_quantization,
)


class _RecordingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moves: list[str] = []

    def to(self, device, *args, **kwargs):
        self.moves.append(str(device))
        return self


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_RecordingModule()])
        self.model.rotary_emb = _RecordingModule()


def test_layerwise_cpu_offload_loads_the_base_model_on_cpu() -> None:
    args = Namespace(
        layerwise_cpu_offload=True,
        hf_device_map="",
        hf_max_memory="",
    )

    kwargs = build_model_load_kwargs(args)

    assert kwargs["device_map"] == "cpu"
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["torch_dtype"].__str__() == "torch.bfloat16"


def test_layerwise_block_moves_to_quant_device_then_back_to_cpu() -> None:
    model = _DummyModel()

    move_layer_for_quantization(model, 0, "cuda:0", enabled=True)
    offload_layer_after_quantization(model, 0, enabled=True)

    assert model.model.layers[0].moves == ["cuda:0", "cpu"]
    assert model.model.rotary_emb.moves == ["cuda:0"]

