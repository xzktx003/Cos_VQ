"""Bridge dense Hugging Face Qwen3 blocks to QTIP QuantizedLinear modules.

The QTIP optimizer itself is architecture agnostic at the Linear level, but
the upstream repository only ships a LLaMA model wrapper.  This module keeps
the architecture-specific surface deliberately small: the seven dense Qwen3
decoder projections are replaced before a QTIP state dict is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import attrgetter
from typing import Any, Mapping, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class Qwen3QtipLinearSpec:
    attribute: str
    save_name: str


QWEN3_QTIP_LINEAR_SPECS = (
    Qwen3QtipLinearSpec("self_attn.q_proj", "q"),
    Qwen3QtipLinearSpec("self_attn.k_proj", "k"),
    Qwen3QtipLinearSpec("self_attn.v_proj", "v"),
    Qwen3QtipLinearSpec("self_attn.o_proj", "o"),
    Qwen3QtipLinearSpec("mlp.up_proj", "up"),
    Qwen3QtipLinearSpec("mlp.gate_proj", "gate"),
    Qwen3QtipLinearSpec("mlp.down_proj", "down"),
)

REQUIRED_QUIP_FIELDS = (
    "td_x",
    "td_y",
    "L",
    "K",
    "V",
    "tlut_bits",
    "decode_mode",
)


def validate_qwen3_qtip_config(config: object) -> Mapping[str, Any]:
    model_type = str(getattr(config, "model_type", ""))
    if model_type != "qwen3":
        raise ValueError(
            "QTIP Qwen3 adapter requires model_type='qwen3', "
            f"got {model_type!r}"
        )
    quip_params = getattr(config, "quip_params", None)
    if not isinstance(quip_params, Mapping):
        raise ValueError("Qwen3 QTIP config is missing the quip_params mapping")
    missing = [name for name in REQUIRED_QUIP_FIELDS if name not in quip_params]
    if missing:
        raise ValueError(f"Qwen3 QTIP config is missing fields: {missing}")
    return quip_params


def _set_nested_module(parent: nn.Module, attribute: str, module: nn.Module) -> None:
    pieces = attribute.split(".")
    target = attrgetter(".".join(pieces[:-1]))(parent)
    setattr(target, pieces[-1], module)


def _skip_list(quip_params: Mapping[str, Any]) -> set[str]:
    raw = quip_params.get("skip_list")
    if raw is None:
        return set()
    if isinstance(raw, str):
        return {value for value in raw.split(",") if value}
    if isinstance(raw, Sequence):
        return {str(value) for value in raw}
    raise ValueError(f"Unsupported QTIP skip_list type: {type(raw).__name__}")


def replace_qwen3_linears(
    model: nn.Module,
    config: object,
    *,
    quantized_linear_cls: type[nn.Module],
) -> list[str]:
    """Replace every non-skipped Qwen3 decoder projection with QTIP storage."""
    quip_params = validate_qwen3_qtip_config(config)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise ValueError("Qwen3 model must expose model.layers as ModuleList")
    skipped = _skip_list(quip_params)
    replaced: list[str] = []

    for layer_index, layer in enumerate(layers):
        for spec in QWEN3_QTIP_LINEAR_SPECS:
            key = f"{layer_index}_{spec.save_name}"
            if key in skipped:
                continue
            original = attrgetter(spec.attribute)(layer)
            if not isinstance(original, nn.Linear):
                raise TypeError(
                    f"Expected nn.Linear at layer {layer_index} {spec.attribute}, "
                    f"got {type(original).__name__}"
                )
            if original.bias is not None:
                raise ValueError(
                    "QTIP Qwen3 adapter currently requires bias-free projections: "
                    f"layer {layer_index} {spec.attribute}"
                )
            quantized = quantized_linear_cls(
                original.in_features,
                original.out_features,
                int(quip_params["td_x"]),
                int(quip_params["td_y"]),
                int(quip_params["L"]),
                int(quip_params["K"]),
                int(quip_params["V"]),
                int(quip_params["tlut_bits"]),
                str(quip_params["decode_mode"]),
                bias=False,
                dtype=original.weight.dtype,
            )
            _set_nested_module(layer, spec.attribute, quantized)
            replaced.append(f"model.layers.{layer_index}.{spec.attribute}")

    return replaced


def build_qwen3_qtip_model_class(quantized_linear_cls: type[nn.Module]):
    """Create a Qwen3ForCausalLM subclass whose state shape matches QTIP."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    class Qwen3QTIPForCausalLM(Qwen3ForCausalLM):
        def __init__(self, config) -> None:
            super().__init__(config)
            replace_qwen3_linears(
                self,
                config,
                quantized_linear_cls=quantized_linear_cls,
            )

    Qwen3QTIPForCausalLM.__name__ = "Qwen3QTIPForCausalLM"
    Qwen3QTIPForCausalLM.__qualname__ = "Qwen3QTIPForCausalLM"
    return Qwen3QTIPForCausalLM
