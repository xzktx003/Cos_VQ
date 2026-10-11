"""Lightweight logical checkpoint reconstruction helpers."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


LINEAR_NAMES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def get_layer(model: nn.Module, layer: int) -> nn.Module:
    return model.model.layers[layer]


def get_linear_modules(model: nn.Module, layer: int) -> Dict[str, nn.Linear]:
    block = get_layer(model, layer)
    modules: Dict[str, nn.Linear] = {}
    for name in LINEAR_NAMES:
        current = block
        for part in name.split("."):
            current = getattr(current, part)
        if not isinstance(current, nn.Linear):
            raise TypeError(f"{name} is {type(current).__name__}, expected Linear")
        modules[name] = current
    return modules


def expand_group_scales(
    group_scales: torch.Tensor,
    in_features: int,
    group_size: int,
) -> torch.Tensor:
    """Expand ``[out_features, groups]`` scales to a weight-shaped tensor."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if group_scales.ndim != 2:
        raise ValueError("group_scales must be two-dimensional")
    expected_groups = (in_features + group_size - 1) // group_size
    if group_scales.shape[1] != expected_groups:
        raise ValueError(
            f"group_scales has {group_scales.shape[1]} groups, expected "
            f"{expected_groups} for in_features={in_features}, group_size={group_size}"
        )
    return group_scales.repeat_interleave(group_size, dim=1)[:, :in_features]


def residual_metadata(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Validate the optional serialized FP16 residual and report its full cost."""
    present = ("residual_A" in state, "residual_B" in state)
    if not any(present):
        return {"residual_rank": 0, "residual_parameters": 0, "residual_bits": 0}
    if not all(present):
        raise ValueError("residual_A and residual_B must appear together")
    a, b = state["residual_A"], state["residual_B"]
    for name, factor in (("residual_A", a), ("residual_B", b)):
        if not isinstance(factor, torch.Tensor) or factor.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional tensor")
        if factor.dtype != torch.float16:
            raise ValueError(f"{name} must use float16 storage")
        if not bool(torch.isfinite(factor).all()):
            raise ValueError(f"{name} must be finite")
    rows, columns = (int(value) for value in state["shape"])
    rank = a.shape[0]
    if rank <= 0 or rank > min(rows, columns):
        raise ValueError("residual rank must be positive and no larger than the weight dimensions")
    if tuple(a.shape) != (rank, columns) or tuple(b.shape) != (rows, rank):
        raise ValueError("residual factor shapes must be (rank, columns) and (rows, rank)")
    parameters = a.numel() + b.numel()
    return {"residual_rank": rank, "residual_parameters": parameters,
            "residual_bits": parameters * 16}


def logical_state_bits(state: Dict[str, torch.Tensor]) -> int:
    """Logical representation bits, including all stored compensation factors."""
    bits = state["assignments"].numel() * int(state["assignment_bits"])
    bits += sum(state[key].numel() * 16 for key in ("codebook", "row_norm", "col_norm"))
    if "group_scale" in state:
        bits += state["group_scale"].numel() * 16
    return bits + residual_metadata(state)["residual_bits"]


def reconstruct_logical_state(
    state: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    residual = residual_metadata(state)
    rows, columns = (int(value) for value in state["shape"])
    vector_dim = int(state["vector_dim"])
    assignments = state["assignments"].to(device=device, dtype=torch.long)
    codebook = state["codebook"].to(device=device, dtype=torch.float32)
    padded_columns = math.ceil(columns / vector_dim) * vector_dim
    weight = codebook[assignments].reshape(rows, padded_columns)[:, :columns]
    row_norm = state["row_norm"].to(device=device, dtype=torch.float32)
    col_norm = state["col_norm"].to(device=device, dtype=torch.float32)
    weight = weight * row_norm[:, None] * col_norm[None, :]
    if "group_scale" in state:
        if "group_size" not in state:
            raise ValueError("group_scale checkpoint is missing group_size")
        group_scale = state["group_scale"].to(
            device=device,
            dtype=torch.float32,
        )
        expanded = expand_group_scales(
            group_scale,
            columns,
            int(state["group_size"]),
        )
        weight = weight * expanded
    if residual["residual_rank"]:
        a = state["residual_A"].to(device=device, dtype=torch.float32)
        b = state["residual_B"].to(device=device, dtype=torch.float32)
        weight = weight + b @ a
    return weight
