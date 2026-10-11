"""Dependency fallbacks for QTIP's explicit weight-manifestation mode.

The fallbacks are intentionally unavailable to QTIP's compressed inference
path.  They only let ``train-fixW`` materialize dense weights when the host
does not have ABI-compatible CUDA extensions.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

import torch


def _validate_trellis_layout_inputs(
    packed: torch.Tensor,
    *,
    output_features: int,
    input_features: int,
    bits_per_weight: int,
) -> None:
    if packed.dtype != torch.int16:
        raise TypeError(f"Packed QTIP trellis must be int16, got {packed.dtype}")
    if output_features % 32 or input_features % 32:
        raise ValueError(
            "QTIP CUDA trellis layout requires input and output dimensions "
            "divisible by 32"
        )
    expected_elements = (
        (output_features // 16)
        * (input_features // 16)
        * (16 * bits_per_weight)
    )
    if packed.numel() != expected_elements:
        raise ValueError(
            f"Packed trellis has {packed.numel()} int16 values, expected "
            f"{expected_elements}"
        )


def standard_to_kernel_trellis_layout(
    packed: torch.Tensor,
    *,
    output_features: int,
    input_features: int,
    bits_per_weight: int,
) -> torch.Tensor:
    """Apply QTIP's CUDA-kernel byte permutation to a packed trellis.

    This mirrors the transformation in ``lib/algo/finetune.py`` and is kept
    here so its inverse can be tested without relying on a CUDA extension.
    """
    _validate_trellis_layout_inputs(
        packed,
        output_features=output_features,
        input_features=input_features,
        bits_per_weight=bits_per_weight,
    )
    return (
        packed.contiguous()
        .view(torch.uint8)
        .view(-1, 2)
        .flip(-1)
        .reshape(
            output_features // 32,
            2,
            input_features // 32,
            2,
            32,
            bits_per_weight,
        )
        .permute(0, 2, 4, 3, 1, 5)
        .flip(-1)
        .contiguous()
        .flatten()
        .view(torch.int16)
        .reshape(packed.shape)
    )


def kernel_to_standard_trellis_layout(
    packed: torch.Tensor,
    *,
    output_features: int,
    input_features: int,
    bits_per_weight: int,
) -> torch.Tensor:
    """Undo QTIP's CUDA-kernel byte permutation for extension-free decode."""
    _validate_trellis_layout_inputs(
        packed,
        output_features=output_features,
        input_features=input_features,
        bits_per_weight=bits_per_weight,
    )
    return (
        packed.contiguous()
        .view(torch.uint8)
        .reshape(
            output_features // 32,
            input_features // 32,
            32,
            2,
            2,
            bits_per_weight,
        )
        .flip(-1)
        .permute(0, 4, 1, 3, 2, 5)
        .contiguous()
        .reshape(-1, 2)
        .flip(-1)
        .contiguous()
        .flatten()
        .view(torch.int16)
        .reshape(packed.shape)
    )


def pure_torch_hadamard_transform(
    values: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply a Walsh-Hadamard transform along the last dimension."""
    width = values.shape[-1]
    if width <= 0 or width & (width - 1):
        raise ValueError(
            "Hadamard fallback requires a positive power-of-two last dimension, "
            f"got {width}"
        )

    transformed = values.contiguous().reshape(-1, width)
    stride = 1
    while stride < width:
        blocks = transformed.reshape(-1, width // (2 * stride), 2, stride)
        left = blocks[:, :, 0, :]
        right = blocks[:, :, 1, :]
        transformed = torch.stack((left + right, left - right), dim=2).reshape(
            -1, width
        )
        stride *= 2
    return transformed.reshape(values.shape) * scale


def _import_or_error(module_name: str) -> tuple[ModuleType | None, str | None]:
    try:
        return importlib.import_module(module_name), None
    except (ImportError, OSError) as error:
        sys.modules.pop(module_name, None)
        return None, f"{type(error).__name__}: {error}"


def install_qtip_manifest_fallbacks() -> dict[str, Any]:
    """Install process-local shims needed only to manifest QTIP weights."""
    status: dict[str, Any] = {"mode": "manifest_only"}

    fht_module, fht_error = _import_or_error("fast_hadamard_transform")
    if fht_module is None:
        fht_module = ModuleType("fast_hadamard_transform")
        fht_module.hadamard_transform = pure_torch_hadamard_transform
        sys.modules[fht_module.__name__] = fht_module
        status["fast_hadamard_transform"] = "pure_torch_fallback"
        status["fast_hadamard_transform_import_error"] = fht_error
    else:
        status["fast_hadamard_transform"] = "native_extension"

    kernels_module, kernels_error = _import_or_error("qtip_kernels")
    if kernels_module is None:
        kernels_module = ModuleType("qtip_kernels")
        sys.modules[kernels_module.__name__] = kernels_module
        status["qtip_kernels"] = "disabled_stub"
        status["qtip_kernels_import_error"] = kernels_error
    else:
        status["qtip_kernels"] = "native_extension"

    return status
