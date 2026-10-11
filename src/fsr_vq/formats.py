"""Deployment-format accounting helpers for FSR-VQ."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def tensor_payload_nbytes(tensor: torch.Tensor) -> int:
    """Return logical tensor payload bytes without container overhead."""

    return int(tensor.numel() * tensor.element_size())


def deployable_payload_nbytes(
    *,
    codebook: torch.Tensor,
    assignments: torch.Tensor,
    metadata_tensors: Iterable[torch.Tensor] = (),
) -> int:
    """Count bytes whose shape and dtype belong to the deployment format.

    This deliberately excludes Python/PyTorch container overhead and all
    train-time candidate, score, gradient, and optimizer tensors.  Packed
    sub-byte assignments must be passed as their actual packed byte tensor;
    this helper never infers compression from a nominal bit-width.
    """

    tensors = (codebook, assignments, *tuple(metadata_tensors))
    return sum(tensor_payload_nbytes(tensor) for tensor in tensors)

