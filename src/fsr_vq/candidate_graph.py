"""Function-directed candidate targets for hard vector transitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LinearFunctionQuadratic:
    """Local quadratic metric and its unconstrained residual-optimal target."""

    metric: torch.Tensor
    linear_term: torch.Tensor
    target: torch.Tensor


def linear_function_quadratic(
    *,
    current_codeword: torch.Tensor,
    gram: torch.Tensor,
    gradient: torch.Tensor,
    scale: torch.Tensor,
    damping: float = 0.0,
) -> LinearFunctionQuadratic:
    """Convert an exact Linear transition objective into a nearest-target form.

    If an actual weight-vector update is ``diag(scale) @ (c - c0)``, then

    ``delta_D(c) = 2 (c-c0)^T q + (c-c0)^T A (c-c0)``

    where ``q = diag(scale) @ gradient`` and
    ``A = diag(scale) @ gram @ diag(scale)``.  For positive-definite ``A``,
    this has the same codeword ranking as the Mahalanobis distance from
    ``target = c0 - A^-1 q``.  Optional damping stabilizes singular metrics;
    with damping, the target corresponds to the regularized quadratic.
    """

    if current_codeword.ndim != 1 or gradient.ndim != 1 or scale.ndim != 1:
        raise ValueError("current_codeword, gradient, and scale must be vectors")
    dimension = current_codeword.numel()
    if gradient.numel() != dimension or scale.numel() != dimension:
        raise ValueError("all vectors must have the same dimension")
    if gram.shape != (dimension, dimension):
        raise ValueError(
            f"gram must have shape {(dimension, dimension)}, got {tuple(gram.shape)}"
        )
    tensors = (current_codeword, gram, gradient, scale)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all quadratic tensors must share a device")
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise ValueError("all quadratic tensors must share a dtype")
    if damping < 0:
        raise ValueError("damping must be non-negative")

    metric = scale[:, None] * gram * scale[None, :]
    metric = 0.5 * (metric + metric.T)
    if damping:
        metric = metric + float(damping) * torch.eye(
            dimension, device=metric.device, dtype=metric.dtype
        )
    linear_term = scale * gradient
    target = current_codeword - torch.linalg.solve(metric, linear_term)
    return LinearFunctionQuadratic(
        metric=metric,
        linear_term=linear_term,
        target=target,
    )


def residual_target_distances(
    *,
    codebook: torch.Tensor,
    target: torch.Tensor,
    metric: torch.Tensor,
) -> torch.Tensor:
    """Return squared Mahalanobis distances from codewords to a target."""

    if codebook.ndim != 2 or target.ndim != 1 or metric.ndim != 2:
        raise ValueError("codebook, target, and metric must have ranks 2, 1, and 2")
    dimension = codebook.shape[1]
    if target.numel() != dimension or metric.shape != (dimension, dimension):
        raise ValueError("target and metric dimensions must match the codebook")
    difference = codebook - target.unsqueeze(0)
    return torch.einsum("kd,de,ke->k", difference, metric, difference)


def diagonal_residual_target_distances(
    *,
    codebook: torch.Tensor,
    target: torch.Tensor,
    metric: torch.Tensor,
) -> torch.Tensor:
    """Cheap function-directed proxy using only the quadratic diagonal."""

    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError("metric must be square")
    diagonal = torch.diag(metric).clamp_min(0)
    difference = codebook - target.unsqueeze(0)
    return (difference.square() * diagonal.unsqueeze(0)).sum(dim=1)

