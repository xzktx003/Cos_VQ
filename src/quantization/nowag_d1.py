"""NoWag-normalized d=1 scalar-codebook quantization for GSQ research.

This module is an independent implementation of the equations documented in
``docs/20260716_011801_nowag_d1_trainable_group_scale_gsq_design.md``.  It does
not reuse source code from the GPL-licensed NoWag repository.

The serialized research state deliberately contains both the compact logical
state (2-bit assignments, group scales, norms, and codebook) and the
dequantized hard weight used by the existing GSQ evaluator.  The latter is a
research/evaluation convenience and is excluded from effective-bpp accounting.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


NOWAG_FORMAT = "nowag_d1"


def _group_index(columns: int, groupsize: int, device: torch.device) -> torch.Tensor:
    if groupsize <= 0:
        groupsize = columns
    return torch.arange(columns, device=device, dtype=torch.long) // groupsize


def expand_group_values(values: torch.Tensor, columns: int, groupsize: int) -> torch.Tensor:
    """Expand ``[rows, groups]`` values to ``[rows, columns]``."""
    idx = _group_index(columns, groupsize, values.device)
    return values[:, idx]


def normalize_linear_weight(
    weight: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply fixed NoWag column-L2 then row-L2 normalization."""
    w = weight.float()
    col_norm = torch.linalg.vector_norm(w, ord=2, dim=0).clamp_min(eps)
    col_normalized = w / col_norm.unsqueeze(0)
    row_norm = torch.linalg.vector_norm(col_normalized, ord=2, dim=1).clamp_min(eps)
    normalized = col_normalized / row_norm.unsqueeze(1)
    return normalized, row_norm, col_norm


def denormalize_linear_weight(
    normalized: torch.Tensor,
    row_norm: torch.Tensor,
    col_norm: torch.Tensor,
) -> torch.Tensor:
    return normalized * row_norm.unsqueeze(1) * col_norm.unsqueeze(0)


def group_absmax_scale(
    normalized: torch.Tensor,
    groupsize: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return positive per-row, per-group scales using absmax.

    Ordinary signed ``max`` is intentionally invalid here: it can miss a
    larger-magnitude negative weight and can even return a negative scale for
    all-negative groups.
    """
    rows, columns = normalized.shape
    if groupsize <= 0:
        groupsize = columns
    n_groups = (columns + groupsize - 1) // groupsize
    padded_columns = n_groups * groupsize
    if padded_columns != columns:
        normalized = F.pad(normalized, (0, padded_columns - columns), value=0.0)
    groups = normalized.reshape(rows, n_groups, groupsize)
    return groups.abs().amax(dim=-1).clamp_min(eps)


def _strictly_ordered(codebook: torch.Tensor, eps: float) -> torch.Tensor:
    codebook = torch.sort(codebook.float()).values
    data_range = (codebook[-1] - codebook[0]).abs().clamp_min(1.0)
    min_gap = max(float(eps), float(data_range.item()) * 1e-7)
    repaired = [codebook[0]]
    for value in codebook[1:]:
        repaired.append(torch.maximum(value, repaired[-1] + min_gap))
    return torch.stack(repaired)


@torch.no_grad()
def weighted_kmeans_pp_1d(
    values: torch.Tensor,
    weights: torch.Tensor,
    clusters: int = 4,
    seed: int = 0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Weighted k-means++ initialization for scalar observations."""
    x = values.reshape(-1).float()
    w = weights.reshape(-1).float().clamp_min(0)
    finite = torch.isfinite(x) & torch.isfinite(w) & (w > 0)
    if not finite.any():
        raise ValueError("No finite positive-weight observations for NoWag-d1 codebook")
    x = x[finite]
    w = w[finite]

    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed))
    first = torch.multinomial(w / w.sum().clamp_min(eps), 1, generator=generator)
    centers = [x[first[0]]]
    nearest_sq = (x - centers[0]).square()

    for _ in range(1, clusters):
        probabilities = w * nearest_sq
        total = probabilities.sum()
        if not torch.isfinite(total) or total <= eps:
            # Degenerate input: use a deterministic weighted quantile fallback.
            order = torch.argsort(x)
            cumulative = torch.cumsum(w[order], dim=0)
            target = cumulative[-1] * (len(centers) / max(clusters - 1, 1))
            idx = torch.searchsorted(cumulative, target.clamp_max(cumulative[-1]))
            candidate = x[order[idx.clamp_max(order.numel() - 1)]]
        else:
            idx = torch.multinomial(probabilities / total, 1, generator=generator)
            candidate = x[idx[0]]
        centers.append(candidate)
        nearest_sq = torch.minimum(nearest_sq, (x - candidate).square())

    return _strictly_ordered(torch.stack(centers), eps)


@torch.no_grad()
def weighted_lloyd_1d(
    values: torch.Tensor,
    weights: torch.Tensor,
    clusters: int = 4,
    seed: int = 0,
    max_iters: int = 100,
    eps: float = 1e-12,
    chunk_size: int = 2_000_000,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Fit a shared ordered scalar codebook with Hessian-weighted Lloyd steps."""
    original_shape = values.shape
    x = values.reshape(-1).float()
    w = weights.reshape(-1).float().clamp_min(0)
    if x.numel() != w.numel():
        raise ValueError(f"values/weights size mismatch: {x.numel()} != {w.numel()}")
    if not torch.isfinite(x).all() or not torch.isfinite(w).all():
        raise ValueError("Non-finite values or Hessian weights in NoWag-d1 initialization")

    codebook = weighted_kmeans_pp_1d(x, w, clusters=clusters, seed=seed, eps=eps)
    previous_assignment: Optional[torch.Tensor] = None
    iterations = 0

    for iteration in range(max_iters):
        assignments = torch.empty(x.numel(), device=x.device, dtype=torch.long)
        weighted_sum = torch.zeros(clusters, device=x.device, dtype=torch.float64)
        weight_sum = torch.zeros(clusters, device=x.device, dtype=torch.float64)
        residual_best_value = None
        residual_best_score = torch.tensor(-1.0, device=x.device)

        for start in range(0, x.numel(), chunk_size):
            end = min(start + chunk_size, x.numel())
            xc = x[start:end]
            wc = w[start:end]
            distances = (xc.unsqueeze(1) - codebook.unsqueeze(0)).square()
            ac = distances.argmin(dim=1)
            assignments[start:end] = ac
            weighted_sum.scatter_add_(0, ac, (wc * xc).double())
            weight_sum.scatter_add_(0, ac, wc.double())
            residual = wc * distances.gather(1, ac.unsqueeze(1)).squeeze(1)
            local_score, local_idx = residual.max(dim=0)
            if local_score > residual_best_score:
                residual_best_score = local_score
                residual_best_value = xc[local_idx]

        new_codebook = codebook.clone()
        for cluster in range(clusters):
            if weight_sum[cluster] > eps:
                new_codebook[cluster] = (weighted_sum[cluster] / weight_sum[cluster]).float()
            elif residual_best_value is not None:
                # The design explicitly forbids silently replacing empty clusters by zero.
                new_codebook[cluster] = residual_best_value

        new_codebook = _strictly_ordered(new_codebook, eps)
        iterations = iteration + 1
        converged = previous_assignment is not None and torch.equal(assignments, previous_assignment)
        if converged or torch.allclose(new_codebook, codebook, atol=1e-7, rtol=1e-6):
            codebook = new_codebook
            break
        previous_assignment = assignments
        codebook = new_codebook

    # Recompute IDs after sorting so assignment IDs always match ordered codepoints.
    assignments = torch.empty(x.numel(), device=x.device, dtype=torch.long)
    occupancy = torch.zeros(clusters, device=x.device, dtype=torch.long)
    objective = torch.zeros((), device=x.device, dtype=torch.float64)
    for start in range(0, x.numel(), chunk_size):
        end = min(start + chunk_size, x.numel())
        xc = x[start:end]
        wc = w[start:end]
        distances = (xc.unsqueeze(1) - codebook.unsqueeze(0)).square()
        ac = distances.argmin(dim=1)
        assignments[start:end] = ac
        occupancy.scatter_add_(0, ac, torch.ones_like(ac, dtype=torch.long))
        objective += (wc * distances.gather(1, ac.unsqueeze(1)).squeeze(1)).double().sum()

    diagnostics = {
        "lloyd_iterations": float(iterations),
        "weighted_objective": float(objective.item()),
    }
    for k, count in enumerate(occupancy.tolist()):
        diagnostics[f"occupancy_{k}"] = float(count)
    return codebook, assignments.reshape(original_shape), diagnostics


@torch.no_grad()
def weighted_ls_group_scale(
    normalized: torch.Tensor,
    assignments: torch.Tensor,
    codebook: torch.Tensor,
    hessian_diag: torch.Tensor,
    row_norm: torch.Tensor,
    col_norm: torch.Tensor,
    groupsize: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Positive weighted-LS scale for fixed assignments/codebook."""
    rows, columns = normalized.shape
    idx = _group_index(columns, groupsize, normalized.device)
    n_groups = int(idx[-1].item()) + 1
    selected = codebook[assignments]
    lam = hessian_diag.unsqueeze(0) * (
        row_norm.unsqueeze(1) * col_norm.unsqueeze(0)
    ).square()
    numerator = torch.zeros(rows, n_groups, device=normalized.device, dtype=torch.float32)
    denominator = torch.zeros_like(numerator)
    scatter_idx = idx.unsqueeze(0).expand(rows, -1)
    numerator.scatter_add_(1, scatter_idx, lam * normalized * selected)
    denominator.scatter_add_(1, scatter_idx, lam * selected.square())
    return (numerator / denominator.clamp_min(eps)).clamp_min(eps)


def hessian_objective(weight: torch.Tensor, quantized: torch.Tensor, hessian: torch.Tensor) -> float:
    error = (weight.float() - quantized.float())
    value = ((error @ hessian.float()) * error).sum()
    return float(value.item())


@torch.no_grad()
def codebook_aware_gptq(
    normalized: torch.Tensor,
    hessian: torch.Tensor,
    col_norm: torch.Tensor,
    group_scale: torch.Tensor,
    codebook: torch.Tensor,
    groupsize: int,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sequential GPTQ error feedback using a fixed nonuniform codebook.

    The Hessian is transformed into the normalized-U coordinate system using
    ``diag(col_norm) H diag(col_norm)``.  Per-row ``row_norm`` factors cancel
    from each row's argmin/error propagation and therefore do not enter this
    transformed Hessian.
    """
    rows, columns = normalized.shape
    transformed_h = hessian.float() * col_norm.unsqueeze(0) * col_norm.unsqueeze(1)
    dead = torch.diag(transformed_h) == 0
    transformed_h = transformed_h.clone()
    transformed_h[dead, dead] = 1.0
    damp = percdamp * torch.diag(transformed_h).mean()
    diag = torch.arange(columns, device=normalized.device)
    transformed_h[diag, diag] += damp

    chol = torch.linalg.cholesky(transformed_h)
    hinv = torch.cholesky_inverse(chol)
    hinv = torch.linalg.cholesky(hinv, upper=True)

    work = normalized.float().clone()
    work[:, dead] = 0
    quantized = torch.zeros_like(work)

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1
        w1 = work[:, i1:i2].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]

        for i in range(count):
            absolute_column = i1 + i
            scale = group_scale[:, absolute_column // groupsize].clamp_min(1e-12)
            normalized_value = w1[:, i] / scale
            assignment = (
                normalized_value.unsqueeze(1) - codebook.unsqueeze(0)
            ).square().argmin(dim=1)
            q = codebook[assignment] * scale
            q1[:, i] = q
            error = (w1[:, i] - q) / hinv1[i, i]
            w1[:, i:] -= error.unsqueeze(1) @ hinv1[i, i:].unsqueeze(0)
            err1[:, i] = error

        quantized[:, i1:i2] = q1
        work[:, i2:] -= err1 @ hinv[i1:i2, i2:]

    expanded_scale = expand_group_values(group_scale, columns, groupsize)
    assignments = (
        (quantized / expanded_scale.clamp_min(1e-12)).unsqueeze(-1)
        - codebook.view(1, 1, -1)
    ).square().argmin(dim=-1)
    return quantized, assignments


@torch.no_grad()
def initialize_nowag_d1(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    groupsize: int,
    hessian_weight_mode: str = "original_corrected",
    scale_init: str = "absmax",
    codebook_aware_gptq_enabled: bool = False,
    seed: int = 0,
    max_iters: int = 100,
    blocksize: int = 128,
    percdamp: float = 0.01,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Create the complete hard NoWag-d1 initialization state for one Linear."""
    original = weight.float()
    normalized, row_norm, col_norm = normalize_linear_weight(original, eps=eps)
    scale0 = group_absmax_scale(normalized, groupsize, eps=eps)
    expanded_scale0 = expand_group_values(scale0, normalized.shape[1], groupsize)
    z = normalized / expanded_scale0

    hessian_diag = torch.diag(hessian.float()).clamp_min(eps)
    if hessian_weight_mode == "nowag_faithful":
        omega = hessian_diag.unsqueeze(0).expand_as(z)
    elif hessian_weight_mode == "original_corrected":
        omega = hessian_diag.unsqueeze(0) * (
            row_norm.unsqueeze(1)
            * col_norm.unsqueeze(0)
            * expanded_scale0
        ).square()
    else:
        raise ValueError(
            f"Unsupported hessian_weight_mode={hessian_weight_mode!r}; "
            "expected 'original_corrected' or 'nowag_faithful'"
        )

    codebook, assignments, diagnostics = weighted_lloyd_1d(
        z,
        omega,
        clusters=4,
        seed=seed,
        max_iters=max_iters,
        eps=eps,
    )

    if scale_init == "absmax":
        group_scale = scale0
    elif scale_init == "weighted_ls":
        group_scale = weighted_ls_group_scale(
            normalized,
            assignments,
            codebook,
            hessian_diag,
            row_norm,
            col_norm,
            groupsize,
            eps=eps,
        )
    else:
        raise ValueError(
            f"Unsupported scale_init={scale_init!r}; expected 'absmax' or 'weighted_ls'"
        )

    normalized_quantized = (
        codebook[assignments]
        * expand_group_values(group_scale, normalized.shape[1], groupsize)
    )

    pre_gptq_weight = denormalize_linear_weight(normalized_quantized, row_norm, col_norm)
    pre_gptq_objective = hessian_objective(original, pre_gptq_weight, hessian)

    if codebook_aware_gptq_enabled:
        refined, refined_assignments = codebook_aware_gptq(
            normalized,
            hessian,
            col_norm,
            group_scale,
            codebook,
            groupsize,
            blocksize=blocksize,
            percdamp=percdamp,
        )
        refined_weight = denormalize_linear_weight(refined, row_norm, col_norm)
        refined_objective = hessian_objective(original, refined_weight, hessian)
        # Refinement is accepted only when its full-Hessian objective is non-worse.
        if refined_objective <= pre_gptq_objective * (1.0 + 1e-7):
            normalized_quantized = refined
            assignments = refined_assignments
            diagnostics["gptq_refinement_accepted"] = 1.0
            diagnostics["gptq_refined_objective"] = refined_objective
        else:
            diagnostics["gptq_refinement_accepted"] = 0.0
            diagnostics["gptq_refined_objective"] = refined_objective

    hard_weight = denormalize_linear_weight(normalized_quantized, row_norm, col_norm)
    diagnostics["full_hessian_objective"] = hessian_objective(original, hard_weight, hessian)
    diagnostics["weight_relative_l2"] = float(
        (original - hard_weight).norm() / original.norm().clamp_min(eps)
    )
    diagnostics["roundtrip_max_abs_error"] = float(
        (original - denormalize_linear_weight(normalized, row_norm, col_norm)).abs().max()
    )

    state: Dict[str, torch.Tensor] = {
        "format": NOWAG_FORMAT,
        "weight": hard_weight.to(weight.dtype),
        "assignments": assignments.to(torch.uint8),
        "scale0": scale0.float(),
        "scale": group_scale.float(),
        "log_scale": torch.log(group_scale / scale0).float(),
        "codebook": codebook.float(),
        "row_norm": row_norm.float(),
        "col_norm": col_norm.float(),
        "groupsize": torch.tensor(int(groupsize), device=weight.device, dtype=torch.int32),
        "diagnostics": diagnostics,
    }
    return state


def is_nowag_state(value) -> bool:
    return isinstance(value, dict) and value.get("format") == NOWAG_FORMAT


class NoWagD1GumbelQuantizer(nn.Module):
    """Four-codepoint GSQ with fixed normalization and configurable scale training."""

    def __init__(self, state: Dict[str, torch.Tensor], config, device, dtype, logits_dtype):
        super().__init__()
        if not is_nowag_state(state):
            raise TypeError("NoWagD1GumbelQuantizer requires a NoWag-d1 state dictionary")

        self.device = torch.device(device)
        self.dtype = dtype
        self.logits_dtype = logits_dtype
        self.groupsize = int(state["groupsize"].item())
        self.weight_shape = tuple(state["weight"].shape)
        qcfg = config.quantization

        self.scale_training = qcfg.nowag_scale_training
        self.scale_ratio_min = float(qcfg.nowag_scale_ratio_min)
        self.scale_ratio_max = float(qcfg.nowag_scale_ratio_max)
        self.scale_warmup_epochs = int(qcfg.nowag_scale_warmup_epochs)
        self.norm_trainable = bool(qcfg.nowag_norm_trainable)
        self.norm_ratio_min = float(qcfg.nowag_norm_ratio_min)
        self.norm_ratio_max = float(qcfg.nowag_norm_ratio_max)
        self.codebook_trainable = bool(qcfg.nowag_codebook_trainable)

        self.register_buffer("scale0", state["scale0"].to(self.device, torch.float32))
        initial_log_scale = state["log_scale"].to(self.device, torch.float32)
        if self.scale_training == "fixed":
            self.register_buffer("log_scale", initial_log_scale)
        elif self.scale_training in ("unconstrained", "trust_region"):
            self.log_scale = nn.Parameter(initial_log_scale.clone())
        else:
            raise ValueError(
                f"Unsupported nowag_scale_training={self.scale_training!r}; "
                "expected fixed, unconstrained, or trust_region"
            )

        self.register_buffer("base_row_norm", state["row_norm"].to(self.device, torch.float32))
        self.register_buffer("base_col_norm", state["col_norm"].to(self.device, torch.float32))
        if self.norm_trainable:
            self.row_log_ratio = nn.Parameter(torch.zeros_like(self.base_row_norm))
            self.col_log_ratio = nn.Parameter(torch.zeros_like(self.base_col_norm))

        initial_codebook = state["codebook"].to(self.device, torch.float32)
        self.register_buffer("initial_codebook", initial_codebook)
        self.register_buffer("initial_codebook_norm", initial_codebook.norm().clamp_min(1e-8))
        if self.codebook_trainable:
            self.codebook_base = nn.Parameter(initial_codebook[:1].clone())
            gaps = (initial_codebook[1:] - initial_codebook[:-1]).clamp_min(1e-6)
            inverse_softplus = torch.log(torch.expm1(gaps).clamp_min(1e-12))
            self.codebook_gap_raw = nn.Parameter(inverse_softplus)

        rows, columns = self.weight_shape
        assignments = state["assignments"].to(self.device, torch.long)
        expanded_scale = expand_group_values(state["scale"].to(self.device), columns, self.groupsize)
        normalized = (
            state["weight"].to(self.device, torch.float32)
            / self.base_row_norm.unsqueeze(1)
            / self.base_col_norm.unsqueeze(0)
            / expanded_scale.clamp_min(1e-12)
        )
        distances = -0.5 * (
            normalized.unsqueeze(0) - initial_codebook.view(4, 1, 1)
        ).square()
        distances = distances - distances.mean(dim=0, keepdim=True)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(qcfg.nowag_seed))
        noise = torch.randn(
            (4, rows, columns),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        logits = float(qcfg.std) * (noise + distances * float(qcfg.strength))
        # Preserve the exact initialization assignment despite exploration noise.
        selected = logits.gather(0, assignments.unsqueeze(0)).squeeze(0)
        maximum = logits.amax(dim=0)
        selected = torch.maximum(selected, maximum + max(float(qcfg.std), 1e-4))
        logits.scatter_(0, assignments.unsqueeze(0), selected.unsqueeze(0))
        self.quant_logits = nn.Parameter(logits.to(self.logits_dtype))

    def current_codebook(self) -> torch.Tensor:
        if not self.codebook_trainable:
            return self.initial_codebook
        gaps = F.softplus(self.codebook_gap_raw) + 1e-7
        ordered = torch.cat(
            [self.codebook_base, self.codebook_base + torch.cumsum(gaps, dim=0)]
        )
        # Fixed L2 norm removes the codebook/scale multiplicative ambiguity.
        return ordered * (self.initial_codebook_norm / ordered.norm().clamp_min(1e-8))

    def current_scale(self) -> torch.Tensor:
        log_scale = self.log_scale
        if self.scale_training == "trust_region":
            log_scale = log_scale.clamp(
                math.log(self.scale_ratio_min), math.log(self.scale_ratio_max)
            )
        return self.scale0 * torch.exp(log_scale)

    def current_norms(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.norm_trainable:
            return self.base_row_norm, self.base_col_norm
        row_ratio = self.row_log_ratio.clamp(
            math.log(self.norm_ratio_min), math.log(self.norm_ratio_max)
        )
        col_ratio = self.col_log_ratio.clamp(
            math.log(self.norm_ratio_min), math.log(self.norm_ratio_max)
        )
        return (
            self.base_row_norm * torch.exp(row_ratio),
            self.base_col_norm * torch.exp(col_ratio),
        )

    def set_epoch(self, epoch: int) -> None:
        if isinstance(self.log_scale, nn.Parameter):
            self.log_scale.requires_grad_(epoch >= self.scale_warmup_epochs)

    @torch.no_grad()
    def project_parameters(self) -> None:
        if isinstance(self.log_scale, nn.Parameter) and self.scale_training == "trust_region":
            self.log_scale.clamp_(
                math.log(self.scale_ratio_min), math.log(self.scale_ratio_max)
            )
        if self.norm_trainable:
            self.row_log_ratio.clamp_(
                math.log(self.norm_ratio_min), math.log(self.norm_ratio_max)
            )
            self.col_log_ratio.clamp_(
                math.log(self.norm_ratio_min), math.log(self.norm_ratio_max)
            )

    def forward(self, temperature: float, scale: float = 1.0) -> torch.Tensor:
        eps = 1e-8
        uniform = torch.rand_like(self.quant_logits)
        gumbel = -torch.log(-torch.log(uniform + eps) + eps)
        probabilities = F.softmax(
            (self.quant_logits.float() * float(scale) + gumbel) / float(temperature),
            dim=0,
        )
        codebook = self.current_codebook()
        soft_codepoint = (probabilities * codebook.view(4, 1, 1)).sum(dim=0)
        group_scale = expand_group_values(
            self.current_scale(), self.weight_shape[1], self.groupsize
        )
        row_norm, col_norm = self.current_norms()
        weight = (
            soft_codepoint
            * group_scale
            * row_norm.unsqueeze(1)
            * col_norm.unsqueeze(0)
        )
        return weight.to(self.dtype)

    @torch.no_grad()
    def get_hard_weights(self) -> Dict[str, torch.Tensor]:
        assignments = self.quant_logits.argmax(dim=0)
        codebook = self.current_codebook()
        scale = self.current_scale()
        row_norm, col_norm = self.current_norms()
        normalized = codebook[assignments] * expand_group_values(
            scale, self.weight_shape[1], self.groupsize
        )
        weight = denormalize_linear_weight(normalized, row_norm, col_norm).to(self.dtype)
        return {
            "format": NOWAG_FORMAT,
            "weight": weight,
            "assignments": assignments.to(torch.uint8),
            "scale0": self.scale0.detach().clone(),
            "scale": scale.detach().clone(),
            "log_scale": torch.log(scale / self.scale0).detach().clone(),
            "codebook": codebook.detach().clone(),
            "row_norm": row_norm.detach().clone(),
            "col_norm": col_norm.detach().clone(),
            "groupsize": torch.tensor(
                self.groupsize, device=weight.device, dtype=torch.int32
            ),
        }

    @torch.no_grad()
    def audit(self) -> Dict[str, float]:
        scale_ratio = self.current_scale() / self.scale0
        assignments = self.quant_logits.argmax(dim=0)
        codebook = self.current_codebook()
        result = {
            "scale_ratio_min": float(scale_ratio.min()),
            "scale_ratio_max": float(scale_ratio.max()),
            "scale_ratio_mean": float(scale_ratio.mean()),
            "scale_ratio_saturation_rate": 0.0,
            "codebook_min_gap": float((codebook[1:] - codebook[:-1]).min()),
        }
        if self.scale_training == "trust_region":
            saturated = (scale_ratio <= self.scale_ratio_min + 1e-6) | (
                scale_ratio >= self.scale_ratio_max - 1e-6
            )
            result["scale_ratio_saturation_rate"] = float(saturated.float().mean())
        for k in range(4):
            result[f"occupancy_{k}"] = float((assignments == k).float().mean())
        return result


def logical_effective_bits(state: Dict[str, torch.Tensor]) -> Tuple[int, float]:
    """Return logical serialized bits and effective bpp (excluding FP hard cache)."""
    if not is_nowag_state(state):
        raise TypeError("logical_effective_bits expects a NoWag-d1 state")
    rows, columns = state["weight"].shape
    assignment_bits = rows * columns * 2
    scale_bits = state["scale"].numel() * 16
    norm_bits = (state["row_norm"].numel() + state["col_norm"].numel()) * 16
    codebook_bits = state["codebook"].numel() * 16
    total = assignment_bits + scale_bits + norm_bits + codebook_bits
    return total, total / (rows * columns)
