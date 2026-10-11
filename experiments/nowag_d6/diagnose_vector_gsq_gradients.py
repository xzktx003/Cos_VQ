"""Diagnose GSQ probability gradients after a full NoWag d=6 initialization.

The NoWag initialization is executed through the separately checked-out NoWag
repository.  This file does not copy its GPL implementation into GSQ.  It adds
only an experiment-local differentiable relaxation over local codeword
candidates, so the established NoWag-d1 path remains unchanged.

The experiment has two parts:

1. A small full-codebook probe over all 4096 NoWag codewords.  This measures
   whether gradients are diffuse or saturated without materializing full-model
   4096-way logits.
2. A runnable local-candidate Vector-GSQ experiment.  Each d=6 subvector keeps
   its exact NoWag assignment plus its nearest codeword neighbours and trains
   only the rounding logits.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lion_pytorch import Lion
from transformers import AutoModelForCausalLM


class _StopForward(RuntimeError):
    pass


@dataclass
class Schedule:
    name: str
    temperature_start: float
    temperature_end: float
    kappa_start: float
    kappa_end: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--target", default="model.layers.0.self_attn.k_proj")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--nowag-repo",
        default=str(Path(__file__).resolve().parents[2] / "third_party" / "NoWag"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calib-sequences", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--train-tokens", type=int, default=128)
    parser.add_argument("--vector-dim", type=int, default=6)
    parser.add_argument("--bits-per-weight", type=float, default=2.0)
    parser.add_argument("--lloyd-iters", type=int, default=100)
    parser.add_argument("--local-candidates", type=int, default=8)
    parser.add_argument("--candidate-chunk", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--std", type=float, default=0.01)
    parser.add_argument("--strength", type=float, default=6.0)
    parser.add_argument("--full-probe-vectors", type=int, default=256)
    parser.add_argument("--half-kmeans", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_submodule(model: nn.Module, name: str) -> nn.Module:
    current = model
    for part in name.split("."):
        current = getattr(current, part)
    return current


@torch.inference_mode()
def capture_linear_inputs(
    model: nn.Module,
    module: nn.Linear,
    token_ids: torch.Tensor,
    calib_sequences: int,
    sequence_length: int,
    train_tokens: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    hessian_diag = torch.zeros(module.in_features, device=device, dtype=torch.float64)
    selected: List[torch.Tensor] = []
    total_rows = 0

    def hook(_module, inputs):
        nonlocal total_rows
        x = inputs[0].detach().reshape(-1, module.in_features).float()
        hessian_diag.add_(x.double().square().sum(dim=0))
        total_rows += x.shape[0]
        remaining = train_tokens - sum(item.shape[0] for item in selected)
        if remaining > 0:
            selected.append(x[:remaining].to("cpu", dtype=torch.bfloat16))
        raise _StopForward()

    handle = module.register_forward_pre_hook(hook)
    started = time.time()
    try:
        for index in range(calib_sequences):
            ids = token_ids[index : index + 1, :sequence_length].to(device)
            try:
                model(input_ids=ids, use_cache=False)
            except _StopForward:
                pass
    finally:
        handle.remove()

    train_x = torch.cat(selected, dim=0)[:train_tokens]
    diag = (hessian_diag / max(total_rows, 1)).float()
    return diag, train_x, {
        "captured_activation_rows": float(total_rows),
        "stored_train_rows": float(train_x.shape[0]),
        "capture_seconds": time.time() - started,
        "hessian_diag_min": float(diag.min()),
        "hessian_diag_max": float(diag.max()),
        "hessian_diag_mean": float(diag.mean()),
    }


def load_nowag_linear_vq(nowag_repo: str):
    repo = str(Path(nowag_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    # NoWag imports wandb from utility modules even though VQ initialization
    # does not use it.  The experiment environment has a protobuf-incompatible
    # wandb installation, so keep that unrelated optional dependency outside
    # this diagnostic rather than mutating either repository.
    if "wandb" not in sys.modules:
        wandb_stub = types.ModuleType("wandb")
        wandb_stub.init = lambda *args, **kwargs: None
        wandb_stub.log = lambda *args, **kwargs: None
        wandb_stub.finish = lambda *args, **kwargs: None
        sys.modules["wandb"] = wandb_stub
    from src.quantize_compress import LinearVQ  # type: ignore

    return LinearVQ


@torch.no_grad()
def run_full_nowag_initialization(
    weight: torch.Tensor,
    hessian_diag: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[nn.Module, Dict[str, float]]:
    LinearVQ = load_nowag_linear_vq(args.nowag_repo)
    quantizer = LinearVQ(weight=weight.float(), verbose=False)
    quantizer.hessianDiag = hessian_diag.float()
    started = time.time()
    quantizer.compress(
        d=args.vector_dim,
        n_bits=args.bits_per_weight,
        n_inits=1,
        n_iter=args.lloyd_iters,
        ignore_norms=True,
        normalizer_kwargs={
            "norm_order": [0, 1],
            "zero": [False, False],
            "p": 2,
        },
        initialize_method="kmeans++",
        initialize_kwargs={
            "deterministic": False,
            "multiple_each_time": 1.0,
        },
        half_precision=bool(args.half_kmeans),
    )
    # The diagnostic trains assignment probabilities only.  Freeze the NoWag
    # codebook and normalization parameters so their gradients do not obscure
    # the probability-path measurement or inflate activation memory.
    for parameter in quantizer.parameters():
        parameter.requires_grad_(False)
    reconstructed = quantizer.reconstruct().float()
    error = reconstructed - weight.float()
    weighted_mse = (error.square() * hessian_diag.unsqueeze(0)).mean()
    return quantizer, {
        "initialization_seconds": time.time() - started,
        "codebook_size": float(quantizer.codebook.shape[0]),
        "vector_dim": float(quantizer.codebook.shape[1]),
        "num_vectors": float(quantizer.assignments.numel()),
        "initial_relative_l2": float(error.norm() / weight.float().norm().clamp_min(1e-12)),
        "initial_diag_weighted_mse": float(weighted_mse),
        "used_codewords": float(torch.unique(quantizer.assignments).numel()),
    }


@torch.no_grad()
def normalized_vectors_and_weights(
    quantizer: nn.Module,
    weight: torch.Tensor,
    hessian_diag: torch.Tensor,
    vector_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    normalized = quantizer.normalizer.normalize(weight.float())
    original_columns = normalized.shape[1]
    pad = (-original_columns) % vector_dim
    if pad:
        normalized = F.pad(normalized, (0, pad), value=float(normalized.mean()))
        hessian_diag = F.pad(hessian_diag, (0, pad), value=0.0)
    vector_weights = hessian_diag.unsqueeze(0).expand(normalized.shape[0], -1)
    return (
        normalized.reshape(-1, vector_dim),
        vector_weights.reshape(-1, vector_dim),
        original_columns,
    )


def weighted_distance(
    vectors: torch.Tensor,
    vector_weights: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    # The omitted x^T W x term is constant across codewords.
    return vector_weights @ codebook.square().T - 2.0 * (vectors * vector_weights) @ codebook.T


def full_hessian_objective(
    reference: torch.Tensor,
    quantized: torch.Tensor,
    hessian: torch.Tensor,
) -> float:
    error = reference.float() - quantized.float()
    return float((((error @ hessian.float()) * error).sum()).detach())


def _project_vector_gptq_errors(
    values: torch.Tensor,
    upper_block: torch.Tensor,
) -> torch.Tensor:
    """Apply the exact right-side triangular solve used by GPTQ feedback."""
    # Preserve the legacy per-coordinate lower bound. Cholesky factors normally
    # have a much larger positive diagonal, but retaining the guard keeps the
    # helper well-defined for degenerate numerical fixtures too.
    stabilized_upper = upper_block.clone()
    stabilized_upper.diagonal().clamp_min_(1e-12)
    return torch.linalg.solve_triangular(
        stabilized_upper.transpose(0, 1),
        values.transpose(-2, -1),
        upper=False,
    ).transpose(-2, -1)


def _projected_squared_distances(
    projected_values: torch.Tensor,
    projected_candidates: torch.Tensor,
) -> torch.Tensor:
    """Compute shared- or per-row-candidate squared distances."""
    value_norms = projected_values.square().sum(dim=1, keepdim=True)
    if projected_candidates.ndim == 2:
        candidate_norms = projected_candidates.square().sum(dim=1).unsqueeze(0)
        distances = projected_values @ projected_candidates.transpose(0, 1)
    elif projected_candidates.ndim == 3:
        if projected_candidates.shape[0] != projected_values.shape[0]:
            raise ValueError("per-row candidates must match the value batch")
        candidate_norms = projected_candidates.square().sum(dim=2)
        distances = torch.bmm(
            projected_candidates,
            projected_values.unsqueeze(2),
        ).squeeze(2)
    else:
        raise ValueError("projected_candidates must be rank 2 or 3")
    distances.mul_(-2.0).add_(value_norms).add_(candidate_norms)
    return distances


def _vector_gptq_candidate_losses(
    base: torch.Tensor,
    candidates: torch.Tensor,
    upper_block: torch.Tensor,
) -> torch.Tensor:
    """Vectorize the legacy coordinate feedback loss exactly in d-space.

    The sequential recurrence solves ``error @ upper_block = base - candidate``.
    Applying that triangular solve once and taking its squared norm is the same
    objective, but maps the all-codeword comparison to a small transform plus
    one matrix multiplication.
    """
    projected_base = _project_vector_gptq_errors(base, upper_block)
    projected_candidates = _project_vector_gptq_errors(candidates, upper_block)
    return _projected_squared_distances(projected_base, projected_candidates)


@torch.no_grad()
def _build_codebook_neighbor_table(
    codebook: torch.Tensor,
    candidate_count: int = 128,
    row_chunk: int = 256,
) -> torch.Tensor:
    """Build local codeword neighborhoods while always retaining the anchor ID."""
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    codebook_size = codebook.shape[0]
    if codebook_size == 0:
        raise ValueError("codebook must contain at least one codeword")
    local_count = min(int(candidate_count), codebook_size)
    neighbors = torch.empty(
        codebook_size,
        local_count,
        device=codebook.device,
        dtype=torch.long,
    )
    codeword_norms = codebook.square().sum(dim=1)
    all_ids = torch.arange(codebook_size, device=codebook.device)
    for start in range(0, codebook_size, row_chunk):
        end = min(start + row_chunk, codebook_size)
        anchor_ids = all_ids[start:end]
        neighbors[start:end, 0] = anchor_ids
        if local_count == 1:
            continue
        distances = codeword_norms[start:end].unsqueeze(1)
        distances = distances + codeword_norms.unsqueeze(0)
        distances = distances - 2.0 * (
            codebook[start:end] @ codebook.transpose(0, 1)
        )
        distances[torch.arange(end - start, device=codebook.device), anchor_ids] = (
            torch.inf
        )
        neighbors[start:end, 1:] = distances.topk(
            local_count - 1,
            dim=1,
            largest=False,
            sorted=True,
        ).indices
    return neighbors


@torch.no_grad()
def vector_codebook_aware_gptq(
    source_quantizer: nn.Module,
    weight: torch.Tensor,
    hessian: torch.Tensor,
    vector_dim: int,
    percdamp: float = 0.1,
    row_chunk: int = 256,
    candidate_count: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Run sequential GPTQ while preserving one codeword ID per d-vector.

    For each contiguous column block, every output row selects one complete
    NoWag codeword.  Candidate losses are simulated through the same triangular
    error feedback used by scalar GPTQ, then the selected vector error is
    propagated to all remaining columns. By default, exact scoring is restricted
    to the 128 nearest codewords around each vector's NoWag assignment. The
    refinement is accepted only when the undamped full-Hessian objective is
    non-worse.
    """
    original = weight.float()
    normalized = source_quantizer.normalizer.normalize(original).float()
    codebook = source_quantizer.codebook.detach().float()
    rows, columns = normalized.shape
    groups_per_row = math.ceil(columns / vector_dim)
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    effective_candidate_count = min(int(candidate_count), codebook.shape[0])
    initial_assignment_matrix = (
        source_quantizer.assignments.detach().long().reshape(rows, groups_per_row)
    )

    # NoWag [column, row] normalization transforms the input-side Hessian by
    # diag(column_norm) H diag(column_norm).  The row norm is constant within
    # each output row and therefore does not change its codeword argmin.
    column_norm = source_quantizer.normalizer.norms[0].detach().float()
    transformed = hessian.float() * column_norm.unsqueeze(0) * column_norm.unsqueeze(1)
    transformed = transformed.clone()
    dead = torch.diag(transformed) <= 0
    transformed[dead, dead] = 1.0
    diagonal = torch.arange(columns, device=transformed.device)
    damp = float(percdamp) * torch.diag(transformed).mean()
    transformed[diagonal, diagonal] += damp

    chol = torch.linalg.cholesky(transformed)
    inverse = torch.cholesky_inverse(chol)
    upper = torch.linalg.cholesky(inverse, upper=True)

    work = normalized.clone()
    work[:, dead] = 0.0
    quantized_normalized = torch.zeros_like(work)
    refined_assignments = torch.empty(
        rows, groups_per_row, device=work.device, dtype=torch.long
    )
    started = time.time()
    neighbor_started = time.time()
    neighbor_table = None
    if effective_candidate_count < codebook.shape[0]:
        neighbor_table = _build_codebook_neighbor_table(
            codebook,
            candidate_count=effective_candidate_count,
            row_chunk=row_chunk,
        )
    neighbor_build_seconds = time.time() - neighbor_started

    for group_index, start in enumerate(range(0, columns, vector_dim)):
        end = min(start + vector_dim, columns)
        width = end - start
        upper_block = upper[start:end, start:end]
        selected_ids = torch.empty(rows, device=work.device, dtype=torch.long)
        projected_candidates = None
        if neighbor_table is None:
            projected_candidates = _project_vector_gptq_errors(
                codebook[:, :width], upper_block
            )

        for row_start in range(0, rows, row_chunk):
            row_end = min(row_start + row_chunk, rows)
            base = work[row_start:row_end, start:end]
            projected_base = _project_vector_gptq_errors(
                base, upper_block
            )
            candidate_ids = None
            chunk_projected_candidates = projected_candidates
            if neighbor_table is not None:
                anchors = initial_assignment_matrix[
                    row_start:row_end, group_index
                ]
                candidate_ids = neighbor_table[anchors]
                chunk_projected_candidates = _project_vector_gptq_errors(
                    codebook[candidate_ids, :width],
                    upper_block,
                )
            losses = _projected_squared_distances(
                projected_base, chunk_projected_candidates
            )
            local_ids = losses.argmin(dim=1)
            if candidate_ids is None:
                selected_ids[row_start:row_end] = local_ids
            else:
                selected_ids[row_start:row_end] = candidate_ids.gather(
                    1, local_ids.unsqueeze(1)
                ).squeeze(1)
            del projected_base, losses, local_ids

        refined_assignments[:, group_index] = selected_ids
        selected = codebook[selected_ids, :width]
        errors = _project_vector_gptq_errors(
            work[:, start:end] - selected,
            upper_block,
        )
        work[:, start:end] = selected
        quantized_normalized[:, start:end] = selected
        if end < columns:
            work[:, end:] -= errors @ upper[start:end, end:]

    refined_weight = source_quantizer.normalizer.denormalize(quantized_normalized).float()
    initial_weight = source_quantizer.reconstruct().float()
    initial_objective = full_hessian_objective(original, initial_weight, hessian)
    refined_objective = full_hessian_objective(original, refined_weight, hessian)
    accepted = refined_objective <= initial_objective * (1.0 + 1e-7)
    candidate_stats = {
        "gptq_codebook_size": int(codebook.shape[0]),
        "gptq_candidate_count": int(effective_candidate_count),
        "gptq_candidate_fraction": float(
            effective_candidate_count / codebook.shape[0]
        ),
        "gptq_neighbor_build_seconds": float(neighbor_build_seconds),
    }
    if not accepted:
        return initial_weight, source_quantizer.assignments.detach().clone(), {
            "gptq_seconds": time.time() - started,
            "initial_full_hessian_objective": initial_objective,
            "refined_full_hessian_objective": refined_objective,
            "accepted_full_hessian_objective": initial_objective,
            "gptq_refinement_accepted": 0.0,
            "assignment_change_rate": 0.0,
            **candidate_stats,
        }

    flat_assignments = refined_assignments.reshape(-1)
    initial_assignments = source_quantizer.assignments.detach().long()
    change_rate = (flat_assignments != initial_assignments).float().mean()
    source_quantizer.assignments.copy_(flat_assignments)
    return refined_weight, flat_assignments, {
        "gptq_seconds": time.time() - started,
        "initial_full_hessian_objective": initial_objective,
        "refined_full_hessian_objective": refined_objective,
        "accepted_full_hessian_objective": refined_objective,
        "gptq_refinement_accepted": 1.0,
        "assignment_change_rate": float(change_rate),
        **candidate_stats,
    }


@torch.no_grad()
def build_local_candidates(
    vectors: torch.Tensor,
    vector_weights: torch.Tensor,
    codebook: torch.Tensor,
    assignments: torch.Tensor,
    candidates: int,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    all_ids = torch.empty(
        vectors.shape[0], candidates, device=vectors.device, dtype=torch.long
    )
    all_distances = torch.empty(
        vectors.shape[0], candidates, device=vectors.device, dtype=torch.float32
    )
    assignment_missing = 0
    started = time.time()

    for start in range(0, vectors.shape[0], chunk_size):
        end = min(start + chunk_size, vectors.shape[0])
        distances = weighted_distance(
            vectors[start:end].float(),
            vector_weights[start:end].float(),
            codebook.float(),
        )
        values, ids = torch.topk(distances, candidates, dim=1, largest=False, sorted=True)
        expected = assignments[start:end].long()
        present = (ids == expected.unsqueeze(1)).any(dim=1)
        if not present.all():
            missing_rows = (~present).nonzero(as_tuple=False).squeeze(1)
            assignment_missing += int(missing_rows.numel())
            ids[missing_rows, -1] = expected[missing_rows]
            values[missing_rows, -1] = distances[missing_rows, expected[missing_rows]]

        position = (ids == expected.unsqueeze(1)).float().argmax(dim=1)
        row = torch.arange(ids.shape[0], device=ids.device)
        first_id = ids[:, 0].clone()
        first_distance = values[:, 0].clone()
        ids[:, 0] = expected
        values[:, 0] = distances[row, expected]
        ids[row, position] = first_id
        values[row, position] = first_distance

        all_ids[start:end] = ids
        all_distances[start:end] = values
        del distances

    return all_ids, all_distances, {
        "candidate_build_seconds": time.time() - started,
        "assignment_missing_from_raw_topk": float(assignment_missing),
    }


def initialize_logits(
    distances: torch.Tensor,
    std: float,
    strength: float,
    seed: int,
    selected_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    centered = distances - distances[:, :1]
    scale = centered.std(dim=1, keepdim=True).clamp_min(1e-12)
    scores = -(centered / scale)
    scores = scores - scores.mean(dim=1, keepdim=True)
    generator = torch.Generator(device=distances.device)
    generator.manual_seed(seed)
    noise = torch.randn(
        scores.shape,
        device=scores.device,
        dtype=torch.float32,
        generator=generator,
    )
    logits = float(std) * (noise + scores * float(strength))
    if selected_positions is None:
        selected_positions = torch.zeros(
            logits.shape[0], device=logits.device, dtype=torch.long
        )
    rows = torch.arange(logits.shape[0], device=logits.device)
    selected = logits[rows, selected_positions]
    maximum = logits.amax(dim=1)
    logits[rows, selected_positions] = torch.maximum(
        selected, maximum + max(float(std), 1e-4)
    )
    return logits


def probability_metrics(logits: torch.Tensor, temperature: float, kappa: float) -> Dict[str, float]:
    with torch.no_grad():
        probabilities = F.softmax(logits.float() * float(kappa) / float(temperature), dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=1)
        normalizer = math.log(probabilities.shape[1])
        max_probability = probabilities.max(dim=1).values
        top2 = torch.topk(logits.float(), min(2, logits.shape[1]), dim=1).values
        gap = top2[:, 0] - top2[:, -1]
        return {
            "entropy_mean": float(entropy.mean()),
            "normalized_entropy_mean": float((entropy / max(normalizer, 1e-12)).mean()),
            "max_probability_mean": float(max_probability.mean()),
            "probability_gt_0_99_rate": float((max_probability > 0.99).float().mean()),
            "probability_gt_0_999_rate": float((max_probability > 0.999).float().mean()),
            "top1_top2_logit_gap_mean": float(gap.mean()),
        }


def gradient_metrics(gradient: torch.Tensor) -> Dict[str, float]:
    grad = gradient.detach().float()
    return {
        "grad_mean_abs": float(grad.abs().mean()),
        "grad_rms": float(grad.square().mean().sqrt()),
        "grad_max_abs": float(grad.abs().max()),
        "grad_lt_1e_12_rate": float((grad.abs() < 1e-12).float().mean()),
        "grad_lt_1e_10_rate": float((grad.abs() < 1e-10).float().mean()),
        "grad_nonfinite_rate": float((~torch.isfinite(grad)).float().mean()),
    }


def denormalize_from_vectors(
    vector_values: torch.Tensor,
    quantizer: nn.Module,
    rows: int,
    padded_columns: int,
    original_columns: int,
) -> torch.Tensor:
    normalized = vector_values.reshape(rows, padded_columns)[:, :original_columns]
    return quantizer.normalizer.denormalize(normalized)


class _LocalCodebookExpectation(torch.autograd.Function):
    """Compute local-codebook expectations without retaining [N, K, d]."""

    @staticmethod
    def forward(
        ctx,
        probabilities: torch.Tensor,
        candidate_ids: torch.Tensor,
        codebook: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        rows = probabilities.shape[0]
        output = torch.empty(
            rows,
            codebook.shape[1],
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            candidates = codebook[candidate_ids[start:end]]
            output[start:end] = (
                probabilities[start:end].unsqueeze(2) * candidates
            ).sum(dim=1)
        ctx.save_for_backward(candidate_ids, codebook)
        ctx.chunk_size = chunk_size
        return output

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        candidate_ids, codebook = ctx.saved_tensors
        rows, candidates_per_row = candidate_ids.shape
        probability_gradient = torch.empty(
            rows,
            candidates_per_row,
            device=output_gradient.device,
            dtype=output_gradient.dtype,
        )
        for start in range(0, rows, ctx.chunk_size):
            end = min(start + ctx.chunk_size, rows)
            candidates = codebook[candidate_ids[start:end]]
            probability_gradient[start:end] = (
                output_gradient[start:end].unsqueeze(1) * candidates
            ).sum(dim=2)
        return probability_gradient, None, None, None


class LocalVectorGumbelQuantizer(nn.Module):
    def __init__(
        self,
        quantizer: nn.Module,
        candidate_ids: torch.Tensor,
        initial_logits: torch.Tensor,
        original_columns: int,
    ):
        super().__init__()
        self.source_quantizer = quantizer
        self.register_buffer("codebook", quantizer.codebook.detach().float())
        self.register_buffer("candidate_ids", candidate_ids.long())
        self.quant_logits = nn.Parameter(initial_logits.float().clone())
        self.rows = int(quantizer.out_features)
        self.padded_columns = int(quantizer.padded_in_features)
        self.original_columns = int(original_columns)

    def forward(
        self,
        temperature: float,
        kappa: float,
        add_gumbel: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.quant_logits * float(kappa)
        if add_gumbel:
            eps = 1e-8
            uniform = torch.rand_like(logits)
            logits = logits - torch.log(-torch.log(uniform + eps) + eps)
        probabilities = F.softmax(logits / float(temperature), dim=1)
        expectation_chunk = max(
            1,
            int(os.environ.get("VECTOR_GSQ_EXPECTATION_CHUNK", "65536")),
        )
        soft_vectors = _LocalCodebookExpectation.apply(
            probabilities,
            self.candidate_ids,
            self.codebook,
            expectation_chunk,
        )
        weight = denormalize_from_vectors(
            soft_vectors,
            self.source_quantizer,
            self.rows,
            self.padded_columns,
            self.original_columns,
        )
        return weight, probabilities, soft_vectors

    @torch.no_grad()
    def hard_weight(self) -> Tuple[torch.Tensor, torch.Tensor]:
        local_ids = self.quant_logits.argmax(dim=1)
        rows = torch.arange(local_ids.numel(), device=local_ids.device)
        global_ids = self.candidate_ids[rows, local_ids]
        return self.weight_from_global_ids(global_ids), global_ids

    @torch.no_grad()
    def weight_from_global_ids(self, global_ids: torch.Tensor) -> torch.Tensor:
        vectors = self.codebook[global_ids]
        return denormalize_from_vectors(
            vectors,
            self.source_quantizer,
            self.rows,
            self.padded_columns,
            self.original_columns,
        )


def cancellation_ratio(
    soft_vectors: torch.Tensor,
    soft_vector_gradient: torch.Tensor,
    candidate_vectors: torch.Tensor,
) -> Dict[str, float]:
    with torch.no_grad():
        contributions = soft_vector_gradient.unsqueeze(1) * (
            candidate_vectors - soft_vectors.detach().unsqueeze(1)
        )
        signed = contributions.sum(dim=2).abs()
        absolute = contributions.abs().sum(dim=2).clamp_min(1e-30)
        ratio = signed / absolute
        return {
            "dimension_cancellation_ratio_mean": float(ratio.mean()),
            "dimension_cancellation_ratio_median": float(ratio.median()),
            "dimension_cancellation_ratio_lt_0_1_rate": float((ratio < 0.1).float().mean()),
        }


def full_codebook_probe(
    vectors: torch.Tensor,
    vector_weights: torch.Tensor,
    codebook: torch.Tensor,
    assignments: torch.Tensor,
    args: argparse.Namespace,
    temperature: float,
    kappa: float,
) -> Dict[str, Dict[str, float]]:
    count = min(args.full_probe_vectors, vectors.shape[0])
    x = vectors[:count].float()
    weights = vector_weights[:count].float()
    full_distances = weighted_distance(x, weights, codebook.float())
    full_logits = initialize_logits(
        full_distances,
        args.std,
        args.strength,
        args.seed,
        selected_positions=assignments[:count].long(),
    )
    full_logits = nn.Parameter(full_logits)

    eps = 1e-8
    uniform = torch.rand_like(full_logits)
    noise = -torch.log(-torch.log(uniform + eps) + eps)
    probabilities = F.softmax((full_logits * kappa + noise) / temperature, dim=1)
    soft_vectors = probabilities @ codebook.float()
    soft_vectors.retain_grad()
    loss = (weights * (soft_vectors - x).square()).mean()
    loss.backward()

    full_result = {
        "loss": float(loss.detach()),
        **probability_metrics(full_logits, temperature, kappa),
        **gradient_metrics(full_logits.grad),
        **cancellation_ratio(
            soft_vectors,
            soft_vectors.grad,
            codebook.float().unsqueeze(0).expand(count, -1, -1),
        ),
        "initial_assignment_top1_rate": float(
            (full_logits.argmax(dim=1) == assignments[:count]).float().mean()
        ),
    }

    local_k = min(args.local_candidates, codebook.shape[0])
    values, ids = torch.topk(full_distances.detach(), local_k, dim=1, largest=False)
    expected = assignments[:count].long()
    rows = torch.arange(count, device=x.device)
    present = (ids == expected.unsqueeze(1)).any(dim=1)
    ids[~present, -1] = expected[~present]
    values[~present, -1] = full_distances.detach()[rows[~present], expected[~present]]
    position = (ids == expected.unsqueeze(1)).float().argmax(dim=1)
    first_ids = ids[:, 0].clone()
    first_values = values[:, 0].clone()
    ids[:, 0] = expected
    values[:, 0] = full_distances.detach()[rows, expected]
    ids[rows, position] = first_ids
    values[rows, position] = first_values

    local_logits = nn.Parameter(initialize_logits(values, args.std, args.strength, args.seed))
    uniform = torch.rand_like(local_logits)
    noise = -torch.log(-torch.log(uniform + eps) + eps)
    local_probabilities = F.softmax((local_logits * kappa + noise) / temperature, dim=1)
    local_candidates = codebook.float()[ids]
    local_soft_vectors = (local_probabilities.unsqueeze(2) * local_candidates).sum(dim=1)
    local_soft_vectors.retain_grad()
    local_loss = (weights * (local_soft_vectors - x).square()).mean()
    local_loss.backward()
    local_result = {
        "loss": float(local_loss.detach()),
        **probability_metrics(local_logits, temperature, kappa),
        **gradient_metrics(local_logits.grad),
        **cancellation_ratio(
            local_soft_vectors,
            local_soft_vectors.grad,
            local_candidates,
        ),
        "initial_assignment_top1_rate": float((local_logits.argmax(dim=1) == 0).float().mean()),
    }
    return {"full_4096": full_result, f"local_{local_k}": local_result}


def train_schedule(
    schedule: Schedule,
    source_quantizer: nn.Module,
    candidate_ids: torch.Tensor,
    candidate_distances: torch.Tensor,
    original_columns: int,
    dense_weight: torch.Tensor,
    train_x_cpu: torch.Tensor,
    initial_assignments: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, object]:
    seed_everything(args.seed)
    initial_logits = initialize_logits(
        candidate_distances,
        args.std,
        args.strength,
        args.seed,
    )
    quantizer = LocalVectorGumbelQuantizer(
        source_quantizer,
        candidate_ids,
        initial_logits,
        original_columns,
    ).to(dense_weight.device)
    optimizer = Lion(
        [quantizer.quant_logits],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.steps, 1),
        eta_min=args.lr * 0.1,
    )

    train_x = train_x_cpu.to(dense_weight.device, dtype=torch.float32)
    with torch.no_grad():
        dense_y = F.linear(train_x, dense_weight.float())
        initial_weight, initial_global_ids = quantizer.hard_weight()
        initial_hard_loss = F.mse_loss(F.linear(train_x, initial_weight.float()), dense_y)

    checkpoints = {0, 1, 5, 10, 25, 50, 100, 200, args.steps - 1}
    trace: List[Dict[str, float]] = []
    started = time.time()
    for step in range(args.steps):
        t = step / max(args.steps - 1, 1)
        temperature = schedule.temperature_start + (
            schedule.temperature_end - schedule.temperature_start
        ) * t
        kappa = schedule.kappa_start + (schedule.kappa_end - schedule.kappa_start) * t

        optimizer.zero_grad(set_to_none=True)
        soft_weight, _probabilities, soft_vectors = quantizer(
            temperature,
            kappa,
            add_gumbel=True,
        )
        soft_vectors.retain_grad()
        output = F.linear(train_x, soft_weight.float())
        loss = F.mse_loss(output, dense_y)
        loss.backward()

        if step in checkpoints:
            candidates = quantizer.codebook[quantizer.candidate_ids]
            item = {
                "step": float(step),
                "temperature": float(temperature),
                "kappa": float(kappa),
                "loss": float(loss.detach()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **probability_metrics(quantizer.quant_logits, temperature, kappa),
                **gradient_metrics(quantizer.quant_logits.grad),
                **cancellation_ratio(soft_vectors, soft_vectors.grad, candidates),
            }
            with torch.no_grad():
                hard_weight, hard_ids = quantizer.hard_weight()
                item["hard_output_mse"] = float(
                    F.mse_loss(F.linear(train_x, hard_weight.float()), dense_y)
                )
                item["assignment_switch_rate"] = float(
                    (hard_ids != initial_assignments.long()).float().mean()
                )
            trace.append(item)

        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        final_weight, final_ids = quantizer.hard_weight()
        final_hard_loss = F.mse_loss(F.linear(train_x, final_weight.float()), dense_y)
        relative_l2 = (
            (final_weight.float() - dense_weight.float()).norm()
            / dense_weight.float().norm().clamp_min(1e-12)
        )
        switch_rate = (final_ids != initial_global_ids).float().mean()

    return {
        "schedule": asdict(schedule),
        "training_seconds": time.time() - started,
        "initial_hard_output_mse": float(initial_hard_loss),
        "final_hard_output_mse": float(final_hard_loss),
        "hard_output_mse_relative_change": float(
            (final_hard_loss - initial_hard_loss) / initial_hard_loss.clamp_min(1e-30)
        ),
        "final_weight_relative_l2": float(relative_l2),
        "final_assignment_switch_rate": float(switch_rate),
        "trace": trace,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_payload = torch.load(args.token_cache, map_location="cpu", weights_only=True)
    token_ids = token_payload["input_ids"]

    model_started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    module = get_submodule(model, args.target)
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Target is not Linear: {args.target}: {type(module).__name__}")

    hessian_diag, train_x, capture_stats = capture_linear_inputs(
        model,
        module,
        token_ids,
        args.calib_sequences,
        args.sequence_length,
        args.train_tokens,
        device,
    )
    weight = module.weight.detach().float().clone()
    shape = list(weight.shape)
    del model, module
    gc.collect()
    torch.cuda.empty_cache()

    source_quantizer, init_stats = run_full_nowag_initialization(
        weight,
        hessian_diag,
        args,
    )
    vectors, vector_weights, original_columns = normalized_vectors_and_weights(
        source_quantizer,
        weight,
        hessian_diag,
        args.vector_dim,
    )
    codebook = source_quantizer.codebook.detach().float()
    assignments = source_quantizer.assignments.detach().long()
    candidate_ids, candidate_distances, candidate_stats = build_local_candidates(
        vectors,
        vector_weights,
        codebook,
        assignments,
        args.local_candidates,
        args.candidate_chunk,
    )

    original_schedule = Schedule(
        name="gsq_original",
        temperature_start=2.0,
        temperature_end=0.05,
        kappa_start=100.0,
        kappa_end=500.0,
    )
    gentle_schedule = Schedule(
        name="gradient_friendly_control",
        temperature_start=2.0,
        temperature_end=0.2,
        kappa_start=25.0,
        kappa_end=100.0,
    )

    probes = {
        "original_schedule_start": full_codebook_probe(
            vectors,
            vector_weights,
            codebook,
            assignments,
            args,
            original_schedule.temperature_start,
            original_schedule.kappa_start,
        ),
        "original_schedule_end": full_codebook_probe(
            vectors,
            vector_weights,
            codebook,
            assignments,
            args,
            original_schedule.temperature_end,
            original_schedule.kappa_end,
        ),
    }

    schedules = []
    for schedule in (original_schedule, gentle_schedule):
        schedules.append(
            train_schedule(
                schedule,
                source_quantizer,
                candidate_ids,
                candidate_distances,
                original_columns,
                weight,
                train_x,
                assignments,
                args,
            )
        )
        gc.collect()
        torch.cuda.empty_cache()

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment": "full_nowag_d6_then_local_vector_gsq_gradient_diagnostic",
        "arguments": vars(args),
        "model_load_and_capture_total_seconds": time.time() - model_started,
        "target_shape": shape,
        "capture": capture_stats,
        "nowag_initialization": init_stats,
        "local_candidates": candidate_stats,
        "full_codebook_probes": probes,
        "schedules": schedules,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
