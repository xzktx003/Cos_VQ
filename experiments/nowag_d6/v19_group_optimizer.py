"""V19 hard-code search over *existing* FP16 groups, without E2E training.

The surrogate uses FP64 ideal linear decoding and its current full-row residual.
Acceptance separately uses the original FP32 decoder, cast to deployment_dtype.
Thus finite-precision acceptance is measured, rather than inferred from the
ideal-arithmetic majorization theorem. No codebook or normalization is trained.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import torch

from experiments.nowag_d6.scale_interval_solver import solve_scale_intervals


_MIN_SCALE = 2.0 ** -24
_MAX_SCALE = 65504.0
_REL_TOL = 1e-10
_ABS_TOL = 1e-12
_AUDIT_REL_TOL = 1e-8
_AUDIT_ABS_TOL = 1e-10


@dataclass
class _GroupGeometry:
    start: int
    end: int
    code_indices: tuple[int, ...]
    local_slices: tuple[slice, ...]
    component_slices: tuple[slice, ...]
    mutable: tuple[bool, ...]
    d_matrix: torch.Tensor
    cholesky: torch.Tensor
    rho: torch.Tensor
    eta: float
    candidate_alpha: dict[int, torch.Tensor] = field(default_factory=dict)


def _build_geometry(hessian, group_index, group_size, vector_dim, *, eta_relative, majorizer='block_frobenius'):
    """Build the block majorizer or explicitly unguaranteed diagonal ablation."""
    columns = hessian.shape[0]
    start, end = group_index * group_size, min((group_index + 1) * group_size, columns)
    if not 0 <= start < end:
        raise ValueError("group index lies outside the weight row")
    code_indices = tuple(range(start // vector_dim, (end - 1) // vector_dim + 1))
    local_slices, components, mutable, labels = [], [], [], []
    for position, code in enumerate(code_indices):
        code_start, code_end = code * vector_dim, min((code + 1) * vector_dim, columns)
        low, high = max(start, code_start), min(end, code_end)
        local_slices.append(slice(low - start, high - start))
        components.append(slice(low - code_start, high - code_start))
        mutable.append(code_start // group_size == (code_end - 1) // group_size)
        labels.extend([position] * (high - low))
    local_h = hessian[start:end, start:end]
    label = torch.tensor(labels, dtype=torch.long, device=hessian.device)
    count = len(code_indices)
    pair_ids = label[:, None] * count + label[None, :]
    squared = torch.zeros(count * count, dtype=torch.float64, device=hessian.device)
    squared.scatter_add_(0, pair_ids.reshape(-1), local_h.square().reshape(-1))
    block_norms = squared.reshape(count, count).sqrt()
    # Zeroing the diagonal before summation avoids subtractive cancellation.
    block_norms.fill_diagonal_(0.0)
    rho = block_norms.sum(dim=1)
    eta = eta_relative * max(float(local_h.diagonal().abs().mean()), 1e-12)
    d_matrix = local_h * (label[:, None] == label[None, :])
    if majorizer == 'diagonal_heuristic':
        rho = torch.zeros_like(rho)
        d_matrix = torch.diag(local_h.diag())
    elif majorizer != 'block_frobenius':
        raise ValueError('Unknown majorizer')
    d_matrix = d_matrix + torch.diag(rho[label] + eta)
    cholesky = torch.linalg.cholesky(d_matrix)
    return _GroupGeometry(
        start, end, code_indices, tuple(local_slices), tuple(components),
        tuple(mutable), d_matrix, cholesky, rho, eta,
    )


def _decode(ids, scales, codebook, row_norm, col_norm, columns, vector_dim, group_size,
            *, deployment_dtype=None):
    weight = codebook[ids].reshape(ids.shape[0], -1)[:, :columns]
    # Preserve the checkpoint decoder's order: ((code * row) * column) * group.
    weight = weight * row_norm[:, None]
    weight = weight * col_norm[None, :]
    weight = weight * scales.repeat_interleave(group_size, dim=1)[:, :columns]
    return weight.to(deployment_dtype).double() if deployment_dtype is not None else weight


def _decode_group(ids, scales, codebook, row_norm, col_norm, geometry, vector_dim,
                  *, deployment_dtype=None):
    coordinates = torch.arange(geometry.start, geometry.end, device=ids.device)
    positions = coordinates.div(vector_dim, rounding_mode="floor") - geometry.code_indices[0]
    components = coordinates.remainder(vector_dim)
    weight = codebook[ids[:, positions], components[None, :]]
    weight = weight * row_norm[:, None]
    weight = weight * col_norm[geometry.start:geometry.end][None, :]
    weight = weight * scales[:, None]
    return weight.to(deployment_dtype).double() if deployment_dtype is not None else weight


def _scale_neighbours(alpha, beta, anchor):
    """Rounded quadratic optimum and adjacent FP16 values for deployed checks."""
    target = torch.where(alpha > 0, beta / alpha, anchor).clamp(_MIN_SCALE, _MAX_SCALE)
    nearest = target.half()
    # torch 2.4 CUDA has no Half nextafter kernel. Positive finite binary16
    # numbers have monotonically increasing bit patterns, including subnormals.
    bits = nearest.view(torch.int16)
    lower = (bits - 1).clamp(1, 0x7BFF).view(torch.float16)
    upper = (bits + 1).clamp(1, 0x7BFF).view(torch.float16)
    return [value.double().clamp(_MIN_SCALE, _MAX_SCALE) for value in (nearest, lower, upper)]


def _fit_scales(unit, deployed_group, deployed_gradient, local_h, anchor):
    alpha = ((unit @ local_h) * unit).sum(dim=1).clamp_min(0.0)
    beta = (((deployed_group @ local_h) - deployed_gradient) * unit).sum(dim=1)
    return _scale_neighbours(alpha, beta, anchor)


def _interval_proposal(ids, current_scales, row_norm, col_norm, codebook, ideal_group,
                       ideal_error, ideal_gradient, geometry, active_positions,
                       candidates_per_ranking, *, fixed_scale=False):
    """Build coefficients on the tensor device; transfer only the compact pool."""
    row_count = ids.shape[0]
    target = ideal_group - torch.cholesky_solve(ideal_gradient.T, geometry.cholesky).T
    unit = ideal_group / current_scales[:, None]
    alpha_current, beta_current, residual_scores = [], [], []
    for support in geometry.local_slices:
        d_block = geometry.d_matrix[support, support]
        current_unit = unit[:, support]
        alpha_current.append(((current_unit @ d_block) * current_unit).sum(dim=1))
        beta_current.append(((current_unit @ d_block) * target[:, support]).sum(dim=1))
        residual_scores.append(ideal_error[:, support].square().sum(dim=1))
    current_a = torch.stack(alpha_current, dim=1)
    current_b = torch.stack(beta_current, dim=1)
    scores = torch.stack(residual_scores, dim=1)
    mutable = torch.tensor(geometry.mutable, device=ids.device, dtype=torch.bool)
    active_count = min(active_positions, sum(geometry.mutable))
    scores[:, ~mutable] = -math.inf
    active = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :active_count]
    active_mask = torch.zeros_like(scores, dtype=torch.bool)
    active_mask.scatter_(1, active, True)
    fixed_a = current_a.masked_fill(active_mask, 0.0).sum(dim=1)
    fixed_b = current_b.masked_fill(active_mask, 0.0).sum(dim=1)
    ranking_count = min(candidates_per_ranking, codebook.shape[0])
    pool_width = 1 + 2 * ranking_count
    packed_ids = torch.empty((row_count, active_count, pool_width), dtype=torch.long, device=ids.device)
    packed_a = torch.empty_like(packed_ids, dtype=torch.float64)
    packed_b = torch.empty_like(packed_a)
    scanned_positions = 0
    for position, (support, component) in enumerate(zip(geometry.local_slices, geometry.component_slices)):
        occurrences = (active == position).nonzero(as_tuple=False)
        if not occurrences.numel():
            continue
        selected_rows, slots = occurrences[:, 0], occurrences[:, 1]
        column_slice = col_norm[geometry.start + support.start:geometry.start + support.stop]
        base_codes = codebook[:, component] * column_slice[None, :]
        d_block = geometry.d_matrix[support, support]
        if position not in geometry.candidate_alpha:
            geometry.candidate_alpha[position] = ((base_codes @ d_block) * base_codes).sum(dim=1)
        row_factors = row_norm[selected_rows]
        alpha = row_factors[:, None].square() * geometry.candidate_alpha[position][None, :]
        beta = ((target[selected_rows, support] @ d_block) @ base_codes.T) * row_factors[:, None]
        scale = current_scales[selected_rows, None]
        fixed_scores = alpha * scale.square() - 2 * beta * scale
        fitted = torch.where(alpha > 0, beta / alpha, scale).clamp(_MIN_SCALE, _MAX_SCALE).half().double()
        fitted_scores = alpha * fitted.square() - 2 * beta * fitted
        fixed_best = torch.argsort(fixed_scores, dim=1, stable=True)[:, :ranking_count]
        fitted_best = torch.argsort(fitted_scores, dim=1, stable=True)[:, :ranking_count]
        choices = torch.cat((ids[selected_rows, position, None], fixed_best, fitted_best), dim=1)
        packed_ids[selected_rows, slots] = choices
        packed_a[selected_rows, slots] = torch.gather(alpha, 1, choices)
        packed_b[selected_rows, slots] = torch.gather(beta, 1, choices)
        scanned_positions += len(selected_rows)
    # These arrays contain <= active_positions * (1+2*K) entries per row.
    active_cpu = active.cpu().numpy()
    pool_cpu, alpha_cpu, beta_cpu = (value.cpu().numpy() for value in (packed_ids, packed_a, packed_b))
    fixed_cpu = torch.stack((fixed_a, fixed_b), dim=1).cpu().numpy()
    scale_cpu = current_scales.cpu().numpy()
    proposal_ids = ids.clone()
    proposal_scales, changes, intervals, pool_candidates = [], [], 0, 0
    for row in range(row_count):
        alpha_rows, beta_rows, code_choices = [], [], []
        for slot in range(active_count):
            seen, keep = set(), []
            for offset, code in enumerate(pool_cpu[row, slot]):
                if int(code) not in seen:
                    seen.add(int(code))
                    keep.append(offset)
            code_choices.append(pool_cpu[row, slot, keep])
            alpha_rows.append(alpha_cpu[row, slot, keep])
            beta_rows.append(beta_cpu[row, slot, keep])
            pool_candidates += len(keep)
        alpha_rows.append([fixed_cpu[row, 0]])
        beta_rows.append([fixed_cpu[row, 1]])
        solved = solve_scale_intervals(
            alpha_rows, beta_rows, float(scale_cpu[row]),
            (float(scale_cpu[row]), float(scale_cpu[row])) if fixed_scale else (_MIN_SCALE, _MAX_SCALE),
            scale_format="fp16",
        )
        proposal_scales.append(solved.scale)
        intervals += solved.interval_count
        for slot in range(active_count):
            changes.append((row, int(active_cpu[row, slot]), int(code_choices[slot][solved.indices[slot]])))
    if changes:
        updates = torch.tensor(changes, dtype=torch.long, device=ids.device)
        proposal_ids[updates[:, 0], updates[:, 1]] = updates[:, 2]
    return proposal_ids, torch.tensor(proposal_scales, dtype=torch.float64, device=ids.device), {
        "intervals_evaluated": intervals,
        "candidate_positions_scanned": scanned_positions,
        "candidate_codes_scanned": scanned_positions * codebook.shape[0],
        "pool_candidates": pool_candidates,
    }


def _validate(state, reference_weight, hessian, active_positions, candidates_per_ranking,
              groups_per_row, row_batch_size, eta_relative, deployment_dtype, mode):
    for name, value in (("active_positions", active_positions), ("candidates_per_ranking", candidates_per_ranking),
                        ("groups_per_row", groups_per_row), ("row_batch_size", row_batch_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if mode not in {"interval", "scale_only", "fixed_scale"}:
        raise ValueError("mode must be interval, scale_only or fixed_scale")
    if not math.isfinite(eta_relative) or eta_relative <= 0:
        raise ValueError("eta_relative must be positive and finite")
    if deployment_dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise ValueError("deployment_dtype must be a floating-point torch dtype")
    required = {"shape", "vector_dim", "assignment_bits", "group_size", "group_scale", "assignments",
                "codebook", "row_norm", "col_norm"}
    if required - state.keys():
        raise ValueError(f"grouped checkpoint is missing {sorted(required - state.keys())}")
    if "residual_A" in state or "residual_B" in state:
        raise ValueError("residual compensation is outside this grouped V19 representation")
    rows, columns = (int(value) for value in state["shape"])
    vector_dim, group_size = int(state["vector_dim"]), int(state["group_size"])
    if min(rows, columns, vector_dim, group_size) <= 0:
        raise ValueError("weight dimensions, vector_dim and group_size must be positive")
    groups = math.ceil(columns / group_size)
    expected = {"row_norm": (rows,), "col_norm": (columns,), "group_scale": (rows, groups)}
    for key in ("codebook", "row_norm", "col_norm", "group_scale"):
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float16:
            raise ValueError(f"{key} must retain FP16 storage")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{key} must be finite")
        if key in expected and tuple(value.shape) != expected[key]:
            raise ValueError(f"{key} has an invalid shape")
    if not bool((state["group_scale"] > 0).all()):
        raise ValueError("group_scale must be positive and finite")
    codebook = state["codebook"]
    if codebook.ndim != 2 or codebook.shape[1] != vector_dim or not codebook.shape[0]:
        raise ValueError("codebook shape does not match vector_dim")
    assignments = state["assignments"]
    if not isinstance(assignments, torch.Tensor) or assignments.is_floating_point() or assignments.dtype == torch.bool:
        raise ValueError("assignments must use an integer tensor")
    if assignments.numel() != rows * math.ceil(columns / vector_dim):
        raise ValueError("assignments do not match padded row geometry")
    try:
        storage_max = torch.iinfo(assignments.dtype).max
    except TypeError as error:
        raise ValueError("assignments must use an integer storage dtype") from error
    assignment_bits = int(state["assignment_bits"])
    if assignment_bits <= 0 or codebook.shape[0] > 2 ** assignment_bits:
        raise ValueError("codebook exceeds assignment bit budget")
    if codebook.shape[0] - 1 > storage_max:
        raise ValueError("codebook indices exceed assignment storage dtype")
    integer_ids = assignments.long()
    if int(integer_ids.min()) < 0 or int(integer_ids.max()) >= codebook.shape[0]:
        raise ValueError("assignment index outside codebook")
    if tuple(reference_weight.shape) != (rows, columns) or not bool(torch.isfinite(reference_weight).all()):
        raise ValueError("reference_weight must match shape and be finite")
    if tuple(hessian.shape) != (columns, columns) or not bool(torch.isfinite(hessian).all()):
        raise ValueError("hessian must be finite and square with input dimension")
    if not torch.allclose(hessian, hessian.T, rtol=1e-6, atol=1e-8):
        raise ValueError("hessian must be symmetric")
    return rows, columns, vector_dim, group_size, groups


@torch.no_grad()
def optimize_groups(state, reference_weight, hessian, *, active_positions=8,
                    candidates_per_ranking=4, groups_per_row=1, row_batch_size=32,
                    eta_relative=1e-6, deployment_dtype=torch.bfloat16, mode="interval",
                    majorizer='block_frobenius'):
    """Return an independent state and audit stats after one preselected group pass.

    Groups are selected per row by the initial ideal weight-residual squared norm,
    then processed in that order. The selection budget is fixed before updates.
    Matrix work stays on hessian.device; only compact candidate pools go to the
    CPU interval solver. All crossing d-dimensional code indices stay frozen.
    """
    started = time.perf_counter()
    if majorizer not in ('block_frobenius', 'diagonal_heuristic'):
        raise ValueError('Unknown majorizer')
    rows, columns, vector_dim, group_size, group_count = _validate(
        state, reference_weight, hessian, active_positions, candidates_per_ranking,
        groups_per_row, row_batch_size, eta_relative, deployment_dtype, mode,
    )
    result = copy.deepcopy(state)
    result["assignments"] = result["assignments"].contiguous()
    device = hessian.device
    h = hessian.detach().to(dtype=torch.float64)
    h = (h + h.T) * 0.5
    code32 = state["codebook"].detach().to(device=device, dtype=torch.float32)
    code64 = code32.double()
    column32 = state["col_norm"].detach().to(device=device, dtype=torch.float32)
    column64 = column32.double()
    original_ids = state["assignments"].reshape(rows, -1)
    output_ids = result["assignments"].reshape(rows, -1)
    geometry_cache = {}
    stats = {
        "mode": mode, "majorizer": majorizer, "surrogate_valid_majorizer": majorizer == 'block_frobenius',
        "rows": rows, "columns": columns, "group_size": group_size,
        "groups_per_row_requested": groups_per_row, "groups_per_row": min(groups_per_row, group_count),
        "active_positions": active_positions, "candidates_per_ranking": candidates_per_ranking,
        "row_batch_size": row_batch_size, "eta_relative": eta_relative,
        "deployment_dtype": str(deployment_dtype), "compute_device": str(device),
        "scale_domain": [_MIN_SCALE, _MAX_SCALE],
        "group_selection": "initial_ideal_weight_residual_squared_norm",
        "groups_selected": 0, "groups_accepted": 0, "groups_no_update": 0,
        "assignments_changed": 0, "scales_changed": 0, "accepted_objective_increases": 0,
        "intervals_evaluated": 0, "candidate_positions_scanned": 0,
        "candidate_codes_scanned": 0, "pool_candidates": 0,
        "proposal_evaluations": 0, "nonfinite_proposals": 0,
        "winner_counts": {"anchor": 0, "scale_only": 0, "interval": 0, "interval_refit": 0},
        "deployed_objective_before": 0.0, "deployed_objective_after": 0.0,
        "ideal_objective_before": 0.0, "ideal_objective_after": 0.0,
        "acceptance_relative_tolerance": _REL_TOL, "acceptance_absolute_tolerance": _ABS_TOL,
        "objective_audit_relative_tolerance": _AUDIT_REL_TOL,
        "objective_audit_absolute_tolerance": _AUDIT_ABS_TOL,
        "objective_update_drift_max_abs": 0.0,
    }
    for row_start in range(0, rows, row_batch_size):
        row_end = min(rows, row_start + row_batch_size)
        batch_rows = row_end - row_start
        ids = original_ids[row_start:row_end].to(device=device, dtype=torch.long).clone()
        scales = state["group_scale"][row_start:row_end].to(device=device, dtype=torch.float64).clone()
        row32 = state["row_norm"][row_start:row_end].to(device=device, dtype=torch.float32)
        row64 = row32.double()
        reference = reference_weight[row_start:row_end].detach().to(device=device, dtype=torch.float64)
        ideal_weight = _decode(ids, scales, code64, row64, column64, columns, vector_dim, group_size)
        deployed_weight = _decode(ids, scales.float(), code32, row32, column32, columns, vector_dim,
                                  group_size, deployment_dtype=deployment_dtype)
        if not bool(torch.isfinite(deployed_weight).all()):
            raise ValueError("initial checkpoint has nonfinite deployed weights")
        ideal_error, deployed_error = ideal_weight - reference, deployed_weight - reference
        ideal_gradient, deployed_gradient = ideal_error @ h, deployed_error @ h
        current_objective = (deployed_error * deployed_gradient).sum(dim=1)
        if not bool(torch.isfinite(current_objective).all()) or not bool(torch.isfinite(ideal_gradient).all()):
            raise ValueError("nonfinite initial Hessian objective or residual gradient")
        initial_objective = current_objective.clone()
        stats["deployed_objective_before"] += float(current_objective.sum())
        stats["ideal_objective_before"] += float((ideal_error * ideal_gradient).sum())
        group_scores = torch.zeros((batch_rows, group_count), dtype=torch.float64, device=device)
        group_labels = torch.arange(columns, device=device).div(group_size, rounding_mode="floor")
        group_scores.scatter_add_(1, group_labels[None, :].expand(batch_rows, -1), ideal_error.square())
        selected_groups = torch.argsort(group_scores, dim=1, descending=True, stable=True)[:, :stats["groups_per_row"]]
        for rank in range(stats["groups_per_row"]):
            for group in torch.unique(selected_groups[:, rank]).tolist():
                selected_rows = (selected_groups[:, rank] == group).nonzero(as_tuple=False).flatten()
                if group not in geometry_cache:
                    geometry_cache[group] = _build_geometry(h, group, group_size, vector_dim, eta_relative=eta_relative,
                                                            majorizer=majorizer)
                geometry = geometry_cache[group]
                start, end = geometry.start, geometry.end
                count = len(selected_rows)
                stats["groups_selected"] += count
                code_indices = torch.tensor(geometry.code_indices, dtype=torch.long, device=device)
                group_ids = ids[selected_rows[:, None], code_indices[None, :]]
                anchor_scale = scales[selected_rows, group]
                anchor_deployed = deployed_weight[selected_rows, start:end]
                anchor_ideal = ideal_weight[selected_rows, start:end]
                gradient_actual = deployed_gradient[selected_rows, start:end]
                local_h = h[start:end, start:end]
                local_row32, local_row64 = row32[selected_rows], row64[selected_rows]
                unit_anchor = anchor_ideal / anchor_scale[:, None]
                best_delta = torch.zeros(count, device=device, dtype=torch.float64)
                best_ids, best_scale = group_ids.clone(), anchor_scale.clone()
                best_deployed, best_ideal = anchor_deployed.clone(), anchor_ideal.clone()
                best_kind = torch.zeros(count, device=device, dtype=torch.long)
                stats["proposal_evaluations"] += count  # Anchor.

                def inspect(proposed_ids, proposed_scale, kind):
                    nonlocal best_delta, best_ids, best_scale, best_deployed, best_ideal, best_kind
                    candidate = _decode_group(
                        proposed_ids, proposed_scale.float(), code32, local_row32, column32,
                        geometry, vector_dim, deployment_dtype=deployment_dtype,
                    )
                    finite = torch.isfinite(candidate).all(dim=1)
                    safe_candidate = torch.where(finite[:, None], candidate, anchor_deployed)
                    difference = safe_candidate - anchor_deployed
                    delta = 2 * (gradient_actual * difference).sum(dim=1)
                    delta = delta + ((difference @ local_h) * difference).sum(dim=1)
                    finite = finite & torch.isfinite(delta)
                    better = finite & (delta < best_delta)
                    best_delta = torch.where(better, delta, best_delta)
                    best_ids = torch.where(better[:, None], proposed_ids, best_ids)
                    best_scale = torch.where(better, proposed_scale, best_scale)
                    best_deployed = torch.where(better[:, None], candidate, best_deployed)
                    ideal_candidate = _decode_group(
                        proposed_ids, proposed_scale, code64, local_row64, column64, geometry, vector_dim,
                    )
                    best_ideal = torch.where(better[:, None], ideal_candidate, best_ideal)
                    best_kind = torch.where(better, kind, best_kind)
                    stats["proposal_evaluations"] += count
                    stats["nonfinite_proposals"] += int((~finite).sum())

                if mode != 'fixed_scale':
                    for proposed_scale in _fit_scales(unit_anchor, anchor_deployed, gradient_actual, local_h, anchor_scale):
                        inspect(group_ids, proposed_scale, 1)
                if mode in ('interval', 'fixed_scale'):
                    interval_ids, interval_scale, search_stats = _interval_proposal(
                        group_ids, anchor_scale, local_row64, column64, code64, anchor_ideal,
                        ideal_error[selected_rows, start:end], ideal_gradient[selected_rows, start:end],
                        geometry, active_positions, candidates_per_ranking, fixed_scale=mode == 'fixed_scale',
                    )
                    for key, value in search_stats.items():
                        stats[key] += value
                    inspect(interval_ids, interval_scale, 2)
                    interval_unit = _decode_group(
                        interval_ids, torch.ones_like(interval_scale), code64, local_row64,
                        column64, geometry, vector_dim,
                    )
                    if mode != 'fixed_scale':
                        for proposed_scale in _fit_scales(interval_unit, anchor_deployed, gradient_actual, local_h, anchor_scale):
                            inspect(interval_ids, proposed_scale, 3)
                tolerance = _ABS_TOL + _REL_TOL * current_objective[selected_rows].abs()
                accepted = best_delta < -tolerance
                best_ids = torch.where(accepted[:, None], best_ids, group_ids)
                best_scale = torch.where(accepted, best_scale, anchor_scale)
                best_deployed = torch.where(accepted[:, None], best_deployed, anchor_deployed)
                best_ideal = torch.where(accepted[:, None], best_ideal, anchor_ideal)
                best_kind = torch.where(accepted, best_kind, 0)
                actual_change, ideal_change = best_deployed - anchor_deployed, best_ideal - anchor_ideal
                ids[selected_rows[:, None], code_indices[None, :]] = best_ids
                scales[selected_rows, group] = best_scale
                deployed_weight[selected_rows, start:end] = best_deployed
                ideal_weight[selected_rows, start:end] = best_ideal
                deployed_error[selected_rows, start:end] += actual_change
                ideal_error[selected_rows, start:end] += ideal_change
                deployed_gradient[selected_rows] += actual_change @ h[start:end, :]
                ideal_gradient[selected_rows] += ideal_change @ h[start:end, :]
                current_objective[selected_rows] += torch.where(accepted, best_delta, 0.0)
                accepted_count = int(accepted.sum())
                stats["groups_accepted"] += accepted_count
                stats["groups_no_update"] += count - accepted_count
                for kind, name in enumerate(stats["winner_counts"]):
                    stats["winner_counts"][name] += int((best_kind == kind).sum())
        final_objective = ((deployed_error @ h) * deployed_error).sum(dim=1)
        audit_tolerance = _AUDIT_ABS_TOL + _AUDIT_REL_TOL * initial_objective.abs().clamp_min(1.0)
        if not bool(torch.isfinite(final_objective).all()):
            raise RuntimeError("nonfinite final deployed objective")
        increases = final_objective > initial_objective + audit_tolerance
        stats["accepted_objective_increases"] += int(increases.sum())
        if bool(increases.any()):
            raise RuntimeError("final deployed row objective increased beyond audit tolerance")
        drift = (current_objective - final_objective).abs()
        stats["objective_update_drift_max_abs"] = max(stats["objective_update_drift_max_abs"], float(drift.max()))
        if bool((drift > audit_tolerance).any()):
            raise RuntimeError("incremental objective disagrees with full residual recomputation")
        stats["deployed_objective_after"] += float(final_objective.sum())
        stats["ideal_objective_after"] += float(((ideal_error @ h) * ideal_error).sum())
        output_ids[row_start:row_end].copy_(ids.to(device=output_ids.device, dtype=output_ids.dtype))
        result["group_scale"][row_start:row_end].copy_(
            scales.to(device=result["group_scale"].device, dtype=torch.float16)
        )
    stats["assignments_changed"] = int((result["assignments"].long() != state["assignments"].long()).sum())
    stats["scales_changed"] = int((result["group_scale"] != state["group_scale"]).sum())
    stats["deployed_objective_decrease"] = stats["deployed_objective_before"] - stats["deployed_objective_after"]
    stats["ideal_objective_decrease"] = stats["ideal_objective_before"] - stats["ideal_objective_after"]
    rho_values = torch.cat([geometry.rho for geometry in geometry_cache.values()])
    stats["rho_mean"] = float(rho_values.mean())
    stats["rho_max"] = float(rho_values.max())
    stats["eta_min"] = min(geometry.eta for geometry in geometry_cache.values())
    stats["eta_max"] = max(geometry.eta for geometry in geometry_cache.values())
    stats["cached_groups"] = len(geometry_cache)
    stats["globally_frozen_crossing_indices_per_row"] = sum(
        (index * vector_dim) // group_size != (min((index + 1) * vector_dim, columns) - 1) // group_size
        for index in range(math.ceil(columns / vector_dim))
    )
    stats["elapsed_seconds"] = time.perf_counter() - started
    # Common report fields shared with the whole-row optimizer.
    stats["objective_before"] = stats["deployed_objective_before"]
    stats["objective_after"] = stats["deployed_objective_after"]
    stats["changed_assignments"] = stats["assignments_changed"]
    stats["changed_scales"] = stats["scales_changed"]
    stats["interval_count"] = stats["intervals_evaluated"]
    stats["true_decrease"] = stats["deployed_objective_decrease"]
    stats["proposal_wins"] = {name: count for name, count in stats["winner_counts"].items() if name != "anchor"}
    stats["seconds"] = stats["elapsed_seconds"]
    return result, stats
