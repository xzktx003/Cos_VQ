"""Read-only, bounded finite-candidate diagnostics for V19.

Oracles minimize ideal FP64 quadratic objectives over *all* positive finite
FP16 scales. BF16 deployed errors are measured separately: rounding the dense
weight destroys the quadratic scale model, so there is no deployed-F oracle
claim. All work runs on CPU, including copies of inputs supplied on a GPU.
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np
import torch

from experiments.nowag_d6.scale_interval_solver import _fp16_scales, solve_scale_intervals
from experiments.nowag_d6.v19_group_optimizer import _build_geometry
from experiments.nowag_d6.v19_row_optimizer import _majorizer


_DOMAIN = (2.0**-24, 65504.0)
_ETA = 1e-6


def _scale_options(alpha, beta, anchor):
    """Exact neighboring legal values for a convex scalar quadratic."""
    if alpha < -1e-10 or not np.isfinite([alpha, beta]).all():
        raise ValueError("Invalid quadratic coefficients")
    if alpha <= 0:
        if abs(beta) > 1e-10:
            raise ValueError("Zero-curvature quadratic has a nonzero linear term")
        return [anchor]
    target = np.clip(np.longdouble(beta) / np.longdouble(alpha), *_DOMAIN)
    legal = _fp16_scales()
    index = int(np.searchsorted(legal, target))
    return [float(legal[i]) for i in sorted({max(0, min(len(legal) - 1, index - 1)),
                                            max(0, min(len(legal) - 1, index))})]


def _quadratic(alpha, beta, scale):
    scale = np.longdouble(scale)
    return (np.longdouble(alpha) * scale - 2 * np.longdouble(beta)) * scale


def _rank(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def _decode_row(state, row, *, deployed=False):
    dtype = torch.float32 if deployed else torch.float64
    columns = int(state["shape"][1])
    codebook = state["codebook"].detach().to("cpu", dtype)
    ids = state["assignments"][row].detach().to("cpu", torch.long).reshape(-1)
    value = codebook[ids].reshape(-1)[:columns]
    value = value * state["row_norm"][row].detach().to("cpu", dtype)
    value = value * state["col_norm"].detach().to("cpu", dtype).reshape(-1)
    if "group_scale" in state:
        scales = state["group_scale"][row].detach().to("cpu", dtype)
        value = value * scales.repeat_interleave(int(state["group_size"]))[:columns]
    return value.bfloat16().double() if deployed else value


def _probe_row(state, reference, h, row, active_positions, per_ranking):
    started = time.perf_counter()
    columns, dim = int(state["shape"][1]), int(state["vector_dim"])
    grouped = "group_scale" in state
    ideal = _decode_row(state, row)
    deployed = _decode_row(state, row, deployed=True)
    error, deployed_error = ideal - reference, deployed - reference
    gradient, deployed_gradient = error @ h, deployed_error @ h
    before, deployed_before = float(error @ gradient), float(deployed_error @ deployed_gradient)
    ids = state["assignments"][row].detach().to("cpu", torch.long).reshape(-1)
    row_factor = float(state["row_norm"][row])
    if not grouped and row_factor == 0:
        return {"row": row, "status": "skipped_zero_row_scale", "reason": "The row kernel freezes zero scales."}
    sign = 1.0 if grouped or row_factor > 0 else -1.0
    if grouped:
        group_size = int(state["group_size"])
        scores = [float(error[i:i + group_size].square().sum()) for i in range(0, columns, group_size)]
        group = int(np.argmax(scores))
        geometry = _build_geometry(h, group, group_size, dim, eta_relative=_ETA)
        start, end = geometry.start, geometry.end
        positions = list(geometry.code_indices)
        slices, components, mutable = geometry.local_slices, geometry.component_slices, geometry.mutable
        blocks = [geometry.d_matrix[s, s] for s in slices]
        rho, eta = geometry.rho, geometry.eta
        scale = float(state["group_scale"][row, group])
    else:
        group, start, end = None, 0, columns
        d, rho, eta = _majorizer(h, dim, _ETA)
        positions = list(range(len(ids)))
        slices = tuple(slice(i * dim, min((i + 1) * dim, columns)) for i in positions)
        components = tuple(slice(0, s.stop - s.start) for s in slices)
        mutable = (True,) * len(positions)
        blocks = [d[i, :s.stop - s.start, :s.stop - s.start] for i, s in enumerate(slices)]
        scale = abs(row_factor)
    if not _DOMAIN[0] <= scale <= _DOMAIN[1] or float(np.float16(scale)) != scale:
        raise ValueError("Selected scale must be a positive finite FP16 magnitude")
    local_h = h[start:end, start:end]
    anchor = ideal[start:end]
    unit = anchor / scale
    local_gradient = gradient[start:end]
    cb = state["codebook"].detach().to("cpu", torch.float64)
    col = state["col_norm"].detach().to("cpu", torch.float64).reshape(-1)[start:end]
    factors = col * (row_factor if grouped else sign)
    anchor_a, anchor_b, scores, dt = [], [], [], []
    for support, block in zip(slices, blocks):
        coefficient = block @ anchor[support] - local_gradient[support]
        anchor_a.append(float(unit[support] @ block @ unit[support]))
        anchor_b.append(float(unit[support] @ coefficient))
        dt.append(coefficient)
        priority = error[start:end][support] if grouped else local_gradient[support]
        scores.append(float(priority.square().sum()))
    active = sorted((i for i, can_change in enumerate(mutable) if can_change),
                    key=lambda i: (-scores[i], positions[i]))[:active_positions]
    pools, code_units, aa, bb, rank_changes = [], [], [], [], []
    fixed_a = sum(a for i, a in enumerate(anchor_a) if i not in active)
    fixed_b = sum(b for i, b in enumerate(anchor_b) if i not in active)
    for i in active:
        support = slices[i]
        candidates = cb[:, components[i]] * factors[support]
        alpha = ((candidates @ blocks[i]) * candidates).sum(-1).numpy()
        beta = (candidates @ dt[i]).numpy()
        fixed = alpha * scale**2 - 2 * beta * scale
        free = np.array([min(_quadratic(a, b, s) for s in _scale_options(a, b, scale))
                         for a, b in zip(alpha, beta)])
        current = int(ids[positions[i]])
        alternatives = set(np.argsort(fixed, kind="stable")[:per_ranking].tolist())
        alternatives.update(np.argsort(free, kind="stable")[:per_ranking].tolist())
        pool = [current] + sorted(alternatives - {current})
        pools.append(pool)
        code_units.append(candidates[pool].numpy())
        aa.append(alpha[pool])
        bb.append(beta[pool])
        rank_fixed, rank_free = _rank(fixed[pool]), _rank(free[pool])
        correlation = float(np.corrcoef(rank_fixed, rank_free)[0, 1]) if min(rank_fixed.std(), rank_free.std()) > 0 else None
        rank_changes.append({"code_position": positions[i], "candidate_ids": pool,
                             "fixed_scale_ranks": rank_fixed.tolist(), "refitted_ranks": rank_free.tolist(),
                             "spearman": correlation,
                             "best_candidate_changed": pool[int(np.argmin(fixed[pool]))] != pool[int(np.argmin(free[pool]))]})

    # Only active supports vary between combinations. The shared scale still
    # multiplies ALL inactive and crossing-code contributions in the group.
    coordinates = np.array([j for i in active for j in range(slices[i].start, slices[i].stop)], dtype=np.int64)
    active_h = local_h.numpy()[np.ix_(coordinates, coordinates)]
    unit_np = unit.numpy()
    anchor_active = unit_np[coordinates]
    hu = (local_h @ unit).numpy()
    conditional_target = (local_h @ anchor - local_gradient).numpy()
    true_a = float(unit @ local_h @ unit)
    true_b = float(unit.numpy() @ conditional_target)
    all_combinations = list(itertools.product(*(range(len(p)) for p in pools)))
    anchor_choices = (0,) * len(active)
    def coefficients(choices):
        values = np.concatenate([u[k] for u, k in zip(code_units, choices)]) if choices else np.empty(0)
        difference = values - anchor_active
        a = true_a + 2 * difference @ hu[coordinates] + difference @ active_h @ difference
        b = true_b + difference @ conditional_target[coordinates]
        if a < -1e-10 * max(1., abs(true_a)):
            raise ValueError("H must be positive semidefinite for the quadratic oracle")
        sa = fixed_a + sum(a[k] for a, k in zip(aa, choices))
        sb = fixed_b + sum(b[k] for b, k in zip(bb, choices))
        return max(0., float(a)), float(b), float(sa), float(sb)

    def selection(combinations, objective, *, fixed=False):
        clock = time.perf_counter()
        best, evaluated = None, 0
        for choices in combinations:
            a, b, sa, sb = coefficients(choices)
            a, b = (sa, sb) if objective == "surrogate" else (a, b)
            scales = [scale] if fixed else _scale_options(a, b, scale)
            for s in scales:
                evaluated += 1
                key = (_quadratic(a, b, s), abs(s - scale), s, choices)
                if best is None or key < best[0]:
                    best = key, choices, s
        return {"choices": best[1], "scale": best[2], "candidate_combinations_evaluated": len(combinations),
                "quadratic_scale_evaluations": evaluated, "seconds": time.perf_counter() - clock}

    singles = [c for c in all_combinations if sum(k != 0 for k in c) <= 1]
    pairs = [c for c in all_combinations if sum(k != 0 for k in c) <= 2]
    setup_seconds = time.perf_counter() - started
    selected = {"anchor": {"choices": anchor_choices, "scale": scale,
                           "candidate_combinations_evaluated": 1, "quadratic_scale_evaluations": 1, "seconds": 0.},
                "scale_only": selection([anchor_choices], "true"),
                "fixed_scale": selection(all_combinations, "surrogate", fixed=True),
                "single_refit": selection(singles, "true"),
                "pair_refit": selection(pairs, "true"),
                "oracle_true": selection(all_combinations, "true"),
                "oracle_surrogate": selection(all_combinations, "surrogate")}
    clock = time.perf_counter()
    interval = solve_scale_intervals([[fixed_a]] + aa, [[fixed_b]] + bb, scale, _DOMAIN, scale_format="fp16")
    selected["v19_interval"] = {"choices": tuple(int(k) for k in interval.indices[1:]), "scale": interval.scale,
                                "candidate_combinations_evaluated": None, "quadratic_scale_evaluations": None,
                                "intervals_evaluated": interval.interval_count, "seconds": time.perf_counter() - clock}
    refit = selection([selected["v19_interval"]["choices"]], "true")
    proposals = [selected["anchor"], selected["scale_only"], selected["v19_interval"], refit]

    def true_key(entry):
        a, b, _, _ = coefficients(entry["choices"])
        return _quadratic(a, b, entry["scale"]), abs(entry["scale"] - scale), entry["scale"], entry["choices"]

    selected["v19"] = dict(min(proposals, key=true_key))
    selected["v19"]["seconds"] = selected["scale_only"]["seconds"] + selected["v19_interval"]["seconds"] + refit["seconds"]
    selected["v19"]["candidate_combinations_evaluated"] = len({p["choices"] for p in proposals})
    selected["v19"]["quadratic_scale_evaluations"] = None

    def objectives(entry):
        proposed = unit.clone()
        for slot, (i, k) in enumerate(zip(active, entry["choices"])):
            proposed[slices[i]] = torch.from_numpy(code_units[slot][k])
        delta = proposed * entry["scale"] - anchor
        true = before + float(2 * local_gradient @ delta + delta @ local_h @ delta)
        surrogate = before + float(2 * local_gradient @ delta)
        surrogate += sum(float(delta[s] @ block @ delta[s]) for s, block in zip(slices, blocks))
        return true, surrogate

    for name, entry in selected.items():
        true, surrogate = objectives(entry)
        entry.update(true_objective=true, surrogate_objective=surrogate,
                     active_code_ids=[p[k] for p, k in zip(pools, entry["choices"])],
                     hamming_changes=sum(k != 0 for k in entry["choices"]), ideal_improvement=before - true,
                     surrogate_improvement=before - surrogate, surrogate_true_gap=surrogate - true)
    # Decode endpoint values in the production multiplication order. Reuse the
    # complete deployed residual/gradient so outside-group coupling survives.
    local_positions = torch.arange(start, end).div(dim, rounding_mode="floor")
    local_components = torch.arange(start, end).remainder(dim)
    cb32 = cb.float()
    col32 = col.float()
    deltas = []
    for entry in selected.values():
        updated_ids = ids.clone()
        for i, pool, k in zip(active, pools, entry["choices"]):
            updated_ids[positions[i]] = pool[k]
        value = cb32[updated_ids[local_positions], local_components]
        value *= row_factor if grouped else sign * entry["scale"]
        value *= col32
        if grouped:
            value *= entry["scale"]
        deltas.append(value.bfloat16().double() - deployed[start:end])
    deltas = torch.stack(deltas)
    deployed_values = deployed_before + 2 * (deltas * deployed_gradient[start:end]).sum(-1)
    deployed_values += ((deltas @ local_h) * deltas).sum(-1)
    for entry, value in zip(selected.values(), deployed_values):
        entry.update(deployed_objective=float(value), deployed_improvement=deployed_before - float(value))
        del entry["choices"]

    surrogate_scale = selection([anchor_choices], "surrogate")
    surrogate_single = selection(singles, "surrogate")
    ss = objectives(surrogate_scale)[1]
    si = objectives(surrogate_single)[1]
    tolerance = 1e-9 * max(1., abs(before))

    def synergy(baseline, single, joint):
        return {"scale_optimized_anchor": baseline, "best_single_refit": single, "best_joint": joint,
                "single_stalled": single >= baseline - tolerance,
                "joint_improves": joint < baseline - tolerance, "joint_gain": baseline - joint,
                "cooperative_escape": single >= baseline - tolerance and joint < baseline - tolerance,
                "tolerance": tolerance}

    gap_square = float(local_h.square().sum())
    gap_trace = -float(local_h.diagonal().sum())
    for support, block in zip(slices, blocks):
        gap_square += float(block.square().sum() - 2 * (block * local_h[support, support]).sum())
        gap_trace += float(block.diagonal().sum())
    gap_eigenvalue = None
    if end - start <= 512:
        dense_d = torch.block_diag(*blocks)
        gap_eigenvalue = float(torch.linalg.eigvalsh(dense_d - local_h).min())
    gap_fro = math.sqrt(max(0., gap_square))
    return {"row": row, "status": "completed", "group_index": group, "column_range": [start, end],
            "scale_sign": sign, "anchor_scale": scale, "active_code_positions": [positions[i] for i in active],
            "frozen_cross_boundary_code_positions": [p for p, mutable_ in zip(positions, mutable) if not mutable_],
            "inactive_support_count": len(positions) - len(active), "candidate_pools": pools,
            "enumerated_combinations": len(all_combinations), "methods": selected, "rank_changes": rank_changes,
            "majorizer": {"kind": "block_frobenius", "eta_relative": _ETA, "eta": float(eta),
                          "rho_mean": float(rho.mean()), "rho_max": float(rho.max()),
                          "gap_frobenius": gap_fro, "gap_trace": gap_trace,
                          "relative_gap_frobenius": gap_fro / max(float(local_h.norm()), 1e-30),
                          "gap_min_eigenvalue": gap_eigenvalue,
                          "eigenvalue_audit": "computed" if gap_eigenvalue is not None else "omitted_above_512_columns; valid bound by construction"},
            "surrogate_oracle_gap": selected["v19_interval"]["surrogate_objective"] - selected["oracle_surrogate"]["surrogate_objective"],
            "synergy": {"true": synergy(selected["scale_only"]["true_objective"], selected["single_refit"]["true_objective"], selected["oracle_true"]["true_objective"]),
                        "surrogate": synergy(ss, si, selected["oracle_surrogate"]["surrogate_objective"])},
            "best_listed_deployed_method": min(selected, key=lambda name: selected[name]["deployed_objective"]),
            "shared_setup_seconds": setup_seconds, "deployed_objective_evaluations": len(selected),
            "seconds": time.perf_counter() - started}


@torch.no_grad()
def analyze_linear(state, reference_weight, hessian, *, max_rows=4, active_positions=3,
                   candidates_per_ranking=2, seed=0):
    """Return compact diagnostics without changing the checkpoint or input H.

    At most three active positions and five candidates per position ensure an
    exhaustive <=125-combination oracle. Row-scale states are a whole-row
    specialization with fixed sign, col_norm and codebook; group states also
    freeze row_norm and globally crossing indices. This is one frozen-anchor
    probe, not a pass over the model or a replacement for deployed acceptance.
    """
    started = time.perf_counter()
    if max_rows < 1 or not 0 <= active_positions <= 3 or not 1 <= candidates_per_ranking <= 2:
        raise ValueError("Require max_rows>0, 0<=active_positions<=3 and 1<=candidates_per_ranking<=2")
    rows, columns = map(int, state["shape"])
    dim = int(state["vector_dim"])
    if min(rows, columns, dim) <= 0 or tuple(reference_weight.shape) != (rows, columns):
        raise ValueError("Invalid Linear geometry")
    if any(k in state for k in ("residual_A", "residual_B")):
        raise ValueError("Residual-augmented states are unsupported")
    blocks_per_row = math.ceil(columns / dim)
    assignments = state['assignments']
    if tuple(assignments.shape) not in ((rows * blocks_per_row,), (rows, blocks_per_row)):
        raise ValueError("Assignment geometry does not match the Linear")
    # Production checkpoints may store a flat stream, with padding per row.
    # Normalize only the probe's local view; preserve the caller's stored shape.
    state = dict(state, assignments=assignments.reshape(rows, blocks_per_row))
    if tuple(state["row_norm"].shape) != (rows,) or tuple(state["col_norm"].shape) != (columns,):
        raise ValueError("Norm geometry does not match the Linear")
    if state["codebook"].ndim != 2 or state["codebook"].shape[1] != dim:
        raise ValueError("Invalid codebook geometry")
    if int(state["assignment_bits"]) <= 0 or len(state["codebook"]) > 2**int(state["assignment_bits"]):
        raise ValueError("Codebook exceeds the existing assignment bit budget")
    grouped = "group_scale" in state
    if grouped:
        size = int(state["group_size"])
        if size <= 0 or tuple(state["group_scale"].shape) != (rows, math.ceil(columns / size)):
            raise ValueError("Invalid group-scale geometry")
    for name in ("codebook", "row_norm", "col_norm") + (("group_scale",) if grouped else ()):
        if state[name].dtype != torch.float16 or not bool(torch.isfinite(state[name].detach().cpu()).all()):
            raise ValueError("Codebook and scales require finite FP16 storage")
    if grouped and not bool((state["group_scale"].detach().cpu() > 0).all()):
        raise ValueError("Group scales must be positive")
    h = hessian.detach().to("cpu", torch.float64)
    if h.shape != (columns, columns) or not torch.isfinite(h).all() or not torch.allclose(h, h.T, atol=1e-10, rtol=1e-10):
        raise ValueError("H must be finite, symmetric and match the Linear")
    selected_rows = sorted(np.random.default_rng(seed).choice(rows, min(max_rows, rows), replace=False).tolist())
    results = []
    for row in selected_rows:
        reference = reference_weight[row].detach().to("cpu", torch.float64)
        if not torch.isfinite(reference).all():
            raise ValueError("Reference weight must be finite")
        row_ids = state["assignments"][row].detach().cpu()
        if int(row_ids.min()) < 0 or int(row_ids.max()) >= len(state["codebook"]):
            raise ValueError("Assignment outside codebook")
        results.append(_probe_row(state, reference, h, row, active_positions, candidates_per_ranking))
    return {"schema_version": 1, "supported": True, "scale_layout": "group" if grouped else "row",
            "shape": [rows, columns], "seed": seed, "selected_rows": selected_rows,
            "configuration": {"max_rows": max_rows, "active_positions": active_positions,
                              "candidates_per_ranking": candidates_per_ranking, "scale_domain": list(_DOMAIN)},
            "scope": "Fixed-anchor, shared finite candidate pools; inactive contributions remain in every scale fit.",
            "true_oracle_semantics": "Complete H ideal FP64 linear decoding, including outside-group residual; all positive finite FP16 scales. H must be PSD (as X.T@X/N). Not an oracle over BF16 rounding.",
            "v19_semantics": "Interval upper-bound endpoint plus true-quadratic refit, scale-only and anchor; selection by ideal true F. Deployed BF16 F is reported separately, not used to claim global oracle optimality.",
            "fixed_scale_semantics": "Exact separable surrogate minimizer at anchor scale; exhaustive table used only in this bounded diagnostic.",
            "timing_semantics": "Per-method CPU search time excludes shared candidate/geometry setup and batched endpoint deployment audit. Diagnostic enumeration overhead is not a production speed benchmark.",
            "cost_warning": "Copies full H to CPU; full-row probes evaluate dense H and can require gigabytes and substantial CPU time. Limit representative layers and max_rows.",
            "rows": results, "seconds": time.perf_counter() - started}
