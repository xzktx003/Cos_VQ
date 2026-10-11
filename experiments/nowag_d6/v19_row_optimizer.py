"""One V19 pass over independent output rows, with deployed-weight acceptance."""

import copy
import math
import time

import torch
import torch.nn.functional as F

from experiments.nowag_d6.scale_interval_solver import solve_scale_intervals


def _nearest_scales(target):
    """Both neighboring legal positive FP16 scales, computed on target's device."""
    clipped = target.clamp(2**-24, 65504)
    rounded = clipped.half()
    # Positive finite half values are ordered by their bit patterns. Torch 2.4
    # CUDA has no Half nextafter kernel; integer neighbors work on both devices.
    bits = rounded.view(torch.int16)
    below = (bits - 1).clamp(1, 0x7bff).view(torch.float16).double()
    above = (bits + 1).clamp(1, 0x7bff).view(torch.float16).double()
    return torch.stack((torch.where(rounded.double() > clipped, below, rounded.double()),
                        torch.where(rounded.double() < clipped, above, rounded.double())), -1)


def _majorizer(h, vector_dim, eta_relative, majorizer='block_frobenius'):
    columns = h.shape[0]
    blocks = math.ceil(columns / vector_dim)
    padded = blocks * vector_dim
    diagonal, rho = [], []
    for i in range(blocks):
        start, end = i * vector_dim, min((i + 1) * vector_dim, columns)
        slab = F.pad(h[start:end], (0, padded - columns, 0, vector_dim - (end-start)))
        norms = slab.reshape(vector_dim, blocks, vector_dim).square().sum((0, 2)).sqrt()
        rho.append(norms.sum() - norms[i])
        diagonal.append(slab[:, start:start+vector_dim])
    rho = torch.stack(rho)
    eta = max(float(h.diag().abs().max()) * eta_relative, torch.finfo(torch.float64).eps)
    if majorizer == 'diagonal_heuristic':
        rho = torch.zeros_like(rho)
        diagonal = [torch.diag(block.diag()) for block in diagonal]
    elif majorizer != 'block_frobenius':
        raise ValueError('Unknown majorizer')
    d = torch.stack(diagonal) + (rho + eta)[:, None, None] * torch.eye(vector_dim, device=h.device, dtype=h.dtype)
    if torch.any(torch.linalg.eigvalsh(d) <= 0):
        raise ValueError('Block majorizer must be positive definite')
    return d, rho, eta


def _surrogate_coefficients(unit_blocks, current_scale, ideal_gradient, d):
    """Quadratic terms tangent at the exactly representable ideal FP64 anchor."""
    z = unit_blocks * current_scale
    dt = torch.einsum('bij,bj->bi', d, z) - ideal_gradient
    alpha = torch.einsum('bi,bij,bj->b', unit_blocks, d, unit_blocks)
    beta = (unit_blocks * dt).sum(-1)
    return dt, alpha, beta


@torch.no_grad()
def optimize_rows(state, reference_weight, hessian, *, active_positions=8,
                  candidates_per_ranking=4, row_batch_size=32, eta_relative=1e-6,
                  deployment_dtype=torch.bfloat16, mode='interval', majorizer='block_frobenius'):
    """Optimize one Linear without changing its representation or its input state.

    Dense quadratic/candidate work stays on hessian.device in FP64. Only compact
    active-candidate coefficients are transferred to the CPU interval solver.
    The surrogate is tangent to the ideal FP64 Hessian objective at the original
    FP16-index/scale anchor. A separate deployed objective uses the original
    FP32 multiplication order and deployment cast. Only strict improvement of
    that second objective is committed; surrogate improvement alone is not a
    claim about the rounded model. Dense refits and acceptance are row-batched.
    diagonal_heuristic is an ablation without the majorization guarantee.
    """
    started = time.perf_counter()
    if mode not in ('interval', 'scale_only', 'fixed_scale') or min(row_batch_size, candidates_per_ranking) < 1 or active_positions < 0:
        raise ValueError('Invalid mode or search sizes')
    if not math.isfinite(eta_relative) or eta_relative <= 0:
        raise ValueError('eta_relative must be positive and finite')
    if any(key in state for key in ('group_scale', 'residual_A', 'residual_B')):
        raise ValueError('Only original row-scale logical states are supported')
    rows, columns = map(int, state['shape'])
    dim = int(state['vector_dim'])
    if min(rows, columns, dim) <= 0 or tuple(reference_weight.shape) != (rows, columns):
        raise ValueError('Invalid weight geometry')
    if deployment_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('Unsupported deployment dtype')
    device = hessian.device
    h = hessian.to(dtype=torch.float64)
    if h.shape != (columns, columns) or not torch.isfinite(h).all() or not torch.allclose(h, h.T, atol=1e-10, rtol=1e-10):
        raise ValueError('Hessian must be finite and symmetric with matching shape')
    reference = reference_weight.to(device=device, dtype=torch.float64)
    if not torch.isfinite(reference).all():
        raise ValueError('Reference must be finite')
    blocks = math.ceil(columns / dim)
    padded = blocks * dim
    ids = state['assignments'].to(device=device, dtype=torch.long).reshape(rows, blocks)
    cb = state['codebook'].to(device=device, dtype=torch.float32)
    scales = state['row_norm'].to(device=device, dtype=torch.float32).reshape(rows)
    col = state['col_norm'].to(device=device, dtype=torch.float32).reshape(columns)
    if any(state[k].dtype != torch.float16 for k in ('codebook', 'row_norm', 'col_norm')):
        raise ValueError('Codebook and norms must have FP16 storage')
    if cb.ndim != 2 or cb.shape[1] != dim or not 0 <= int(ids.min()) <= int(ids.max()) < len(cb):
        raise ValueError('Invalid codebook or assignments')
    if int(state['assignment_bits']) <= 0 or len(cb) > 2**int(state['assignment_bits']):
        raise ValueError('Codebook exceeds assignment bit budget')
    if not all(bool(torch.isfinite(t).all()) for t in (cb, scales, col)):
        raise ValueError('Logical parameters must be finite')
    d, rho, eta = _majorizer(h, dim, eta_relative, majorizer)
    col_blocks = F.pad(col.double(), (0, padded-columns)).reshape(blocks, dim)
    result = copy.deepcopy(state)
    new_ids, new_scales = ids.clone(), scales.clone()
    stats = {'rows': rows, 'updated_rows': 0, 'zero_scale_rows': 0, 'unchanged_rows': 0,
             'changed_assignments': 0, 'changed_scales': 0, 'interval_count': 0,
             'objective_before': 0., 'objective_after': 0.,
             'proposal_wins': {'scale_only': 0, 'interval': 0, 'interval_refit': 0},
             'rho_min': float(rho.min()), 'rho_max': float(rho.max()),
             'rho_mean': float(rho.mean()), 'eta': eta, 'mode': mode,
             'majorizer': majorizer, 'surrogate_valid_majorizer': majorizer == 'block_frobenius'}

    def decode(assignments, signed_scales):
        return ((cb[assignments].reshape(len(assignments), padded)[:, :columns]
                 * signed_scales.float()[:, None]) * col).to(deployment_dtype).double()

    def refit(unit, hu, target, current):
        a, b = (hu * unit).sum(-1), (hu * target).sum(-1)
        optimum = torch.where(a > 0, b / a.clamp_min(torch.finfo(torch.float64).tiny), current)
        return _nearest_scales(optimum)

    for batch_start in range(0, rows, row_batch_size):
        batch_end = min(rows, batch_start + row_batch_size)
        deployed = decode(ids[batch_start:batch_end], scales[batch_start:batch_end])
        error = deployed - reference[batch_start:batch_end]
        gradient = error @ h
        before_values = (gradient * error).sum(-1)
        if not torch.isfinite(before_values).all():
            raise ValueError('Nonfinite deployed baseline objective')
        batch_units = (cb[ids[batch_start:batch_end]].double() * col_blocks[None]
                       * scales[batch_start:batch_end].sign().double()[:, None, None])
        flat_units = batch_units.reshape(batch_end-batch_start, padded)[:, :columns]
        unit_h = flat_units @ h
        batch_current = scales[batch_start:batch_end].abs().double()
        # Ideal and deployed residuals are intentionally distinct.
        ideal_gradient = unit_h * batch_current[:, None] - reference[batch_start:batch_end] @ h
        original_refits = refit(flat_units, unit_h, reference[batch_start:batch_end], batch_current)
        pending = []
        for offset, row in enumerate(range(batch_start, batch_end)):
            before = float(before_values[offset])
            stats['objective_before'] += before
            if scales[row] == 0:
                stats['zero_scale_rows'] += 1
                stats['unchanged_rows'] += 1
                stats['objective_after'] += before
                continue
            sign = scales[row].sign().double()
            current = scales[row].abs().double()
            anchor_ids = ids[row]
            unit_blocks = batch_units[offset]
            proposals, proposal_scales, names = [], [], []

            def add(assignment, values, name):
                for value in values.reshape(-1):
                    proposals.append(assignment)
                    proposal_scales.append(value * sign)
                    names.append(name)

            if mode != 'fixed_scale':
                add(anchor_ids, original_refits[offset], 'scale_only')
            selected_unit = None
            if mode in ('interval', 'fixed_scale'):
                b = F.pad(ideal_gradient[offset], (0, padded-columns)).reshape(blocks, dim)
                dt, anchor_alpha, anchor_beta = _surrogate_coefficients(unit_blocks, current, b, d)
                active = torch.argsort(b.square().sum(-1), descending=True, stable=True)[:min(active_positions, blocks)]
                mask = torch.ones(blocks, device=device, dtype=torch.bool)
                mask[active] = False
                aa = [[float(anchor_alpha[mask].sum())]]
                bb = [[float(anchor_beta[mask].sum())]]
                candidate_ids = []
                # Batch all active codebook scans; memory is O(active*K*dim).
                candidates = cb.double()[None] * col_blocks[active, None] * sign
                alpha = torch.einsum('bki,bij,bkj->bk', candidates, d[active], candidates)
                beta = torch.einsum('bki,bi->bk', candidates, dt[active])
                fixed_scores = alpha * current.square() - 2 * beta * current
                ideal = torch.where(alpha > 0, beta / alpha.clamp_min(torch.finfo(torch.float64).tiny), current)
                legal = _nearest_scales(ideal)
                free_scores = (alpha[..., None]*legal.square() - 2*beta[..., None]*legal).min(-1).values
                top = min(candidates_per_ranking, len(cb))
                fixed_top = torch.argsort(fixed_scores, stable=True)[:, :top]
                free_top = torch.argsort(free_scores, stable=True)[:, :top]
                for j, position in enumerate(active):
                    pool = torch.unique(torch.cat((anchor_ids[position:position+1], fixed_top[j], free_top[j])), sorted=True)
                    candidate_ids.append(pool)
                    aa.append(alpha[j, pool].cpu().numpy())
                    bb.append(beta[j, pool].cpu().numpy())
                domain = (float(current), float(current)) if mode == 'fixed_scale' else (2**-24, 65504.)
                solved = solve_scale_intervals(aa, bb, float(current), domain, scale_format='fp16')
                stats['interval_count'] += solved.interval_count
                selected = anchor_ids.clone()
                for j, position in enumerate(active):
                    selected[position] = candidate_ids[j][int(solved.indices[j+1])]
                add(selected, current.new_tensor([solved.scale]), 'interval')
                selected_unit = (cb[selected].double() * col_blocks * sign).reshape(-1)[:columns]
            pending.append({'row': row, 'before': before, 'ids': proposals,
                            'scales': proposal_scales, 'names': names, 'unit': selected_unit})
        if not pending:
            continue
        if mode == 'interval':
            selected_units = torch.stack([entry['unit'] for entry in pending])
            target_rows = torch.tensor([entry['row'] for entry in pending], device=device)
            fitted = refit(selected_units, selected_units @ h, reference[target_rows], scales[target_rows].abs().double())
            for entry, values in zip(pending, fitted):
                assignment = entry['ids'][-1]
                for value in values:
                    entry['ids'].append(assignment)
                    entry['scales'].append(value * scales[entry['row']].sign())
                    entry['names'].append('interval_refit')
        proposal_ids = torch.stack([assignment for entry in pending for assignment in entry['ids']])
        proposal_scales_tensor = torch.stack([scale for entry in pending for scale in entry['scales']])
        target_rows = torch.tensor([entry['row'] for entry in pending for _ in entry['ids']], device=device)
        proposal_error = decode(proposal_ids, proposal_scales_tensor) - reference[target_rows]
        all_values = ((proposal_error @ h) * proposal_error).sum(-1)
        all_values = torch.where(torch.isfinite(all_values), all_values, torch.full_like(all_values, float('inf')))
        begin = 0
        for entry in pending:
            row, before = entry['row'], entry['before']
            count = len(entry['ids'])
            values = all_values[begin:begin+count]
            winner = int(values.argmin())
            after = float(values[winner])
            if after < before - 1e-10 * max(1., abs(before)):
                new_ids[row] = proposal_ids[begin+winner]
                new_scales[row] = proposal_scales_tensor[begin+winner].float()
                stats['updated_rows'] += 1
                stats['proposal_wins'][entry['names'][winner]] += 1
                stats['objective_after'] += after
            else:
                stats['unchanged_rows'] += 1
                stats['objective_after'] += before
            begin += count
    stats['changed_assignments'] = int((new_ids != ids).sum())
    stats['changed_scales'] = int((new_scales != scales).sum())
    result['assignments'] = new_ids.to(device=state['assignments'].device, dtype=state['assignments'].dtype).reshape_as(state['assignments'])
    result['row_norm'] = new_scales.to(device=state['row_norm'].device, dtype=state['row_norm'].dtype).reshape_as(state['row_norm'])
    stats['true_decrease'] = stats['objective_before'] - stats['objective_after']
    stats['seconds'] = time.perf_counter() - started
    return result, stats
