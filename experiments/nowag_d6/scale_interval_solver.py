"""Event sweep for separable shared-scale quadratics (paper v19, section 4.2).

Minimize sum_i(alpha[i,k_i] * s**2 - 2 * beta[i,k_i] * s).
Candidate-independent constants are omitted. Inputs are float64 coefficients;
extended-precision intermediates reduce intersection and sweep cancellation.
This is a numerical prototype, not an exact rational-arithmetic certificate.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class ScaleIntervalResult:
    indices: np.ndarray
    scale: float
    objective: float
    interval_count: int


@lru_cache(maxsize=1)
def _fp16_scales():
    # Match intersection precision so searchsorted does not cast the full table
    # on every interval of every group.
    values = np.arange(1, 0x7c00, dtype=np.uint16).view(np.float16).astype(np.longdouble)
    values.flags.writeable = False
    return values


def _lower_envelope(alpha, beta):
    """Return (start, original candidate id) segments, sorted by start."""
    # Descending slope: the lowest line moves toward smaller slopes as s grows.
    order = sorted(range(len(alpha)), key=lambda k: (-alpha[k], -beta[k], k))
    hull, starts = [], []
    for k in order:
        if hull and alpha[k] == alpha[hull[-1]]:
            continue  # First equal slope has smallest intercept, then smallest id.
        start = np.longdouble(-np.inf)
        while hull:
            previous = hull[-1]
            start = 2 * (beta[previous] - beta[k]) / (alpha[previous] - alpha[k])
            if start > starts[-1]:
                break
            hull.pop()
            starts.pop()
        if not hull:
            start = np.longdouble(-np.inf)
        hull.append(k)
        starts.append(start)
    return list(zip(starts, hull))


def _interval_scales(a, b, left, right, current, scale_format):
    target = np.clip(b / a if a > 0 else current, left, right)
    if scale_format == 'continuous':
        # Intersection bounds may be longdouble; don't let float64 conversion
        # create a point outside an extremely narrow representable interval.
        low, high = float(left), float(right)
        if np.longdouble(low) < left:
            low = np.nextafter(low, np.inf)
        if np.longdouble(high) > right:
            high = np.nextafter(high, -np.inf)
        if low > high:
            return []
        return [float(np.clip(float(target), low, high))]
    legal = _fp16_scales()
    begin = int(np.searchsorted(legal, left, side='left'))
    end = int(np.searchsorted(legal, right, side='right'))
    if begin == end:
        return []
    pivot = int(np.searchsorted(legal, target, side='left'))
    return [float(legal[j]) for j in sorted({max(begin, min(end - 1, pivot - 1)),
                                           max(begin, min(end - 1, pivot))})]


def solve_scale_intervals(alpha, beta, current_scale, scale_domain, *, scale_format='continuous'):
    """Solve one group's finite-candidate upper bound over a positive closed domain.

    ``alpha`` and ``beta`` are matching ragged sequences, one row per position.
    alpha must be nonnegative; a zero-alpha candidate must also have beta zero.
    Empty candidate rows are invalid, but zero positions form a constant problem.
    The domain must contain current_scale; fp16 mode requires a legal FP16 anchor.
    Tie policy: closest scale to current, then smaller scale; at that scale choose
    the lowest original candidate index among equal minimum lines. No tolerance
    merges distinct events. interval_count includes intervals without FP16 values.
    The solver includes anchor/scale-only states in its upper-bound search space;
    the caller must separately compare their *true reconstruction* objectives.
    """
    if scale_format not in ('continuous', 'fp16'):
        raise ValueError('scale_format must be continuous or fp16')
    try:
        left, right = (float(v) for v in scale_domain)
        current = float(current_scale)
        aa = [np.asarray(row, dtype=np.float64) for row in alpha]
        bb = [np.asarray(row, dtype=np.float64) for row in beta]
    except (TypeError, OverflowError) as error:
        raise ValueError('Invalid coefficients or scale domain') from error
    if not np.isfinite([left, right, current]).all() or not 0 < left <= current <= right:
        raise ValueError('Positive finite closed domain must contain current_scale')
    if scale_format == 'fp16' and (current > 65504 or float(np.float16(current)) != current):
        raise ValueError('Current scale must be a positive finite FP16 value')
    if len(aa) != len(bb):
        raise ValueError('alpha and beta must have matching position counts')
    for a, b in zip(aa, bb):
        if a.ndim != 1 or b.shape != a.shape or not a.size:
            raise ValueError('Every position requires matching nonempty candidate rows')
        if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or ((a == 0) & (b != 0)).any():
            raise ValueError('Expected finite alpha>=0, with beta=0 for zero-alpha candidates')
    aa = [a.astype(np.longdouble) for a in aa]
    bb = [b.astype(np.longdouble) for b in bb]
    selections = np.empty(len(aa), dtype=np.int64)
    events = []
    for i, (a, b) in enumerate(zip(aa, bb)):
        envelope = _lower_envelope(a, b)
        for start, k in envelope:
            if start <= left:
                selections[i] = k
            elif start < right:
                events.append((start, i, k))
    events.sort()
    total_a = sum((a[k] for a, k in zip(aa, selections)), np.longdouble(0))
    total_b = sum((b[k] for b, k in zip(bb, selections)), np.longdouble(0))
    best_key, best_scale = None, current
    intervals = 0

    def inspect(low, high):
        nonlocal best_key, best_scale, intervals
        intervals += 1
        for scale in _interval_scales(total_a, total_b, low, high, current, scale_format):
            s = np.longdouble(scale)
            objective = (total_a * s - 2 * total_b) * s
            key = (objective, abs(s - current), s)
            if best_key is None or key < best_key:
                best_key, best_scale = key, scale

    previous, index = np.longdouble(left), 0
    while index < len(events):
        boundary = events[index][0]
        inspect(previous, boundary)
        while index < len(events) and events[index][0] == boundary:
            _, position, k = events[index]
            old = selections[position]
            total_a += aa[position][k] - aa[position][old]
            total_b += bb[position][k] - bb[position][old]
            selections[position] = k
            index += 1
        previous = boundary
    inspect(previous, np.longdouble(right))
    if best_key is None:
        raise ValueError('No representable scale in domain')
    # Recover once instead of copying an entire assignment at every event.
    s = np.longdouble(best_scale)
    indices = np.array([int(np.argmin(a * s - 2 * b)) for a, b in zip(aa, bb)], dtype=np.int64)
    objective = sum(((a[k] * s - 2 * b[k]) * s for a, b, k in zip(aa, bb, indices)), np.longdouble(0))
    return ScaleIntervalResult(indices, best_scale, float(objective), intervals)
