import itertools

import numpy as np
import pytest

from experiments.nowag_d6.scale_interval_solver import solve_scale_intervals


def oracle(alpha, beta, domain, fmt):
    best = np.inf
    legal = np.arange(1, 0x7c00, dtype=np.uint16).view(np.float16).astype(np.float64)
    legal = legal[(legal >= domain[0]) & (legal <= domain[1])]
    for ids in itertools.product(*(range(len(a)) for a in alpha)):
        a = sum(alpha[i][k] for i, k in enumerate(ids))
        b = sum(beta[i][k] for i, k in enumerate(ids))
        scales = legal if fmt == 'fp16' else [np.clip(b / a, *domain) if a else domain[0]]
        best = min(best, np.min(a * np.asarray(scales)**2 - 2 * b * np.asarray(scales)))
    return best


@pytest.mark.parametrize('fmt', ['continuous', 'fp16'])
def test_random_exhaustive_combinations_and_independent_scale_oracle(fmt):
    rng = np.random.default_rng(321)
    for _ in range(35):
        alpha = [rng.uniform(.001, 20, size=k) for k in (2, 3, 2)]
        beta = [rng.normal(0, 8, size=len(a)) for a in alpha]
        result = solve_scale_intervals(alpha, beta, 1., (.125, 8.), scale_format=fmt)
        assert result.objective == pytest.approx(oracle(alpha, beta, (.125, 8.), fmt), rel=1e-12, abs=1e-12)
        assert result.interval_count <= 1 + sum(len(a) - 1 for a in alpha)
        assert result.objective == pytest.approx(sum(alpha[i][k] * result.scale**2 - 2 * beta[i][k] * result.scale
                                                    for i, k in enumerate(result.indices)), abs=1e-12)


def test_coordinated_escape_from_single_change_stall():
    a, b = [[1., 25.], [4., 25.]], [[1., 5.], [2., 5.]]
    result = solve_scale_intervals(a, b, .6, (.01, 2.))
    assert result.indices.tolist() == [1, 1]
    assert result.scale == pytest.approx(.2)
    assert result.objective == pytest.approx(-2.)
    for ids in ([0, 0], [0, 1], [1, 0]):
        aa, bb = sum(a[i][k] for i, k in enumerate(ids)), sum(b[i][k] for i, k in enumerate(ids))
        assert -bb * bb / aa >= -1.8


@pytest.mark.parametrize('fmt', ['continuous', 'fp16'])
def test_zero_parallel_ties_and_anchor_scale_only_coverage(fmt):
    zero = solve_scale_intervals([[0., 0.]], [[0., 0.]], 1., (.5, 2.), scale_format=fmt)
    assert (zero.scale, zero.objective, zero.indices.tolist()) == (1., 0., [0])
    result = solve_scale_intervals([[1., 1., 1.], [2.]], [[1., 1., .5], [3.]], 1., (.5, 2.), scale_format=fmt)
    assert result.indices.tolist() == [0, 0]
    assert result.objective <= 3. - 8.
    assert result.objective == pytest.approx(oracle([[1.], [2.]], [[1.], [3.]], (.5, 2.), fmt))


def test_fp16_narrow_intervals_and_exact_endpoints():
    # Crossings closer than an FP16 step must not be merged or rounded across intervals.
    x, y = 1.0002, 1.0003
    alpha = [[3., 2., 1.], [1.]]
    beta = [[0., -x / 2, -(x + y) / 2], [4.]]
    domain = (1., float(np.nextafter(np.float16(1), np.float16(2))))
    result = solve_scale_intervals(alpha, beta, 1., domain, scale_format='fp16')
    assert result.interval_count == 3
    assert result.objective == pytest.approx(oracle(alpha, beta, domain, 'fp16'), abs=1e-14)
    one = solve_scale_intervals([[1., 2.]], [[1., 1.5]], 1., (1., 1.), scale_format='fp16')
    assert one.indices.tolist() == [0]


@pytest.mark.parametrize('alpha,beta,current,domain', [([[]], [[]], 1., (.5, 2.)),
    ([[0.]], [[1.]], 1., (.5, 2.)), ([[-1.]], [[0.]], 1., (.5, 2.)),
    ([[np.nan]], [[0.]], 1., (.5, 2.)), ([[1.]], [[0.]], 1., (0., 2.)),
    ([[1.]], [[0.]], 3., (.5, 2.))])
def test_rejects_invalid_inputs(alpha, beta, current, domain):
    with pytest.raises(ValueError):
        solve_scale_intervals(alpha, beta, current, domain)


def test_fp16_requires_legal_anchor():
    with pytest.raises(ValueError):
        solve_scale_intervals([[1.]], [[1.]], 1.00001, (1., 2.), scale_format='fp16')


@pytest.mark.parametrize('current,domain', [(2**-24, (2**-24, 2**-22)), (65504., (65000., 65504.))])
def test_fp16_subnormal_and_largest_values(current, domain):
    alpha, beta = [[1., 2.]], [[current, current * 1.3]]
    result = solve_scale_intervals(alpha, beta, current, domain, scale_format='fp16')
    assert float(np.float16(result.scale)) == result.scale
    assert domain[0] <= result.scale <= domain[1]
    assert result.objective == pytest.approx(oracle(alpha, beta, domain, 'fp16'), rel=1e-14, abs=1e-30)


def test_near_parallel_lines_keep_distinct_events():
    eps = 2**-42
    alpha = [[1. + 2 * eps, 1. + eps, 1.], [1.]]
    beta = [[eps * 1.25, eps * .75, 0.], [2.]]
    result = solve_scale_intervals(alpha, beta, 1., (.5, 2.))
    assert result.interval_count == 3
    assert result.objective == pytest.approx(oracle(alpha, beta, (.5, 2.), 'continuous'), abs=1e-14)


def test_anchor_and_fixed_assignment_scale_search_are_covered():
    rng = np.random.default_rng(4)
    for _ in range(20):
        alpha = rng.uniform(.01, 5., (4, 3))
        beta = rng.normal(size=(4, 3))
        result = solve_scale_intervals(alpha, beta, 1., (.125, 8.))
        anchor = np.sum(alpha[:, 0] - 2 * beta[:, 0])
        scale_only = oracle(alpha[:, :1], beta[:, :1], (.125, 8.), 'continuous')
        assert result.objective <= anchor + 1e-12
        assert result.objective <= scale_only + 1e-12
