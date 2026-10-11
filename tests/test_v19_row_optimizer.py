import torch
import pytest

from experiments.nowag_d6.v19_row_optimizer import optimize_rows
from experiments.nowag_d6.logical_state import reconstruct_logical_state


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_fp16_neighbors_bit_patterns_without_half_nextafter(device, monkeypatch):
    from experiments.nowag_d6.v19_row_optimizer import _nearest_scales

    def unsupported(*args, **kwargs):
        raise RuntimeError('nextafter_cuda not implemented for Half')

    monkeypatch.setattr(torch, 'nextafter', unsupported)
    targets = torch.tensor([0., 2**-24, 1.5*2**-24, 2**-14 - 2**-25,
                            1., 1.0002, 65504., 1e9], dtype=torch.float64, device=device)
    actual = _nearest_scales(targets).cpu()
    expected = torch.tensor([[2**-24, 2**-24], [2**-24, 2**-24],
                             [2**-24, 2*2**-24], [2**-14-2**-24, 2**-14],
                             [1., 1.], [1., 1.+2**-10],
                             [65504., 65504.], [65504., 65504.]], dtype=torch.float64)
    assert torch.equal(actual, expected)


def make_state(rows=3, columns=7):
    torch.manual_seed(8)
    return {'shape': torch.tensor([rows, columns]), 'vector_dim': torch.tensor(3),
            'assignment_bits': torch.tensor(2), 'assignments': torch.randint(4, (rows * ((columns+2)//3),)),
            'codebook': torch.randn(4, 3).half(), 'row_norm': torch.tensor([.3, -.4, 0.])[:rows].half(),
            'col_norm': torch.rand(columns).half()}


@pytest.mark.parametrize('mode', ['interval', 'scale_only'])
def test_deployed_true_objective_nonincrease_budget_and_tail(mode):
    state = make_state()
    reference = torch.randn(3, 7).bfloat16()
    x = torch.randn(7, 9).double()
    h = x @ x.T
    old = reconstruct_logical_state(state, torch.device('cpu')).bfloat16().double()
    result, stats = optimize_rows(state, reference, h, active_positions=2, row_batch_size=2, mode=mode)
    new = reconstruct_logical_state(result, torch.device('cpu')).bfloat16().double()
    old_loss = torch.einsum('ri,ij,rj->r', old-reference.double(), h, old-reference.double())
    new_loss = torch.einsum('ri,ij,rj->r', new-reference.double(), h, new-reference.double())
    assert torch.all(new_loss <= old_loss + 1e-10)
    assert stats['objective_after'] <= stats['objective_before'] + 1e-10
    assert set(result) == set(state)
    for key in state:
        assert result[key].shape == state[key].shape and result[key].dtype == state[key].dtype
        if key not in ('assignments', 'row_norm'):
            assert torch.equal(result[key], state[key])
    assert result['row_norm'][2] == 0
    assert torch.signbit(result['row_norm']).equal(torch.signbit(state['row_norm']))
    if mode == 'scale_only':
        assert torch.equal(result['assignments'], state['assignments'])


def test_scale_only_recovers_scaled_reference_and_does_not_mutate_input():
    state = make_state(rows=1, columns=6)
    saved = {k: v.clone() for k, v in state.items()}
    target = {k: v.clone() for k, v in state.items()}
    target['row_norm'].fill_(.75)
    reference = reconstruct_logical_state(target, torch.device('cpu'))
    result, stats = optimize_rows(state, reference, torch.eye(6).double(), mode='scale_only', deployment_dtype=torch.float32)
    assert stats['objective_after'] < stats['objective_before']
    for key in state:
        assert torch.equal(state[key], saved[key])
    assert result['row_norm'].item() == .75


def test_assignment_and_scale_update_and_majorizer_bound():
    from experiments.nowag_d6.v19_row_optimizer import _majorizer
    state = {'shape': torch.tensor([1, 2]), 'vector_dim': torch.tensor(1),
             'assignment_bits': torch.tensor(2), 'assignments': torch.tensor([0, 1]),
             'codebook': torch.tensor([[1.], [2.], [5.]]).half(),
             'row_norm': torch.tensor([.6]).half(), 'col_norm': torch.ones(2).half()}
    updated, stats = optimize_rows(state, torch.ones(1, 2), torch.eye(2).double(),
                                  active_positions=2, candidates_per_ranking=3)
    assert updated['assignments'].tolist() == [1, 1]
    assert stats['changed_assignments'] == 1
    assert stats['objective_after'] == 0
    torch.manual_seed(42)
    x = torch.randn(7, 9).double()
    h = x @ x.T
    d, _, _ = _majorizer(h, 3, 1e-6)
    dense_d = torch.block_diag(*d)[:7, :7]
    assert torch.linalg.eigvalsh(dense_d - h).min() >= -1e-12


def test_rho_majorizer_and_batch_size_equivalence():
    state = make_state()
    reference = torch.randn(3, 7)
    h = torch.full((7, 7), .1).double() + torch.eye(7).double()
    first, a = optimize_rows(state, reference, h, row_batch_size=1)
    second, b = optimize_rows(state, reference, h, row_batch_size=3)
    assert torch.equal(first['assignments'], second['assignments'])
    assert torch.equal(first['row_norm'], second['row_norm'])
    assert a['rho_min'] > 0 and a['rho_max'] >= a['rho_min']
    assert a['objective_after'] == pytest.approx(b['objective_after'])


def test_ideal_surrogate_is_tangent_and_anchor_is_feasible(monkeypatch):
    import experiments.nowag_d6.v19_row_optimizer as optimizer
    from experiments.nowag_d6.scale_interval_solver import solve_scale_intervals
    state = make_state(rows=1, columns=7)
    reference = torch.randn(1, 7).double()
    x = torch.randn(7, 9).double()
    h = x @ x.T
    d, _, _ = optimizer._majorizer(h, 3, 1e-6)
    ids = state['assignments'].long()
    col = torch.nn.functional.pad(state['col_norm'].double(), (0, 2)).reshape(3, 3)
    unit = state['codebook'][ids].double() * col
    scale = state['row_norm'][0].double()
    z = unit * scale
    b = torch.nn.functional.pad((z.reshape(-1)[:7] - reference[0]) @ h, (0, 2)).reshape(3, 3)
    dt, alpha, beta = optimizer._surrogate_coefficients(unit, scale, b, d)
    t = torch.linalg.solve(d, dt[..., None]).squeeze(-1)
    # Q(z)-Q(z0) has gradient 2b at the anchor, before deployment rounding.
    torch.testing.assert_close(torch.einsum('bij,bj->bi', d, z-t), b, atol=1e-12, rtol=1e-12)
    z_variable = z.detach().requires_grad_()
    q = torch.einsum('bi,bij,bj->', z_variable-t, d, z_variable-t)
    q.backward()
    torch.testing.assert_close(z_variable.grad, 2*b, atol=1e-12, rtol=1e-12)
    anchor_score = float((alpha*scale.square()-2*beta*scale).sum())
    seen = []
    original_coefficients = optimizer._surrogate_coefficients

    def checked_coefficients(units, current, ideal_gradient, majorizer):
        torch.testing.assert_close(ideal_gradient, b, atol=1e-12, rtol=1e-12)
        return original_coefficients(units, current, ideal_gradient, majorizer)

    def checked_solve(aa, bb, current, domain, **kwargs):
        result = solve_scale_intervals(aa, bb, current, domain, **kwargs)
        assert result.objective <= anchor_score + 1e-10
        seen.append(result)
        return result

    monkeypatch.setattr(optimizer, 'solve_scale_intervals', checked_solve)
    monkeypatch.setattr(optimizer, '_surrogate_coefficients', checked_coefficients)
    optimizer.optimize_rows(state, reference, h, active_positions=2)
    assert len(seen) == 1


def test_refit_and_deployment_acceptance_use_batch_gemm():
    from torch.utils._python_dispatch import TorchDispatchMode

    class CountHessianProducts(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.row_counts = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.mm.default and tuple(args[1].shape) == (7, 7):
                self.row_counts.append(args[0].shape[0])
            return func(*args, **(kwargs or {}))

    tracker = CountHessianProducts()
    with tracker:
        optimize_rows(make_state(), torch.ones(3, 7), torch.eye(7).double(), row_batch_size=3)
    # Three initial batched products, then the two nonzero rows' refits and ten
    # candidate deployments. No per-row vector @ full Hessian is permitted.
    assert tracker.row_counts == [3, 3, 3, 2, 10]


@pytest.mark.parametrize('bad', ['group', 'residual', 'nan', 'asymmetric', 'batch'])
def test_rejects_unsupported_or_unsafe_inputs(bad):
    state = make_state()
    h = torch.eye(7).double()
    kwargs = {}
    if bad == 'group':
        state['group_scale'] = torch.ones(3, 1).half()
    elif bad == 'residual':
        state['residual_A'] = torch.ones(1, 7).half()
    elif bad == 'nan':
        h[0, 0] = float('nan')
    elif bad == 'asymmetric':
        h[0, 1] = .1
    else:
        kwargs['row_batch_size'] = 0
    with pytest.raises(ValueError):
        optimize_rows(state, torch.ones(3, 7), h, **kwargs)
