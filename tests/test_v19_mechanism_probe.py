"""Independent finite-space checks for the read-only V19 mechanism probe."""

import copy
import itertools
import json

import numpy as np
import pytest
import torch

from experiments.nowag_d6.logical_state import reconstruct_logical_state
from experiments.nowag_d6.v19_mechanism_probe import analyze_linear


def state_for(columns=4, rows=1, dim=2, group_size=4):
    state = {
        "shape": (rows, columns), "vector_dim": dim, "assignment_bits": 2,
        "assignments": torch.zeros((rows, (columns + dim - 1) // dim), dtype=torch.int16),
        "codebook": torch.tensor([[1., 0.], [0., 2.], [5., 0.], [0., 5.]]).half()[:, :dim],
        "row_norm": torch.ones(rows).half(), "col_norm": torch.ones(columns).half(),
    }
    if group_size is not None:
        state.update(group_size=group_size,
                     group_scale=torch.ones(rows, (columns + group_size - 1) // group_size).half())
    return state


def ideal_decode(state):
    rows, columns = state["shape"]
    value = state["codebook"].double()[state["assignments"].long()].reshape(rows, -1)[:, :columns]
    value = value * state["row_norm"].double()[:, None] * state["col_norm"].double()[None, :]
    if "group_scale" in state:
        value *= state["group_scale"].double().repeat_interleave(state["group_size"], 1)[:, :columns]
    return value


@pytest.mark.parametrize('group_size', [None, 8])
def test_flat_d6_assignments_match_matrix_probe_with_per_row_padding(group_size):
    # Production checkpoints store flat int32 IDs. Padding belongs to each row:
    # two 13-column rows need 2*ceil(13/6)=6 IDs, not ceil(26/6)=5.
    matrix = state_for(columns=13, rows=2, group_size=group_size)
    matrix['vector_dim'] = 6
    matrix['codebook'] = matrix['codebook'].repeat(1, 3)
    matrix['assignments'] = torch.tensor([[0, 1, 2], [3, 0, 1]], dtype=torch.int32)
    flat = copy.deepcopy(matrix)
    flat['assignments'] = flat['assignments'].reshape(-1)
    original = flat['assignments'].clone()
    torch.manual_seed(44)
    reference = torch.randn(2, 13).double()
    h = torch.eye(13).double()
    expected = analyze_linear(matrix, reference, h, max_rows=2, active_positions=2)
    actual = analyze_linear(flat, reference, h, max_rows=2, active_positions=2)
    def untimed(value):
        if isinstance(value, dict):
            return {k: untimed(v) for k, v in value.items() if not k.endswith('seconds')}
        if isinstance(value, list):
            return [untimed(v) for v in value]
        return value
    assert untimed(actual) == untimed(expected)
    assert flat['assignments'].shape == (6,)
    assert torch.equal(flat['assignments'], original)
    for probe in actual['rows']:
        assert_objectives(matrix, reference, h, probe)
    malformed = dict(flat, assignments=flat['assignments'][:-1])
    with pytest.raises(ValueError, match='Assignment geometry'):
        analyze_linear(malformed, reference, h)


def apply_endpoint(state, probe, method):
    result = copy.deepcopy(state)
    row = probe["row"]
    for position, code in zip(probe["active_code_positions"], method["active_code_ids"]):
        result["assignments"][row, position] = code
    if "group_scale" in state:
        result["group_scale"][row, probe["group_index"]] = method["scale"]
    else:
        result["row_norm"][row] = method["scale"] * probe["scale_sign"]
    return result


def assert_objectives(state, reference, hessian, probe):
    row = probe["row"]
    for method in probe["methods"].values():
        changed = apply_endpoint(state, probe, method)
        error = ideal_decode(changed)[row] - reference[row].double()
        assert method["true_objective"] == pytest.approx(float(error @ hessian.double() @ error), abs=1e-9)
        deployed = reconstruct_logical_state(changed, torch.device("cpu")).bfloat16().double()
        error = deployed[row] - reference[row].double()
        assert method["deployed_objective"] == pytest.approx(float(error @ hessian.double() @ error), abs=1e-9)


def test_joint_change_escapes_scale_refitted_single_coordinate_stall():
    state = state_for()
    state["assignments"][0] = torch.tensor([0, 1])
    state["group_scale"].fill_(0.6)
    reference = torch.tensor([[1., 0., 0., 1.]], dtype=torch.float64)
    result = analyze_linear(state, reference, torch.eye(4), active_positions=2)
    probe = result["rows"][0]
    assert probe["synergy"]["true"]["single_stalled"]
    assert probe["synergy"]["true"]["joint_improves"]
    assert probe["synergy"]["surrogate"]["single_stalled"]
    assert probe["synergy"]["surrogate"]["joint_improves"]
    assert probe["methods"]["oracle_true"]["true_objective"] < 1e-6
    assert probe["methods"]["single_refit"]["true_objective"] > 0.19
    assert probe["surrogate_oracle_gap"] == pytest.approx(0., abs=1e-10)
    assert_objectives(state, reference, torch.eye(4), probe)


@pytest.mark.parametrize("columns,group_size", [(4, 4), (4, 2), (4, None), (6, 6)])
def test_complete_fp16_oracle_matches_independent_dense_enumeration(columns, group_size):
    state = state_for(columns=columns, group_size=group_size)
    state["assignments"][0, 1] = 1
    rng = np.random.default_rng(31)
    reference = torch.from_numpy(rng.normal(size=(1, columns)))
    matrix = torch.from_numpy(rng.normal(size=(columns, columns)))
    hessian = matrix @ matrix.T
    probe = analyze_linear(state, reference, hessian, active_positions=3)["rows"][0]
    scales = np.arange(1, 0x7c00, dtype=np.uint16).view(np.float16).astype(np.float64)
    best = np.inf
    for codes in itertools.product(*probe["candidate_pools"]):
        endpoint = {"active_code_ids": codes, "scale": 1.}
        unit = ideal_decode(apply_endpoint(state, probe, endpoint))[0].numpy()
        values = np.broadcast_to(unit, (len(scales), columns)).copy()
        start, end = probe["column_range"]
        values[:, start:end] *= scales[:, None]
        errors = values - reference[0].numpy()
        best = min(best, np.einsum("ni,ij,nj->n", errors, hessian.numpy(), errors).min())
    assert probe["methods"]["oracle_true"]["true_objective"] == pytest.approx(best, abs=1e-10)
    assert probe["majorizer"]["gap_min_eigenvalue"] >= -1e-10
    assert probe["surrogate_oracle_gap"] == pytest.approx(0., abs=1e-10)
    assert_objectives(state, reference, hessian, probe)


def test_group_objective_retains_outside_residual_linear_term():
    state = state_for(group_size=2)
    state["codebook"] = torch.ones(1, 2).half()
    reference = torch.tensor([[3., 3., 0., 0.]])
    hessian = torch.eye(4, dtype=torch.float64)
    hessian[0, 2] = hessian[2, 0] = .8
    hessian[1, 3] = hessian[3, 1] = .8
    probe = analyze_linear(state, reference, hessian)["rows"][0]
    assert probe["group_index"] == 0
    assert probe["methods"]["scale_only"]["scale"] == pytest.approx(2.2, abs=.002)
    assert_objectives(state, reference, hessian, probe)


def test_cross_boundary_code_frozen_and_partial_tail_in_scale_fit():
    state = state_for(columns=5, dim=2, group_size=3)
    state["codebook"] = torch.tensor([[1., 2.], [3., 4.]]).half()
    target = copy.deepcopy(state)
    target["group_scale"][0, 1] = 2
    reference = ideal_decode(target)
    probe = analyze_linear(state, reference, torch.eye(5))["rows"][0]
    assert probe["group_index"] == 1
    assert 1 in probe["frozen_cross_boundary_code_positions"]
    assert 1 not in probe["active_code_positions"]
    assert probe["methods"]["scale_only"]["scale"] == 2.
    assert_objectives(state, reference, torch.eye(5), probe)


def test_row_scale_specialization_preserves_signed_scale_and_full_target():
    state = state_for(columns=5, group_size=None)
    state["row_norm"].fill_(-.6)
    state["col_norm"] = torch.tensor([.5, 1., 2., .75, 1.5]).half()
    reference = ideal_decode(state) * 1.4
    probe = analyze_linear(state, reference, torch.eye(5))["rows"][0]
    assert probe["scale_sign"] == -1
    assert probe["column_range"] == [0, 5]
    assert_objectives(state, reference, torch.eye(5), probe)


@pytest.mark.parametrize("group_size", [1, None])
def test_no_active_positions_keeps_all_supports_in_scale_fit(group_size):
    state = state_for(group_size=group_size)
    reference = ideal_decode(state) * 1.5
    probe = analyze_linear(state, reference, torch.eye(4), active_positions=0)["rows"][0]
    assert probe["candidate_pools"] == []
    assert probe["enumerated_combinations"] == 1
    assert probe["methods"]["oracle_true"]["scale"] == 1.5
    assert_objectives(state, reference, torch.eye(4), probe)


def test_zero_row_scale_is_explicitly_skipped_without_inventing_a_scale():
    state = state_for(group_size=None)
    state["row_norm"].zero_()
    result = analyze_linear(state, torch.ones(1, 4), torch.eye(4))
    assert result["rows"][0]["status"] == "skipped_zero_row_scale"
    assert float(state["row_norm"][0]) == 0


def test_d6_g160_correlated_hessian_and_nonuniform_norms():
    generator = torch.Generator().manual_seed(94)
    columns, dim = 326, 6
    state = {
        "shape": (1, columns), "vector_dim": dim, "assignment_bits": 5,
        "assignments": torch.randint(0, 32, (1, 55), generator=generator).short(),
        "codebook": torch.randn(32, dim, generator=generator).half(),
        "row_norm": torch.tensor([.7]).half(),
        "col_norm": torch.linspace(.3, 1.2, columns).half(),
        "group_size": 160, "group_scale": torch.ones(1, 3).half(),
    }
    target = copy.deepcopy(state)
    target["group_scale"][0, 1] = 1.3
    reference = ideal_decode(target)
    x = torch.randn(columns, 64, generator=generator, dtype=torch.float64)
    hessian = x @ x.T / 64
    probe = analyze_linear(state, reference, hessian)["rows"][0]
    assert probe["group_index"] == 1
    assert probe["frozen_cross_boundary_code_positions"] == [26, 53]
    assert not {26, 53}.intersection(probe["active_code_positions"])
    assert probe["surrogate_oracle_gap"] == pytest.approx(0., abs=1e-9)
    assert_objectives(state, reference, hessian, probe)


def test_read_only_deterministic_compact_json_and_bounded_space():
    state = state_for(rows=8)
    before = copy.deepcopy(state)
    reference = torch.ones(8, 4)
    first = analyze_linear(state, reference, torch.eye(4), max_rows=3, seed=71)
    second = analyze_linear(state, reference, torch.eye(4), max_rows=3, seed=71)
    assert first["selected_rows"] == second["selected_rows"]
    for a, b in zip(first["rows"], second["rows"]):
        assert a["candidate_pools"] == b["candidate_pools"]
        assert a["methods"]["oracle_true"]["true_objective"] == b["methods"]["oracle_true"]["true_objective"]
        assert a["enumerated_combinations"] <= 125
    json.dumps(first, allow_nan=False)
    for key in state:
        assert torch.equal(state[key], before[key]) if isinstance(state[key], torch.Tensor) else state[key] == before[key]


@pytest.mark.parametrize("target", [2**-25, 1.0006, 70000.])
def test_fp16_scale_boundaries_and_neighbours(target):
    state = state_for(columns=2, group_size=2)
    state["codebook"] = torch.ones(1, 2).half()
    probe = analyze_linear(state, torch.full((1, 2), target, dtype=torch.float64), torch.eye(2))["rows"][0]
    scales = np.arange(1, 0x7c00, dtype=np.uint16).view(np.float16).astype(np.float64)
    expected = scales[np.argmin((scales - target)**2)]
    assert probe["methods"]["oracle_true"]["scale"] == expected


@pytest.mark.parametrize("kwargs", [{"active_positions": 4}, {"candidates_per_ranking": 3}, {"max_rows": 0}])
def test_reject_unbounded_or_invalid_probe_configuration(kwargs):
    with pytest.raises(ValueError):
        analyze_linear(state_for(), torch.ones(1, 4), torch.eye(4), **kwargs)
