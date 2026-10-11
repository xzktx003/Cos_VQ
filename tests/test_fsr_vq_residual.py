from __future__ import annotations

import pytest
import torch

from src.fsr_vq.formats import deployable_payload_nbytes
from src.fsr_vq.candidate_graph import (
    linear_function_quadratic,
    residual_target_distances,
)
from src.fsr_vq.residual import (
    HardLinearVectorState,
    exact_linear_transition_delta,
)


def _make_problem() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    HardLinearVectorState,
]:
    torch.manual_seed(7)
    inputs = torch.randn(11, 6, dtype=torch.float64)
    float_weight = torch.randn(2, 6, dtype=torch.float64)
    codebook = torch.tensor(
        [
            [-1.0, -0.5],
            [-0.2, 0.3],
            [0.4, 0.8],
            [1.1, -0.7],
        ],
        dtype=torch.float64,
    )
    assignments = torch.tensor([[0, 1, 2], [3, 2, 1]], dtype=torch.long)
    state = HardLinearVectorState.from_assignments(
        inputs=inputs,
        float_weight=float_weight,
        codebook=codebook,
        assignments=assignments,
    )
    return inputs, float_weight, codebook, assignments, state


def test_exact_transition_delta_matches_brute_force_loss_change() -> None:
    inputs, float_weight, codebook, _, state = _make_problem()
    output_row = 1
    vector_index = 2
    current_id = int(state.assignments[output_row, vector_index])
    candidate_id = 0
    vector_delta = codebook[candidate_id] - codebook[current_id]
    column_start = vector_index * codebook.shape[1]

    predicted = exact_linear_transition_delta(
        inputs=inputs,
        residual=state.residual,
        output_row=output_row,
        column_start=column_start,
        vector_delta=vector_delta,
    )

    candidate_weight = state.weight.clone()
    candidate_weight[
        output_row, column_start : column_start + codebook.shape[1]
    ] = codebook[candidate_id]
    target = inputs @ float_weight.T
    old_loss = (inputs @ state.weight.T - target).square().sum()
    new_loss = (inputs @ candidate_weight.T - target).square().sum()

    torch.testing.assert_close(predicted, new_loss - old_loss, rtol=1e-12, atol=1e-12)


def test_commit_updates_residual_exactly_and_decreases_loss() -> None:
    inputs, float_weight, codebook, _, state = _make_problem()
    target = inputs @ float_weight.T
    old_loss = state.squared_error()

    best = None
    for output_row in range(state.assignments.shape[0]):
        for vector_index in range(state.assignments.shape[1]):
            current = int(state.assignments[output_row, vector_index])
            for candidate in range(codebook.shape[0]):
                if candidate == current:
                    continue
                proposal = state.propose(output_row, vector_index, candidate)
                if best is None or proposal.predicted_delta < best.predicted_delta:
                    best = proposal

    assert best is not None
    assert best.predicted_delta < 0
    assert state.commit_if_improving(best, epsilon=0.0)

    exact_residual = inputs @ state.weight.T - target
    torch.testing.assert_close(state.residual, exact_residual, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        state.squared_error() - old_loss,
        best.predicted_delta,
        rtol=1e-12,
        atol=1e-12,
    )
    assert state.squared_error() < old_loss


def test_rejected_transition_is_transactional() -> None:
    _, _, codebook, _, state = _make_problem()
    selected = None
    for output_row in range(state.assignments.shape[0]):
        for vector_index in range(state.assignments.shape[1]):
            current = int(state.assignments[output_row, vector_index])
            for candidate in range(codebook.shape[0]):
                if candidate == current:
                    continue
                proposal = state.propose(output_row, vector_index, candidate)
                if proposal.predicted_delta > 0:
                    selected = proposal
                    break
            if selected is not None:
                break
        if selected is not None:
            break

    assert selected is not None
    assignments_before = state.assignments.clone()
    weight_before = state.weight.clone()
    residual_before = state.residual.clone()

    assert not state.commit_if_improving(selected, epsilon=0.0)
    assert torch.equal(state.assignments, assignments_before)
    assert torch.equal(state.weight, weight_before)
    assert torch.equal(state.residual, residual_before)


def test_codebook_value_transport_is_rate_neutral_for_fixed_format() -> None:
    _, _, codebook, assignments, _ = _make_problem()
    before = deployable_payload_nbytes(
        codebook=codebook.to(torch.float16),
        assignments=assignments.to(torch.uint16),
        metadata_tensors=(),
    )
    transported = (codebook + 0.125).to(torch.float16)
    after = deployable_payload_nbytes(
        codebook=transported,
        assignments=assignments.to(torch.uint16),
        metadata_tensors=(),
    )

    assert before == after


def test_residual_target_metric_has_the_same_codeword_ranking_as_exact_delta() -> None:
    inputs, _, codebook, _, state = _make_problem()
    output_row = 0
    vector_index = 1
    current_id = int(state.assignments[output_row, vector_index])
    column_start = vector_index * codebook.shape[1]
    input_slice = inputs[:, column_start : column_start + codebook.shape[1]]
    gram = input_slice.T @ input_slice
    gradient = input_slice.T @ state.residual[:, output_row]
    scale = torch.ones(codebook.shape[1], dtype=codebook.dtype)

    quadratic = linear_function_quadratic(
        current_codeword=codebook[current_id],
        gram=gram,
        gradient=gradient,
        scale=scale,
    )
    target_distances = residual_target_distances(
        codebook=codebook,
        target=quadratic.target,
        metric=quadratic.metric,
    )
    exact_deltas = torch.stack(
        [
            exact_linear_transition_delta(
                inputs=inputs,
                residual=state.residual,
                output_row=output_row,
                column_start=column_start,
                vector_delta=codebook[candidate] - codebook[current_id],
            )
            for candidate in range(codebook.shape[0])
        ]
    )

    centered_distances = target_distances - target_distances[current_id]
    torch.testing.assert_close(centered_distances, exact_deltas, rtol=1e-12, atol=1e-12)
    assert int(target_distances.argmin()) == int(exact_deltas.argmin())


def test_residual_target_remains_finite_for_singular_gram_with_damping() -> None:
    current = torch.tensor([0.25, -0.5], dtype=torch.float64)
    singular_gram = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    gradient = torch.tensor([0.2, -0.1], dtype=torch.float64)
    scale = torch.ones(2, dtype=torch.float64)

    quadratic = linear_function_quadratic(
        current_codeword=current,
        gram=singular_gram,
        gradient=gradient,
        scale=scale,
        damping=1e-3,
    )

    assert torch.isfinite(quadratic.target).all()
    assert torch.isfinite(quadratic.metric).all()
    assert torch.linalg.eigvalsh(quadratic.metric).min() > 0


def test_proposal_is_stale_after_a_different_coordinate_changes_the_residual() -> None:
    _, _, codebook, _, state = _make_problem()
    first = None
    stale = None
    for candidate in range(codebook.shape[0]):
        proposal = state.propose(0, 0, candidate)
        if proposal.predicted_delta < 0:
            first = proposal
            break
    assert first is not None

    for candidate in range(codebook.shape[0]):
        proposal = state.propose(0, 1, candidate)
        if candidate != int(state.assignments[0, 1]):
            stale = proposal
            break
    assert stale is not None

    assert state.commit_if_improving(first, epsilon=0.0)
    with pytest.raises(RuntimeError, match="stale proposal"):
        state.commit_if_improving(stale, epsilon=float("inf"))
