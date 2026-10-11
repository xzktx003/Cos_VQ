from __future__ import annotations

import pytest
import torch

from src.fsr_vq.block import HardBlockTransaction
from src.fsr_vq.residual import HardLinearVectorState


def _state(seed: int) -> HardLinearVectorState:
    torch.manual_seed(seed)
    inputs = torch.randn(9, 4, dtype=torch.float64)
    float_weight = torch.randn(2, 4, dtype=torch.float64)
    codebook = torch.tensor(
        [[-1.0, -0.5], [-0.1, 0.2], [0.4, 0.9], [1.2, -0.8]],
        dtype=torch.float64,
    )
    assignments = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    return HardLinearVectorState.from_assignments(
        inputs=inputs,
        float_weight=float_weight,
        codebook=codebook,
        assignments=assignments,
    )


def _mutate(state: HardLinearVectorState) -> None:
    state.assignments[0, 0] = 3
    state.codebook[0, 0] += 2.0
    state.weight.add_(1.0)
    state.residual.mul_(2.0)
    state.version += 1


def test_uncommitted_block_transaction_restores_every_linear_state() -> None:
    states = {"q_proj": _state(1), "down_proj": _state(2)}
    before = {
        name: (
            state.assignments.clone(),
            state.codebook.clone(),
            state.weight.clone(),
            state.residual.clone(),
            state.version,
        )
        for name, state in states.items()
    }

    with HardBlockTransaction(states):
        for state in states.values():
            _mutate(state)

    for name, state in states.items():
        assignments, codebook, weight, residual, version = before[name]
        assert torch.equal(state.assignments, assignments)
        assert torch.equal(state.codebook, codebook)
        assert torch.equal(state.weight, weight)
        assert torch.equal(state.residual, residual)
        assert state.version == version


def test_exception_restores_every_linear_state() -> None:
    states = {"q_proj": _state(3), "down_proj": _state(4)}
    before = {name: state.weight.clone() for name, state in states.items()}

    with pytest.raises(ValueError, match="candidate failed"):
        with HardBlockTransaction(states):
            for state in states.values():
                _mutate(state)
            raise ValueError("candidate failed")

    for name, state in states.items():
        assert torch.equal(state.weight, before[name])


def test_explicit_commit_keeps_all_mutations() -> None:
    states = {"q_proj": _state(5), "down_proj": _state(6)}

    with HardBlockTransaction(states) as transaction:
        for state in states.values():
            _mutate(state)
        transaction.commit()

    for state in states.values():
        assert int(state.assignments[0, 0]) == 3
        assert state.version == 1
