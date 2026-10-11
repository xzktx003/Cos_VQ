"""Exact hard vector-transition updates for fixed-input linear layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _validate_transition_inputs(
    *,
    inputs: torch.Tensor,
    residual: torch.Tensor,
    output_row: int,
    column_start: int,
    vector_delta: torch.Tensor,
) -> None:
    if inputs.ndim != 2:
        raise ValueError(f"inputs must be rank 2, got shape {tuple(inputs.shape)}")
    if residual.ndim != 2:
        raise ValueError(f"residual must be rank 2, got shape {tuple(residual.shape)}")
    if vector_delta.ndim != 1:
        raise ValueError(
            f"vector_delta must be rank 1, got shape {tuple(vector_delta.shape)}"
        )
    if inputs.shape[0] != residual.shape[0]:
        raise ValueError("inputs and residual must have the same number of rows")
    if not 0 <= output_row < residual.shape[1]:
        raise IndexError(f"output_row {output_row} is out of range")
    column_end = column_start + vector_delta.numel()
    if column_start < 0 or column_end > inputs.shape[1]:
        raise IndexError(
            f"columns [{column_start}, {column_end}) exceed input width {inputs.shape[1]}"
        )
    if inputs.device != residual.device or inputs.device != vector_delta.device:
        raise ValueError("inputs, residual, and vector_delta must share a device")


def exact_linear_transition_delta(
    *,
    inputs: torch.Tensor,
    residual: torch.Tensor,
    output_row: int,
    column_start: int,
    vector_delta: torch.Tensor,
) -> torch.Tensor:
    """Return the exact change in summed squared output error.

    For a fixed-input linear layer, changing one output-row weight vector by
    ``vector_delta`` changes that residual column by ``X_slice @ delta``.
    Expanding ``||r + u||^2 - ||r||^2`` gives the exact expression
    ``2 <r, u> + ||u||^2``; no Taylor approximation is used here.
    """

    _validate_transition_inputs(
        inputs=inputs,
        residual=residual,
        output_row=output_row,
        column_start=column_start,
        vector_delta=vector_delta,
    )
    column_end = column_start + vector_delta.numel()
    output_delta = inputs[:, column_start:column_end] @ vector_delta
    row_residual = residual[:, output_row]
    return 2 * torch.dot(row_residual, output_delta) + torch.dot(
        output_delta, output_delta
    )


@dataclass(frozen=True)
class LinearTransitionProposal:
    """A state-bound hard transition proposal for one vector coordinate."""

    output_row: int
    vector_index: int
    current_code_id: int
    candidate_code_id: int
    predicted_delta: torch.Tensor
    state_version: int


class HardLinearVectorState:
    """Hard deployable vector state plus its exact fixed-input residual.

    The class intentionally owns no optimizer or soft assignments.  A rejected
    proposal performs no writes, while an accepted proposal updates the hard
    ID, reconstructed weight slice, and residual together.
    """

    def __init__(
        self,
        *,
        inputs: torch.Tensor,
        float_weight: torch.Tensor,
        codebook: torch.Tensor,
        assignments: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        self.inputs = inputs
        self.float_weight = float_weight
        self.codebook = codebook
        self.assignments = assignments
        self.weight = weight
        self.residual = residual
        self.version = 0

    @classmethod
    def from_assignments(
        cls,
        *,
        inputs: torch.Tensor,
        float_weight: torch.Tensor,
        codebook: torch.Tensor,
        assignments: torch.Tensor,
    ) -> "HardLinearVectorState":
        if inputs.ndim != 2 or float_weight.ndim != 2:
            raise ValueError("inputs and float_weight must be rank 2")
        if codebook.ndim != 2 or assignments.ndim != 2:
            raise ValueError("codebook and assignments must be rank 2")
        if inputs.shape[1] != float_weight.shape[1]:
            raise ValueError("input width must match float_weight input features")
        if assignments.shape[0] != float_weight.shape[0]:
            raise ValueError("assignment rows must match output features")
        vector_dim = codebook.shape[1]
        if assignments.shape[1] * vector_dim != float_weight.shape[1]:
            raise ValueError(
                "this exact-state primitive requires an unpadded input width divisible "
                "by the vector dimension"
            )
        if assignments.numel() and (
            int(assignments.min()) < 0 or int(assignments.max()) >= codebook.shape[0]
        ):
            raise ValueError("assignments contain code IDs outside the codebook")
        devices = {inputs.device, float_weight.device, codebook.device, assignments.device}
        if len(devices) != 1:
            raise ValueError("all state tensors must share a device")
        if inputs.dtype != float_weight.dtype or inputs.dtype != codebook.dtype:
            raise ValueError("inputs, float_weight, and codebook must share a dtype")

        vectors = codebook[assignments.long()]
        weight = vectors.reshape_as(float_weight).clone()
        target = inputs @ float_weight.T
        residual = inputs @ weight.T - target
        return cls(
            inputs=inputs,
            float_weight=float_weight,
            codebook=codebook,
            assignments=assignments.long().clone(),
            weight=weight,
            residual=residual,
        )

    @property
    def vector_dim(self) -> int:
        return int(self.codebook.shape[1])

    def squared_error(self) -> torch.Tensor:
        return self.residual.square().sum()

    def propose(
        self,
        output_row: int,
        vector_index: int,
        candidate_code_id: int,
    ) -> LinearTransitionProposal:
        if not 0 <= output_row < self.assignments.shape[0]:
            raise IndexError(f"output_row {output_row} is out of range")
        if not 0 <= vector_index < self.assignments.shape[1]:
            raise IndexError(f"vector_index {vector_index} is out of range")
        if not 0 <= candidate_code_id < self.codebook.shape[0]:
            raise IndexError(f"candidate code {candidate_code_id} is out of range")
        current_code_id = int(self.assignments[output_row, vector_index])
        vector_delta = (
            self.codebook[candidate_code_id] - self.codebook[current_code_id]
        )
        predicted_delta = exact_linear_transition_delta(
            inputs=self.inputs,
            residual=self.residual,
            output_row=output_row,
            column_start=vector_index * self.vector_dim,
            vector_delta=vector_delta,
        )
        return LinearTransitionProposal(
            output_row=output_row,
            vector_index=vector_index,
            current_code_id=current_code_id,
            candidate_code_id=int(candidate_code_id),
            predicted_delta=predicted_delta,
            state_version=self.version,
        )

    @torch.no_grad()
    def commit_if_improving(
        self,
        proposal: LinearTransitionProposal,
        *,
        epsilon: float,
    ) -> bool:
        if proposal.state_version != self.version:
            raise RuntimeError(
                "stale proposal: the Linear residual changed after scoring"
            )
        current_code_id = int(
            self.assignments[proposal.output_row, proposal.vector_index]
        )
        if current_code_id != proposal.current_code_id:
            raise RuntimeError(
                "stale proposal: the vector assignment changed after scoring"
            )
        if proposal.candidate_code_id == proposal.current_code_id:
            return False
        if float(proposal.predicted_delta) >= -float(epsilon):
            return False

        column_start = proposal.vector_index * self.vector_dim
        column_end = column_start + self.vector_dim
        vector_delta = (
            self.codebook[proposal.candidate_code_id]
            - self.codebook[proposal.current_code_id]
        )
        output_delta = self.inputs[:, column_start:column_end] @ vector_delta

        self.assignments[
            proposal.output_row, proposal.vector_index
        ] = proposal.candidate_code_id
        self.weight[
            proposal.output_row, column_start:column_end
        ] = self.codebook[proposal.candidate_code_id]
        self.residual[:, proposal.output_row].add_(output_delta)
        self.version += 1
        return True
