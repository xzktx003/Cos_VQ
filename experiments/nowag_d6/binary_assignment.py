"""Two-choice hard assignments with a training-only straight-through gate."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from experiments.nowag_d6.hard_codebook import HardCodebookWeight, _as_int


class BinaryAssignmentWeight(HardCodebookWeight):
    """Select the original or one alternative code without expanding export bits."""

    def __init__(
        self, logical_state: Mapping[str, Any], alternative_assignments: torch.Tensor
    ) -> None:
        super().__init__(logical_state)
        alternatives = self._tensor(alternative_assignments, "alternative_assignments")
        if (
            alternatives.shape != self.assignments.shape
            or alternatives.dtype != self.assignments.dtype
        ):
            raise ValueError(
                "alternative assignments must match original shape and dtype"
            )
        ids = alternatives.detach().long()
        if (ids < 0).any() or (ids >= self.codebook.shape[0]).any():
            raise ValueError("alternative assignments contain out-of-range code IDs")
        if (ids == self.assignments.to(ids.device).long()).any():
            raise ValueError(
                "each alternative assignment must differ from the original"
            )
        self.register_buffer("alternative_assignments", alternatives.detach().clone())
        self.assignment_logits = nn.Parameter(
            torch.full(
                self.assignments.shape,
                -0.1,
                dtype=torch.float32,
                device=self.codebook.device,
            )
        )

    def _selected_ids(self) -> torch.Tensor:
        self._validate_finite("assignment_logits", self.assignment_logits)
        return torch.where(
            self.assignment_logits > 0,
            self.alternative_assignments.long(),
            self.assignments.long(),
        )

    def forward(self) -> torch.Tensor:
        codebook, row_norm, col_norm = self._effective_serialized_params()
        selected = self._selected_ids()
        gate = self.assignment_logits.sigmoid()
        direction = (
            codebook[self.alternative_assignments.long()]
            - codebook[self.assignments.long()]
        ).detach()
        vectors = codebook[selected] + (gate - gate.detach())[..., None] * direction
        weight = vectors.reshape(self._rows, self._blocks_per_row * self._vector_dim)
        weight = weight[:, : self._columns] * row_norm[:, None] * col_norm[None, :]
        if self.group_scale is not None:
            group_size = _as_int(self._metadata["group_size"], "group_size")
            expanded = self.group_scale.float().repeat_interleave(group_size, dim=1)
            weight = weight * expanded[:, : self._columns]
        return weight

    def export_state(self) -> dict[str, Any]:
        selected = self._selected_ids()
        state = super().export_state()
        state["assignments"] = selected.detach().to("cpu", self.assignments.dtype)
        return state


@torch.no_grad()
def build_alternative_assignments(
    state: Mapping[str, Any],
    dense_weight: torch.Tensor,
    device: str = "cpu",
    neighbors: int = 8,
    chunk_size: int = 16384,
) -> torch.Tensor:
    """Minimize actual dense weight error within each code's local neighbors.

    The neighbor shortlist excludes the original ID. Dense comparison ignores
    padding columns, and only a chunk of vectors and their candidates is resident
    at once. Row/column/group factors weight the candidate distance in the
    original dense space; no division is needed for zero or near-zero factors.
    """
    if not isinstance(neighbors, int) or neighbors <= 0:
        raise ValueError("neighbors must be a positive integer")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    module = HardCodebookWeight(state, tune_codebook=False, tune_norms=False).to(device)
    dense_weight = module._tensor(dense_weight, "dense_weight")
    if tuple(dense_weight.shape) != (module._rows, module._columns):
        raise ValueError("dense weight shape does not match logical weight")
    module._validate_finite("dense_weight", dense_weight)
    codebook, row_norm, col_norm = module._effective_serialized_params()
    count = codebook.shape[0]
    if count < 2:
        raise ValueError("at least two codes are required for alternatives")
    neighbors = min(neighbors, count - 1)
    # Avoid retaining a codebook-sized square distance matrix.
    shortlist = torch.empty((count, neighbors), dtype=torch.long, device=device)
    for start in range(0, count, min(chunk_size, 256)):
        stop = min(start + min(chunk_size, 256), count)
        distances = torch.cdist(codebook[start:stop], codebook).square()
        distances[
            torch.arange(stop - start, device=device),
            torch.arange(start, stop, device=device),
        ] = float("inf")
        shortlist[start:stop] = distances.topk(neighbors, largest=False).indices

    original = module.assignments.reshape(-1).long()
    result = torch.empty(original.shape, dtype=torch.long, device="cpu")
    offsets = torch.arange(module._vector_dim, device=device)
    for start in range(0, original.numel(), chunk_size):
        stop = min(start + chunk_size, original.numel())
        positions = torch.arange(start, stop, device=device)
        rows = positions // module._blocks_per_row
        columns = (positions % module._blocks_per_row)[
            :, None
        ] * module._vector_dim + offsets
        valid = columns < module._columns
        safe_columns = columns.clamp(max=module._columns - 1)
        scale = row_norm[rows, None] * col_norm[safe_columns]
        if module.group_scale is not None:
            size = _as_int(module._metadata["group_size"], "group_size")
            scale = (
                scale * module.group_scale.float()[rows[:, None], safe_columns // size]
            )
        # Index at the dense tensor's device before transferring only this chunk.
        targets = dense_weight[
            rows.to(dense_weight.device)[:, None], safe_columns.to(dense_weight.device)
        ].to(device, torch.float32)
        candidates = shortlist[original[start:stop]]
        errors = (
            (codebook[candidates] * scale[:, None, :] - targets[:, None, :]).square()
            * valid[:, None, :]
        ).sum(-1)
        winners = candidates.gather(1, errors.argmin(1, keepdim=True)).squeeze(1)
        result[start:stop] = winners.cpu()
    return result.reshape(module.assignments.shape).to(module.assignments.dtype)
