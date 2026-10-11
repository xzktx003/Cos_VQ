"""Transactional state handling for nonlinear FSR-VQ block verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .residual import HardLinearVectorState


@dataclass(frozen=True)
class _LinearSnapshot:
    assignments: torch.Tensor
    codebook: torch.Tensor
    weight: torch.Tensor
    residual: torch.Tensor
    version: int


class HardBlockTransaction:
    """Snapshot several hard Linear states and roll them back atomically.

    The transaction rolls back by default.  ``commit()`` keeps mutations only
    when the context exits normally; an exception always restores the source
    state.  Tensor storage owned by each state is preserved through ``copy_``
    so external module bindings do not become stale.
    """

    def __init__(self, states: Mapping[str, HardLinearVectorState]) -> None:
        if not states:
            raise ValueError("a block transaction requires at least one Linear state")
        self._states = dict(states)
        self._snapshots: dict[str, _LinearSnapshot] = {}
        self._committed = False
        self._entered = False

    def __enter__(self) -> "HardBlockTransaction":
        if self._entered:
            raise RuntimeError("a block transaction cannot be entered twice")
        self._entered = True
        self._snapshots = {
            name: _LinearSnapshot(
                assignments=state.assignments.clone(),
                codebook=state.codebook.clone(),
                weight=state.weight.clone(),
                residual=state.residual.clone(),
                version=state.version,
            )
            for name, state in self._states.items()
        }
        return self

    def commit(self) -> None:
        if not self._entered:
            raise RuntimeError("cannot commit a transaction before entering it")
        self._committed = True

    @torch.no_grad()
    def rollback(self) -> None:
        if not self._entered:
            raise RuntimeError("cannot roll back a transaction before entering it")
        for name, state in self._states.items():
            snapshot = self._snapshots[name]
            state.assignments.copy_(snapshot.assignments)
            state.codebook.copy_(snapshot.codebook)
            state.weight.copy_(snapshot.weight)
            state.residual.copy_(snapshot.residual)
            state.version = snapshot.version
        self._committed = False

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None or not self._committed:
            self.rollback()
        return False
