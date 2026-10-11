"""Hard assignment actions with analytic local group-scale compensation.

The compensation is deliberately deployment-first: assignments are replaced by
their hard alternatives before the already-serialized FP16 group scales are
recomputed.  Only row/input groups touched by a switched vector may change.
For each such group we solve the one-dimensional weighted least-squares problem
that keeps the resulting hard weight closest to the incumbent hard weight.
There are no optimizer steps, learning rates, or validation-selected knobs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from experiments.nowag_d6.diagnose_block_vector_gsq import get_linear_modules
from experiments.nowag_d6.probe_scale_conditioned_assignment_curvature import (
    RankedCandidates,
)
from experiments.nowag_d6.run_prefix_vector_gsq import reconstruct_logical_state


class CompensationInfeasibleError(ValueError):
    """The logical state is valid, but no positive multiplicative projection exists."""


@dataclass(frozen=True)
class LocalGroupScaleCompensation:
    group_rows: torch.Tensor
    group_ids: torch.Tensor
    source_scales: torch.Tensor
    compensated_scales: torch.Tensor
    weighted_error_before: float
    weighted_error_after: float
    groups_changed: int
    maximum_relative_scale_change: float

    def summary(self) -> dict[str, float | int]:
        return {
            "affected_groups": int(self.group_rows.numel()),
            "groups_changed_after_fp16_rounding": self.groups_changed,
            "weighted_anchor_error_before": self.weighted_error_before,
            "weighted_anchor_error_after": self.weighted_error_after,
            "weighted_anchor_error_reduction": (
                self.weighted_error_before - self.weighted_error_after
            ),
            "maximum_relative_scale_change": self.maximum_relative_scale_change,
        }


def _state_geometry(state: dict[str, Any]) -> tuple[int, int, int, int, int]:
    try:
        rows, columns = (int(value) for value in state["shape"])
        vector_dim = int(state["vector_dim"])
        group_size = int(state["group_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("logical state lacks valid vector/group geometry") from error
    if min(rows, columns, vector_dim, group_size) <= 0:
        raise ValueError("logical vector/group geometry must be positive")
    padded_columns = math.ceil(columns / vector_dim) * vector_dim
    blocks_per_row = padded_columns // vector_dim
    return rows, columns, vector_dim, group_size, blocks_per_row


def _validate_state_tensors(
    state: dict[str, Any],
    *,
    rows: int,
    columns: int,
    blocks_per_row: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assignments = state.get("assignments")
    codebook = state.get("codebook")
    col_norm = state.get("col_norm")
    group_scale = state.get("group_scale")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (assignments, codebook, col_norm, group_scale)
    ):
        raise ValueError(
            "logical state requires tensor assignments, codebook, col_norm, and group_scale"
        )
    assert isinstance(assignments, torch.Tensor)
    assert isinstance(codebook, torch.Tensor)
    assert isinstance(col_norm, torch.Tensor)
    assert isinstance(group_scale, torch.Tensor)
    groups = math.ceil(columns / group_size)
    if assignments.numel() != rows * blocks_per_row:
        raise ValueError("assignment count does not match logical weight geometry")
    if codebook.ndim != 2:
        raise ValueError("codebook must be a matrix")
    if col_norm.numel() < columns:
        raise ValueError("column normalizer does not cover the logical weight")
    if tuple(group_scale.shape) != (rows, groups):
        raise ValueError("group_scale shape does not match logical weight geometry")
    return assignments.reshape(-1), codebook, col_norm.reshape(-1), group_scale


def compute_local_group_scale_compensation(
    state: dict[str, Any], selection: RankedCandidates
) -> LocalGroupScaleCompensation:
    """Solve the exact hard-state scalar projection for every touched group."""
    rows, columns, vector_dim, group_size, blocks_per_row = _state_geometry(state)
    assignments, codebook, col_norm, group_scale = _validate_state_tensors(
        state,
        rows=rows,
        columns=columns,
        blocks_per_row=blocks_per_row,
        group_size=group_size,
    )
    indices = selection.vector_indices.detach().cpu().long().reshape(-1)
    alternatives = selection.alternative_ids.detach().cpu().long().reshape(-1)
    if indices.numel() == 0 or indices.shape != alternatives.shape:
        raise ValueError("compensation requires matched non-empty hard assignments")
    if int(indices.min()) < 0 or int(indices.max()) >= assignments.numel():
        raise ValueError("selected vector index is outside logical assignments")
    if int(alternatives.min()) < 0 or int(alternatives.max()) >= codebook.shape[0]:
        raise ValueError("alternative assignment is outside the codebook")
    if indices.unique().numel() != indices.numel():
        raise ValueError("selected vector indices must be unique")

    new_assignments = assignments.detach().cpu().long().clone()
    new_assignments[indices] = alternatives
    touched: set[tuple[int, int]] = set()
    for vector_index in indices.tolist():
        row = vector_index // blocks_per_row
        block = vector_index % blocks_per_row
        first_column = block * vector_dim
        last_column = min(first_column + vector_dim, columns)
        for column in range(first_column, last_column):
            touched.add((row, column // group_size))
    ordered = sorted(touched)
    if not ordered:
        raise RuntimeError("hard assignments did not touch any deployable column")

    codebook_float = codebook.detach().cpu().float()
    col_norm_float = col_norm.detach().cpu().float()
    group_scale_float = group_scale.detach().cpu().float()
    rows_out: list[int] = []
    groups_out: list[int] = []
    source_out: list[float] = []
    compensated_out: list[float] = []
    error_before = 0.0
    error_after = 0.0

    for row, group in ordered:
        assignment_start = row * blocks_per_row
        assignment_end = assignment_start + blocks_per_row
        old_normalized = codebook_float[
            assignments[assignment_start:assignment_end].detach().cpu().long()
        ].reshape(-1)[:columns]
        new_normalized = codebook_float[
            new_assignments[assignment_start:assignment_end]
        ].reshape(-1)[:columns]
        column_start = group * group_size
        column_end = min(column_start + group_size, columns)
        old_values = old_normalized[column_start:column_end]
        new_values = new_normalized[column_start:column_end]
        weights = col_norm_float[column_start:column_end].square()
        source_scale = float(group_scale_float[row, group])
        if not math.isfinite(source_scale) or source_scale <= 0:
            raise ValueError("source group scale must be finite positive")
        denominator = float((weights * new_values.square()).sum())
        numerator = float((weights * new_values * old_values).sum())
        if not math.isfinite(denominator) or denominator <= 0:
            raise CompensationInfeasibleError(
                "compensation has a non-positive least-squares denominator"
            )
        if not math.isfinite(numerator) or numerator <= 0:
            raise CompensationInfeasibleError(
                "hard action has no positive multiplicative scale projection"
            )
        projected = source_scale * numerator / denominator
        rounded = float(torch.tensor(projected, dtype=torch.float16))
        if not math.isfinite(rounded) or rounded <= 0:
            raise CompensationInfeasibleError(
                "FP16 compensation scale is not finite positive"
            )
        incumbent = source_scale * old_values
        uncompensated = source_scale * new_values
        compensated = rounded * new_values
        error_before += float((weights * (uncompensated - incumbent).square()).sum())
        error_after += float((weights * (compensated - incumbent).square()).sum())
        rows_out.append(row)
        groups_out.append(group)
        source_out.append(source_scale)
        compensated_out.append(rounded)

    source_tensor = torch.tensor(source_out, dtype=torch.float16)
    compensated_tensor = torch.tensor(compensated_out, dtype=torch.float16)
    relative = (
        (compensated_tensor.float() - source_tensor.float()).abs()
        / source_tensor.float().abs().clamp_min(torch.finfo(torch.float32).tiny)
    )
    groups_changed = int((source_tensor != compensated_tensor).sum())
    if groups_changed == 0:
        raise CompensationInfeasibleError(
            "analytic projection is an FP16 no-op on every affected group"
        )
    return LocalGroupScaleCompensation(
        group_rows=torch.tensor(rows_out, dtype=torch.long),
        group_ids=torch.tensor(groups_out, dtype=torch.long),
        source_scales=source_tensor,
        compensated_scales=compensated_tensor,
        weighted_error_before=error_before,
        weighted_error_after=error_after,
        groups_changed=groups_changed,
        maximum_relative_scale_change=float(relative.max()),
    )


def apply_compensation_to_checkpoint(
    checkpoint: dict[str, Any],
    selections: dict[tuple[int, str], RankedCandidates],
    compensations: dict[tuple[int, str], LocalGroupScaleCompensation],
) -> None:
    if set(selections) != set(compensations):
        raise ValueError("selection and compensation keys must match")
    for key, selection in selections.items():
        layer, name = key
        state = checkpoint["layers"][str(layer)][name]
        assignments = state["assignments"]
        assignments[selection.vector_indices.detach().cpu().long()] = (
            selection.alternative_ids.detach().cpu().to(assignments.dtype)
        )
        compensation = compensations[key]
        scales = state["group_scale"]
        scales[
            compensation.group_rows,
            compensation.group_ids,
        ] = compensation.compensated_scales.to(scales.dtype)


@torch.no_grad()
def audit_fresh_compensated_checkpoint(
    source: dict[str, Any],
    fresh: dict[str, Any],
    model: nn.Module,
    selections: dict[tuple[int, str], RankedCandidates],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Prove that a freshly loaded paired checkpoint is the live deployed state."""
    assignment_changes = 0
    group_scale_changes = 0
    codebooks_exact = True
    normalizers_exact = True
    fixed_metadata_exact = True
    untouched_states_exact = True
    group_scale_dtype_fp16 = True
    assignment_dtype_int32 = True
    reconstruction_max_abs = 0.0
    selected_keys = set(selections)
    if set(source["layers"]) != set(fresh["layers"]):
        raise RuntimeError("fresh checkpoint changed the decoder layer set")
    for layer_key, source_layer in source["layers"].items():
        fresh_layer = fresh["layers"][layer_key]
        if set(source_layer) != set(fresh_layer):
            raise RuntimeError("fresh checkpoint changed the Linear set")
        modules = get_linear_modules(model, int(layer_key))
        for name, source_state in source_layer.items():
            fresh_state = fresh_layer[name]
            key = (int(layer_key), name)
            if set(source_state) != set(fresh_state):
                fixed_metadata_exact = False
                continue
            for metadata_key in ("codebook",):
                codebooks_exact = codebooks_exact and torch.equal(
                    source_state[metadata_key], fresh_state[metadata_key]
                )
            for metadata_key in ("row_norm", "col_norm"):
                normalizers_exact = normalizers_exact and torch.equal(
                    source_state[metadata_key], fresh_state[metadata_key]
                )
            fresh_scale = fresh_state.get("group_scale")
            assignment_dtype_int32 = (
                assignment_dtype_int32
                and fresh_state["assignments"].dtype == torch.int32
            )
            if isinstance(fresh_scale, torch.Tensor):
                group_scale_dtype_fp16 = (
                    group_scale_dtype_fp16 and fresh_scale.dtype == torch.float16
                )
            for state_key, source_value in source_state.items():
                fresh_value = fresh_state[state_key]
                allowed_change = key in selected_keys and state_key in {
                    "assignments",
                    "group_scale",
                }
                if isinstance(source_value, torch.Tensor):
                    equal = torch.equal(source_value, fresh_value)
                else:
                    equal = source_value == fresh_value
                if not allowed_change and not equal:
                    fixed_metadata_exact = False
                if key not in selected_keys and not equal:
                    untouched_states_exact = False
            assignment_changes += int(
                torch.count_nonzero(
                    fresh_state["assignments"].long()
                    != source_state["assignments"].long()
                )
            )
            if isinstance(fresh_scale, torch.Tensor):
                source_scale = source_state["group_scale"]
                group_scale_changes += int(
                    torch.count_nonzero(fresh_scale != source_scale)
                )
            if key in selected_keys:
                reconstructed = reconstruct_logical_state(fresh_state, device)
                deployed = modules[name].weight.detach()
                difference = (
                    reconstructed.to(deployed.dtype).float() - deployed.float()
                ).abs()
                reconstruction_max_abs = max(
                    reconstruction_max_abs, float(difference.max())
                )
    contract = {
        "assignments_changed": assignment_changes > 0,
        "assignment_changes": assignment_changes,
        "assignment_dtype": "int32" if assignment_dtype_int32 else "mixed",
        "group_scales_changed": group_scale_changes > 0,
        "group_scale_changes": group_scale_changes,
        "group_scale_dtype": "float16" if group_scale_dtype_fp16 else "mixed",
        "codebooks_changed": not codebooks_exact,
        "normalizers_changed": not normalizers_exact,
        "fixed_metadata_exact": fixed_metadata_exact,
        "untouched_states_exact": untouched_states_exact,
        "fresh_reconstruction_max_abs": reconstruction_max_abs,
        "fresh_reconstruction_exact": reconstruction_max_abs == 0.0,
        "logical_bitrate_changed": False,
    }
    required = (
        contract["assignments_changed"]
        and contract["assignment_dtype"] == "int32"
        and not contract["codebooks_changed"]
        and not contract["normalizers_changed"]
        and contract["fixed_metadata_exact"]
        and contract["untouched_states_exact"]
        and contract["group_scale_dtype"] == "float16"
        and contract["fresh_reconstruction_exact"]
        and not contract["logical_bitrate_changed"]
    )
    if not required:
        raise RuntimeError(f"fresh paired checkpoint contract failed: {contract}")
    return contract


class HardScaleCompensatedApplicator:
    """Apply and exactly restore sparse hard assignment/scale paired actions."""

    def __init__(
        self,
        model: nn.Module,
        checkpoint: dict[str, Any],
        *,
        layers: list[int],
        device: torch.device,
    ) -> None:
        self.model = model
        self.checkpoint = checkpoint
        self.layers = layers
        self.device = device
        self.active: dict[tuple[int, str], LocalGroupScaleCompensation] = {}
        self._selection_cache: dict[tuple[int, str], RankedCandidates] = {}
        self._compensation_cache: dict[
            tuple[int, str], LocalGroupScaleCompensation
        ] = {}

    def _compensation(
        self, key: tuple[int, str], selection: RankedCandidates
    ) -> LocalGroupScaleCompensation:
        cached_selection = self._selection_cache.get(key)
        if cached_selection is not None:
            if not torch.equal(cached_selection.vector_indices, selection.vector_indices):
                raise ValueError("one Linear cannot reuse a different candidate bundle")
            if not torch.equal(cached_selection.alternative_ids, selection.alternative_ids):
                raise ValueError("one Linear cannot reuse different alternative IDs")
            return self._compensation_cache[key]
        layer, name = key
        state = self.checkpoint["layers"][str(layer)][name]
        compensation = compute_local_group_scale_compensation(state, selection)
        self._selection_cache[key] = selection
        self._compensation_cache[key] = compensation
        return compensation

    @torch.no_grad()
    def _write_groups(
        self,
        key: tuple[int, str],
        compensation: LocalGroupScaleCompensation,
        selection: RankedCandidates | None,
    ) -> None:
        layer, name = key
        state = self.checkpoint["layers"][str(layer)][name]
        rows, columns, vector_dim, group_size, blocks_per_row = _state_geometry(state)
        assignments, codebook, col_norm, group_scale = _validate_state_tensors(
            state,
            rows=rows,
            columns=columns,
            blocks_per_row=blocks_per_row,
            group_size=group_size,
        )
        logical_assignments = assignments.detach().cpu().long().clone()
        scales = compensation.source_scales
        if selection is not None:
            logical_assignments[selection.vector_indices.detach().cpu().long()] = (
                selection.alternative_ids.detach().cpu().long()
            )
            scales = compensation.compensated_scales
        module = get_linear_modules(self.model, layer)[name]
        codebook_device = codebook.to(self.device, dtype=torch.float32)
        col_norm_device = col_norm.to(self.device, dtype=torch.float32)
        row_norm_device = state["row_norm"].to(self.device, dtype=torch.float32)
        for position, (row, group) in enumerate(
            zip(
                compensation.group_rows.tolist(),
                compensation.group_ids.tolist(),
                strict=True,
            )
        ):
            if not 0 <= row < rows:
                raise RuntimeError("cached compensation row is outside the weight")
            assignment_start = row * blocks_per_row
            assignment_end = assignment_start + blocks_per_row
            normalized = codebook_device[
                logical_assignments[assignment_start:assignment_end].to(self.device)
            ].reshape(-1)[:columns]
            column_start = group * group_size
            column_end = min(column_start + group_size, columns)
            values = (
                normalized[column_start:column_end]
                * row_norm_device[row]
                * col_norm_device[column_start:column_end]
                * scales[position].to(self.device, dtype=torch.float32)
            )
            module.weight[row, column_start:column_end] = values.to(module.weight.dtype)

    @torch.no_grad()
    def restore(self) -> None:
        for key, compensation in self.active.items():
            self._write_groups(key, compensation, None)
        self.active.clear()

    @torch.no_grad()
    def apply(
        self, selections: dict[tuple[int, str], RankedCandidates]
    ) -> None:
        self.restore()
        for key, selection in selections.items():
            if key[0] not in self.layers:
                raise ValueError("selection layer is outside the applicator contract")
            compensation = self._compensation(key, selection)
            self._write_groups(key, compensation, selection)
            self.active[key] = compensation

    def compensation_for(
        self, key: tuple[int, str], selection: RankedCandidates
    ) -> LocalGroupScaleCompensation:
        return self._compensation(key, selection)

    def selected_compensations(
        self, selections: dict[tuple[int, str], RankedCandidates]
    ) -> dict[tuple[int, str], LocalGroupScaleCompensation]:
        return {key: self._compensation(key, selection) for key, selection in selections.items()}
