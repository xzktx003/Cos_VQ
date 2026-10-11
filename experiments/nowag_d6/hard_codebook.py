"""Trainable hard-codebook view for Vector-GSQ logical checkpoints."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must contain one integer")
        return int(value.detach().cpu().reshape(()))
    return int(value)


def _shape_tuple(value: Any) -> tuple[int, int]:
    if isinstance(value, torch.Tensor):
        shape = tuple(int(item) for item in value.detach().cpu().reshape(-1).tolist())
    else:
        shape = tuple(int(item) for item in value)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"invalid logical weight shape: {shape}")
    return shape


def _round_fp16_st(value: torch.Tensor) -> torch.Tensor:
    rounded = value.to(torch.float16).to(torch.float32)
    return value + (rounded - value).detach()


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return value


class HardCodebookWeight(nn.Module):
    """Dense weight module with fixed assignments and deployable FP16 export.

    The module keeps the logical checkpoint's assignments and metadata fixed,
    while exposing an FP32 codebook and multiplicative row/column norm deltas
    for local optimization.  Forward reconstruction uses straight-through FP16
    rounding for every trainable serialized tensor so optimization sees the
    same precision that ``export_state`` writes back.
    """

    def __init__(
        self,
        logical_state: Mapping[str, Any],
        *,
        tune_codebook: bool = True,
        tune_norms: bool = True,
        tune_group_scales: bool = False,
    ) -> None:
        super().__init__()
        self._validate_required_keys(logical_state)
        self._rows, self._columns = _shape_tuple(logical_state["shape"])
        self._vector_dim = _as_int(logical_state["vector_dim"], "vector_dim")
        if self._vector_dim <= 0:
            raise ValueError("vector_dim must be positive")

        assignments = self._tensor(logical_state["assignments"], "assignments")
        codebook = self._tensor(logical_state["codebook"], "codebook")
        row_norm = self._tensor(logical_state["row_norm"], "row_norm")
        col_norm = self._tensor(logical_state["col_norm"], "col_norm")

        if not torch.is_floating_point(codebook):
            raise ValueError("codebook must be a floating-point tensor")
        if not torch.is_floating_point(row_norm):
            raise ValueError("row_norm must be a floating-point tensor")
        if not torch.is_floating_point(col_norm):
            raise ValueError("col_norm must be a floating-point tensor")
        integer_assignment_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if hasattr(torch, "uint16"):
            integer_assignment_dtypes.add(torch.uint16)
        if assignments.dtype not in integer_assignment_dtypes:
            raise ValueError("assignments must be an integer tensor")
        if assignments.numel() != self._rows * self._blocks_per_row:
            raise ValueError("assignment count does not match logical weight geometry")
        if codebook.ndim != 2 or codebook.shape[1] != self._vector_dim:
            raise ValueError("codebook shape does not match vector_dim")
        if row_norm.numel() != self._rows:
            raise ValueError("row_norm length does not match logical rows")
        if col_norm.numel() != self._columns:
            raise ValueError("col_norm length does not match logical columns")
        if assignments.numel():
            assignment_ids = assignments.detach().long()
            if (
                int(assignment_ids.min()) < 0
                or int(assignment_ids.max()) >= codebook.shape[0]
            ):
                raise ValueError("assignments contain code IDs outside the codebook")

        self._validate_finite("codebook", codebook)
        self._validate_finite("row_norm", row_norm)
        self._validate_finite("col_norm", col_norm)

        self.register_buffer("assignments", assignments.detach().clone())
        self.codebook = nn.Parameter(
            codebook.detach().to(dtype=torch.float32).clone(),
            requires_grad=bool(tune_codebook),
        )
        self.register_buffer(
            "row_norm_base",
            row_norm.detach().reshape(self._rows).to(dtype=torch.float32).clone(),
        )
        self.register_buffer(
            "col_norm_base",
            col_norm.detach().reshape(self._columns).to(dtype=torch.float32).clone(),
        )
        self.row_norm_delta = nn.Parameter(
            torch.ones(self._rows, dtype=torch.float32),
            requires_grad=bool(tune_norms),
        )
        self.col_norm_delta = nn.Parameter(
            torch.ones(self._columns, dtype=torch.float32),
            requires_grad=bool(tune_norms),
        )

        self._metadata = {
            key: _clone_value(value)
            for key, value in logical_state.items()
            if key
            not in {
                "assignments",
                "codebook",
                "row_norm",
                "col_norm",
                "group_scale",
            }
        }
        self.group_scale_delta = None
        if tune_group_scales and 'group_scale' not in logical_state:
            raise ValueError('Group-scale training requires a grouped source checkpoint')
        if "group_scale" in logical_state:
            if "group_size" not in logical_state:
                raise ValueError("group_scale checkpoint is missing group_size")
            group_size = _as_int(logical_state["group_size"], "group_size")
            if group_size <= 0:
                raise ValueError("group_size must be positive")
            group_scale = self._tensor(logical_state["group_scale"], "group_scale")
            expected_groups = math.ceil(self._columns / group_size)
            if tuple(group_scale.shape) != (self._rows, expected_groups):
                raise ValueError("group_scale shape does not match logical weight geometry")
            self._validate_finite("group_scale", group_scale)
            if tune_group_scales:
                if group_scale.dtype != torch.float16 or not bool((group_scale > 0).all()):
                    raise ValueError('Trainable group scales must be positive finite FP16')
                self.group_scale_delta = nn.Parameter(torch.ones_like(group_scale, dtype=torch.float32))
            self.register_buffer("group_scale", group_scale.detach().clone())
        else:
            self.group_scale = None

    @property
    def _blocks_per_row(self) -> int:
        return math.ceil(self._columns / self._vector_dim)

    @staticmethod
    def _validate_required_keys(logical_state: Mapping[str, Any]) -> None:
        required = {
            "assignments",
            "codebook",
            "row_norm",
            "col_norm",
            "shape",
            "vector_dim",
            "assignment_bits",
        }
        missing = required.difference(logical_state)
        if missing:
            raise ValueError(f"logical state is missing keys: {sorted(missing)}")

    @staticmethod
    def _tensor(value: Any, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a tensor")
        return value

    @staticmethod
    def _validate_finite(name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value.detach().to(dtype=torch.float32)).all():
            raise ValueError(f"{name} must contain only finite values")

    def _effective_serialized_params(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        codebook = _round_fp16_st(self.codebook)
        row_norm = _round_fp16_st(self.row_norm_base * self.row_norm_delta)
        col_norm = _round_fp16_st(self.col_norm_base * self.col_norm_delta)
        for name, value in (
            ("codebook", codebook),
            ("row_norm", row_norm),
            ("col_norm", col_norm),
        ):
            self._validate_finite(name, value)
        return codebook, row_norm, col_norm

    def _effective_group_scale(self):
        if self.group_scale_delta is None:
            return self.group_scale
        effective = self.group_scale.float() * self.group_scale_delta
        self._validate_finite('group_scale', effective)
        # Project to the positive finite serialized domain before FP16 STE.
        # Saturated deltas have zero gradient, matching the clamp derivative.
        return _round_fp16_st(effective.clamp(2**-24, 65504))

    def forward(self) -> torch.Tensor:
        codebook, row_norm, col_norm = self._effective_serialized_params()
        assignments = self.assignments.to(device=codebook.device, dtype=torch.long)
        padded_columns = self._blocks_per_row * self._vector_dim
        weight = codebook[assignments].reshape(self._rows, padded_columns)[
            :, : self._columns
        ]
        weight = weight * row_norm[:, None] * col_norm[None, :]
        if self.group_scale is not None:
            group_size = _as_int(self._metadata["group_size"], "group_size")
            group_scale = self._effective_group_scale().to(device=weight.device, dtype=torch.float32)
            expanded = group_scale.repeat_interleave(group_size, dim=1)[:, : self._columns]
            weight = weight * expanded
        return weight

    def export_state(self) -> dict[str, Any]:
        codebook, row_norm, col_norm = self._effective_serialized_params()
        state = {key: _clone_value(value) for key, value in self._metadata.items()}
        state["assignments"] = self.assignments.detach().to("cpu").clone()
        state["codebook"] = codebook.detach().to("cpu", torch.float16)
        state["row_norm"] = row_norm.detach().to("cpu", torch.float16)
        state["col_norm"] = col_norm.detach().to("cpu", torch.float16)
        if self.group_scale is not None:
            group_scale = self._effective_group_scale().detach().to('cpu')
            state['group_scale'] = group_scale.to(torch.float16) if self.group_scale_delta is not None else group_scale.clone()
        return state
