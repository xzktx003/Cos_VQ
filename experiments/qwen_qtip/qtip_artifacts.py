"""Completeness checks for upstream QTIP per-layer artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


QTIP_LINEAR_SUFFIXES = ("q", "k", "v", "o", "up", "gate", "down")
QTIP_LAYER_SUFFIXES = (*QTIP_LINEAR_SUFFIXES, "layernorm")


def audit_qtip_layer_artifacts(
    quantized_path: str | Path,
    *,
    expected_layers: int,
) -> dict[str, Any]:
    root = Path(quantized_path)
    if expected_layers <= 0:
        raise ValueError("expected_layers must be positive")
    artifacts: list[str] = []
    for layer_index in range(expected_layers):
        for suffix in QTIP_LAYER_SUFFIXES:
            artifact = root / f"{layer_index}_{suffix}.pt"
            if not artifact.is_file():
                raise FileNotFoundError(f"Missing QTIP artifact: {artifact}")
            artifacts.append(str(artifact))
    return {
        "quantized_layers": expected_layers,
        "quantized_linears": expected_layers * len(QTIP_LINEAR_SUFFIXES),
        "artifacts": artifacts,
    }
