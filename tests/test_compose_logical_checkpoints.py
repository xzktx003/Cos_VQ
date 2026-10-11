from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.nowag_d6.compose_logical_checkpoints import (
    compose_logical_checkpoints,
    parse_layer_spec,
)


def _state(value: int) -> dict[str, torch.Tensor | int]:
    return {
        "assignments": torch.tensor([[value]], dtype=torch.int64),
        "codebook": torch.tensor([[float(value)]], dtype=torch.float16),
        "assignment_bits": 2,
    }


def _checkpoint(offset: int) -> dict[str, object]:
    return {
        "layers": {
            str(layer): {"linear": _state(offset + layer)}
            for layer in range(4)
        },
        "source": f"checkpoint-{offset}",
    }


def test_parse_layer_spec_supports_ranges_and_deduplicates() -> None:
    assert parse_layer_spec("0-2,2,5,7-8") == {0, 1, 2, 5, 7, 8}


@pytest.mark.parametrize("spec", ["", "2-1", "-1", "1-a"])
def test_parse_layer_spec_rejects_ambiguous_values(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_layer_spec(spec)


def test_compose_replaces_only_requested_layers_and_records_provenance() -> None:
    base = _checkpoint(10)
    donor = _checkpoint(20)

    composed = compose_logical_checkpoints(
        base,
        donor,
        donor_layers={0, 3},
        base_label="new-group160",
        donor_label="old-vector-gsq",
    )

    assert composed["layers"]["0"]["linear"]["assignments"].item() == 20
    assert composed["layers"]["1"]["linear"]["assignments"].item() == 11
    assert composed["layers"]["2"]["linear"]["assignments"].item() == 12
    assert composed["layers"]["3"]["linear"]["assignments"].item() == 23
    assert composed["composition"]["donor_layers"] == [0, 3]
    assert composed["composition"]["base_label"] == "new-group160"
    assert composed["composition"]["donor_label"] == "old-vector-gsq"


def test_compose_rejects_incompatible_linear_coverage() -> None:
    base = _checkpoint(10)
    donor = _checkpoint(20)
    donor["layers"]["2"] = {"different_linear": _state(22)}

    with pytest.raises(ValueError, match="Linear coverage mismatch"):
        compose_logical_checkpoints(base, donor, donor_layers={2})


def test_compose_rejects_unknown_donor_layer() -> None:
    with pytest.raises(ValueError, match="outside checkpoint coverage"):
        compose_logical_checkpoints(
            _checkpoint(10),
            _checkpoint(20),
            donor_layers={4},
        )

