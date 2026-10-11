"""Compose auditable layer-wise hybrids from two logical checkpoints."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Mapping
from pathlib import Path

import torch


def parse_layer_spec(spec: str) -> set[int]:
    """Parse a comma-separated layer list such as ``0-3,28-31``."""
    if not spec.strip():
        raise ValueError("donor layer specification must not be empty")
    layers: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"invalid empty layer in specification: {spec!r}")
        if "-" in part:
            fields = part.split("-")
            if len(fields) != 2 or not all(field.isdigit() for field in fields):
                raise ValueError(f"invalid layer range: {part!r}")
            start, end = (int(field) for field in fields)
            if start > end:
                raise ValueError(f"descending layer range is not allowed: {part!r}")
            layers.update(range(start, end + 1))
        elif part.isdigit():
            layers.add(int(part))
        else:
            raise ValueError(f"invalid layer index: {part!r}")
    return layers


def _layer_mapping(checkpoint: Mapping[str, object], label: str) -> Mapping[str, object]:
    layers = checkpoint.get("layers")
    if not isinstance(layers, Mapping):
        raise ValueError(f"{label} checkpoint is missing the layers mapping")
    return layers


def _validate_layer_compatibility(
    base_layer: object,
    donor_layer: object,
    *,
    layer_index: int,
) -> None:
    if not isinstance(base_layer, Mapping) or not isinstance(donor_layer, Mapping):
        raise ValueError(f"layer {layer_index} must be a mapping in both checkpoints")
    if set(base_layer) != set(donor_layer):
        raise ValueError(
            f"Linear coverage mismatch at layer {layer_index}: "
            f"base={sorted(base_layer)}, donor={sorted(donor_layer)}"
        )
    for name in base_layer:
        base_state = base_layer[name]
        donor_state = donor_layer[name]
        if not isinstance(base_state, Mapping) or not isinstance(donor_state, Mapping):
            raise ValueError(f"layer {layer_index} {name} state must be a mapping")
        for key in ("assignments", "codebook"):
            base_tensor = base_state.get(key)
            donor_tensor = donor_state.get(key)
            if not isinstance(base_tensor, torch.Tensor) or not isinstance(
                donor_tensor, torch.Tensor
            ):
                raise ValueError(
                    f"layer {layer_index} {name} is missing tensor {key!r}"
                )
            if base_tensor.shape != donor_tensor.shape:
                raise ValueError(
                    f"Tensor shape mismatch at layer {layer_index} {name} {key}: "
                    f"base={tuple(base_tensor.shape)}, donor={tuple(donor_tensor.shape)}"
                )


def compose_logical_checkpoints(
    base_checkpoint: Mapping[str, object],
    donor_checkpoint: Mapping[str, object],
    *,
    donor_layers: set[int],
    base_label: str = "base",
    donor_label: str = "donor",
) -> dict[str, object]:
    """Return ``base_checkpoint`` with selected layers taken from the donor."""
    base_layers = _layer_mapping(base_checkpoint, base_label)
    donor_layer_mapping = _layer_mapping(donor_checkpoint, donor_label)
    if set(base_layers) != set(donor_layer_mapping):
        raise ValueError("base and donor checkpoint layer coverage differs")

    available_layers = {int(key) for key in base_layers}
    unknown = sorted(donor_layers - available_layers)
    if unknown:
        raise ValueError(f"donor layers {unknown} are outside checkpoint coverage")

    composed_layers = dict(base_layers)
    for layer_index in sorted(donor_layers):
        key = str(layer_index)
        _validate_layer_compatibility(
            base_layers[key],
            donor_layer_mapping[key],
            layer_index=layer_index,
        )
        composed_layers[key] = donor_layer_mapping[key]

    composed = dict(base_checkpoint)
    composed["layers"] = composed_layers
    composed["composition"] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_label": base_label,
        "donor_label": donor_label,
        "donor_layers": sorted(donor_layers),
        "base_layers": sorted(available_layers - donor_layers),
    }
    return composed


def _load_checkpoint(path: Path, *, trusted: bool) -> Mapping[str, object]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=not trusted,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint {path} must contain a mapping")
    return checkpoint


def write_composed_checkpoint(
    output: Path,
    checkpoint: Mapping[str, object],
    *,
    protected_inputs: tuple[Path, ...],
) -> None:
    """Atomically write a hybrid without allowing source overwrite."""
    resolved_output = output.resolve()
    if resolved_output in {path.resolve() for path in protected_inputs}:
        raise ValueError("output checkpoint must differ from both input checkpoints")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(dict(checkpoint), temporary)
    os.replace(temporary, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--donor-layers", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--donor-label", default="donor")
    parser.add_argument(
        "--trusted-checkpoints",
        action="store_true",
        help="Allow legacy pickle loading for audited project checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    donor_layers = parse_layer_spec(args.donor_layers)
    base = _load_checkpoint(args.base, trusted=args.trusted_checkpoints)
    donor = _load_checkpoint(args.donor, trusted=args.trusted_checkpoints)
    composed = compose_logical_checkpoints(
        base,
        donor,
        donor_layers=donor_layers,
        base_label=args.base_label,
        donor_label=args.donor_label,
    )
    write_composed_checkpoint(
        args.output,
        composed,
        protected_inputs=(args.base, args.donor),
    )
    print(
        f"Wrote {args.output} with donor layers "
        f"{','.join(str(layer) for layer in sorted(donor_layers))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
