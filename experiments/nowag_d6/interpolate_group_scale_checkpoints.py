"""Geometrically interpolate deployable group scales between fixed-code checkpoints."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import torch


def interpolate_group_scale(
    base: torch.Tensor, tuned: torch.Tensor, alpha: float
) -> torch.Tensor:
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    if base.shape != tuned.shape:
        raise ValueError("group-scale shapes do not match")
    if alpha == 0.0:
        return base.clone()
    if alpha == 1.0:
        return tuned.clone()
    base_float = base.float()
    tuned_float = tuned.float()
    if bool((base_float <= 0).any()) or bool((tuned_float <= 0).any()):
        raise ValueError("group scales must be strictly positive for log interpolation")
    interpolated = torch.exp(
        torch.lerp(base_float.log(), tuned_float.log(), float(alpha))
    )
    if not bool(torch.isfinite(interpolated).all()):
        raise ValueError("interpolated group scales contain non-finite values")
    return interpolated.to(base.dtype)


def interpolate_checkpoint_group_scales(
    base: dict[str, Any], tuned: dict[str, Any], *, alpha: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    if set(base.get("layers", {})) != set(tuned.get("layers", {})):
        raise ValueError("checkpoint layer coverage mismatch")
    fixed_tensor_checks = 0
    fixed_scalar_checks = 0
    changed_group_scale_tensors = 0
    for layer_key in sorted(base["layers"], key=int):
        base_layer = base["layers"][layer_key]
        tuned_layer = tuned["layers"][layer_key]
        if set(base_layer) != set(tuned_layer):
            raise ValueError(f"layer {layer_key} Linear coverage mismatch")
        for name in sorted(base_layer):
            base_state = base_layer[name]
            tuned_state = tuned_layer[name]
            if set(base_state) != set(tuned_state):
                raise ValueError(f"state field coverage mismatch at layer {layer_key} {name}")
            if "group_scale" not in base_state:
                raise ValueError(f"missing group_scale at layer {layer_key} {name}")
            for key in sorted(base_state):
                base_value = base_state[key]
                tuned_value = tuned_state[key]
                if key == "group_scale":
                    continue
                if isinstance(base_value, torch.Tensor):
                    fixed_tensor_checks += 1
                    if not isinstance(tuned_value, torch.Tensor) or not torch.equal(
                        base_value, tuned_value
                    ):
                        raise ValueError(
                            f"fixed tensor mismatch at layer {layer_key} {name} {key}"
                        )
                else:
                    fixed_scalar_checks += 1
                    if base_value != tuned_value:
                        raise ValueError(
                            f"fixed metadata mismatch at layer {layer_key} {name} {key}"
                        )
            base_state["group_scale"] = interpolate_group_scale(
                base_state["group_scale"], tuned_state["group_scale"], alpha
            )
            changed_group_scale_tensors += 1
    audit = {
        "alpha": float(alpha),
        "layers": len(base["layers"]),
        "linears": sum(len(layer) for layer in base["layers"].values()),
        "changed_group_scale_tensors": changed_group_scale_tensors,
        "fixed_tensor_checks": fixed_tensor_checks,
        "fixed_scalar_checks": fixed_scalar_checks,
        "assignments_changed": False,
        "codebooks_changed": False,
        "logical_bitrate_changed": False,
    }
    base["group_scale_interpolation"] = audit
    return base, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trusted-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    inputs = {Path(args.base).resolve(), Path(args.tuned).resolve()}
    if output in inputs:
        raise ValueError("output must not overwrite an input checkpoint")
    base = torch.load(
        args.base, map_location="cpu", weights_only=not args.trusted_checkpoints
    )
    tuned = torch.load(
        args.tuned, map_location="cpu", weights_only=not args.trusted_checkpoints
    )
    checkpoint, audit = interpolate_checkpoint_group_scales(
        base, tuned, alpha=args.alpha
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, output)
    print(audit, flush=True)


if __name__ == "__main__":
    main()
