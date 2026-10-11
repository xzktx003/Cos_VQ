"""Adapt the shared RP1T cache to the input format of the original GSQ runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


def tensor_sha256(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous()
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_cache(
    source: dict[str, Any],
    *,
    model_path: str | Path,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Expose Hessian tokens as ``input_ids`` and retain the E2E text split."""
    model = Path(model_path).resolve(strict=True)
    metadata = source.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("RP1T cache is missing metadata")
    source_model = metadata.get("tokenizer_model")
    if not source_model or Path(source_model).resolve() != model:
        raise ValueError("RP1T cache tokenizer path does not match the selected model")

    hessian = source.get("hessian_tokens")
    finetune = source.get("finetune_tokens")
    for name, value in (("hessian_tokens", hessian), ("finetune_tokens", finetune)):
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or not value.numel():
            raise ValueError(f"RP1T cache requires a nonempty matrix: {name}")
        if value.dtype not in (torch.int16, torch.int32, torch.int64):
            raise ValueError(f"RP1T cache token tensor has unsupported dtype: {name}")
        if int(value.min()) < 0:
            raise ValueError(f"RP1T cache contains negative token IDs: {name}")

    expected_hessian = (
        int(metadata["hessian_devset_size"]),
        int(metadata["ctx_size"]),
    )
    expected_finetune = (
        int(metadata["finetune_devset_size"]),
        int(metadata["ctx_size"]),
    )
    if tuple(hessian.shape) != expected_hessian or tuple(finetune.shape) != expected_finetune:
        raise ValueError("RP1T token tensor shapes disagree with cache metadata")

    output_metadata = dict(metadata)
    output_metadata["reproducibility"] = {
        "source": "togethercomputer/RedPajama-Data-1T-Sample",
        "source_cache_sha256": (
            file_sha256(source_path)
            if source_path is not None
            else None
        ),
        "hessian_tokens_sha256": tensor_sha256(hessian),
        "finetune_tokens_sha256": tensor_sha256(finetune),
        "baseline_runner_rows": "input_ids aliases hessian_tokens; train/validation/GPTQ use published row offsets",
    }
    identity = (
        f"redpajama1t|{model}|len={int(metadata['ctx_size'])}|"
        f"hessian={int(metadata['hessian_devset_size'])}|"
        f"finetune={int(metadata['finetune_devset_size'])}|seed={int(metadata['seed'])}"
    )
    return {
        "identity": identity,
        "target_model": str(model),
        "input_ids": hessian,
        "hessian_tokens": hessian,
        "finetune_tokens": finetune,
        "metadata": output_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Raw output from build_calibration_cache")
    parser.add_argument("--model", required=True, help="Resolved local Hugging Face model snapshot")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_path = Path(args.source).resolve(strict=True)
    output_path = Path(args.output).resolve()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite cache: {output_path}")
    source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    result = prepare_cache(source, model_path=args.model, source_path=source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    torch.save(result, temporary)
    os.replace(temporary, output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "identity": result["identity"],
                "hessian_shape": list(result["input_ids"].shape),
                "finetune_shape": list(result["finetune_tokens"].shape),
                "metadata": result["metadata"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
