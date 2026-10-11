"""Validated, reusable calibration-token cache for Qwen3 QTIP runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


CACHE_FORMAT = "qtip-qwen3-calibration-v1"


def load_qtip_calibration_cache(
    path: str | Path,
    *,
    model_type: str,
    vocab_size: int,
    ctx_size: int,
    hessian_devset_size: int,
    finetune_devset_size: int,
) -> dict[str, Any]:
    cache_path = Path(path).resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing QTIP calibration cache: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict) or not isinstance(cache.get("metadata"), dict):
        raise ValueError("QTIP calibration cache is missing metadata")
    metadata = cache["metadata"]
    expected = {
        "format": CACHE_FORMAT,
        "tokenizer_model_type": model_type,
        "tokenizer_vocab_size": vocab_size,
        "ctx_size": ctx_size,
        "hessian_devset_size": hessian_devset_size,
        "finetune_devset_size": finetune_devset_size,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"QTIP calibration cache {key} mismatch: "
                f"{metadata.get(key)!r} != {value!r}"
            )
    shapes = {
        "hessian_tokens": (hessian_devset_size, ctx_size),
        "finetune_tokens": (finetune_devset_size, ctx_size),
    }
    result: dict[str, Any] = {"metadata": metadata}
    for key, expected_shape in shapes.items():
        value = cache.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"QTIP calibration cache is missing tensor {key}")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"QTIP calibration cache {key} shape mismatch: "
                f"{tuple(value.shape)} != {expected_shape}"
            )
        if value.numel() and (int(value.min()) < 0 or int(value.max()) >= vocab_size):
            raise ValueError(f"QTIP calibration cache {key} contains out-of-vocab IDs")
        result[key] = value.to(torch.int64).contiguous()
    return result
