"""Runtime loading for QTIP-compressed dense Qwen3 checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from transformers import AutoConfig

from experiments.qwen_qtip.qwen3_qtip_adapter import (
    build_qwen3_qtip_model_class,
    validate_qwen3_qtip_config,
)


def install_torch24_dtensor_compat() -> None:
    """Expose Torch 2.4's legacy DTensor path where Transformers 4.52 expects it.

    Torch 2.4 defines ``DTensor`` under ``torch.distributed._tensor`` while
    Transformers 4.52 imports it from ``torch.distributed.tensor`` during
    ``save_pretrained``.  The class itself is unchanged; this only publishes
    the existing class through the newer namespace and fills Transformers'
    version-gated module global.
    """
    import torch.distributed.tensor as public_dtensor

    dtensor_cls = getattr(public_dtensor, "DTensor", None)
    if dtensor_cls is None:
        from torch.distributed._tensor import DTensor as dtensor_cls

        public_dtensor.DTensor = dtensor_cls

    import transformers.modeling_utils as modeling_utils

    if not hasattr(modeling_utils, "DTensor"):
        modeling_utils.DTensor = dtensor_cls


def install_qwen3_tp_plan_compat(config: object) -> None:
    """Preserve the Qwen3-only entry point used by the QTIP loader."""
    if str(getattr(config, "model_type", "")) != "qwen3":
        raise ValueError("TP-plan compatibility is restricted to Qwen3")
    install_tp_plan_compat(config)


def install_tp_plan_compat(config: object) -> None:
    """Repair Transformers' missing style registry without enabling TP.

    Transformers 4.52 can leave ``ALL_PARALLEL_STYLES`` as ``None`` when used
    with the Torch 2.4 build required by the QTIP extensions.  Qwen3 and LLaMA still
    carries a declarative TP plan and ``post_init`` then performs membership
    tests against that sentinel.  Only their known col/row styles are
    admitted; an unfamiliar plan remains fail-closed.
    """
    model_type = str(getattr(config, "model_type", ""))
    if model_type not in {"qwen3", "llama"}:
        raise ValueError("TP-plan compatibility is restricted to Qwen3 and LLaMA")
    plan = getattr(config, "base_model_tp_plan", None) or {}
    styles = {str(value) for value in plan.values()}
    allowed = {"colwise", "rowwise"}
    unknown = styles - allowed
    if unknown:
        raise ValueError(f"Unsupported {model_type} TP-plan styles: {sorted(unknown)}")
    import transformers.modeling_utils as modeling_utils

    if getattr(modeling_utils, "ALL_PARALLEL_STYLES", ()) is None:
        modeling_utils.ALL_PARALLEL_STYLES = styles or allowed


def import_qtip_runtime(qtip_repo: str | Path):
    repo = Path(qtip_repo).resolve()
    required = repo / "lib/linear/quantized_linear.py"
    if not required.is_file():
        raise FileNotFoundError(f"Not a QTIP checkout: missing {required}")
    for path in (repo, repo / "qtip-kernels"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    from lib import utils  # type: ignore[import-not-found]
    from lib.linear import QuantizedLinear  # type: ignore[import-not-found]

    return QuantizedLinear, utils


def load_qwen3_qtip_model(
    hf_path: str,
    *,
    qtip_repo: str | Path,
    device_map: Any = None,
    torch_dtype: Any = "auto",
    low_cpu_mem_usage: bool = True,
    **kwargs: Any,
):
    """Load a Qwen3 checkpoint whose seven decoder projections use QTIP."""
    quantized_linear_cls, _ = import_qtip_runtime(qtip_repo)
    config = AutoConfig.from_pretrained(hf_path, trust_remote_code=False)
    install_qwen3_tp_plan_compat(config)
    validate_qwen3_qtip_config(config)
    model_cls = build_qwen3_qtip_model_class(quantized_linear_cls)
    model = model_cls.from_pretrained(
        hf_path,
        config=config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        **kwargs,
    )
    base_model = str(
        config.quip_params.get("base_model")
        or getattr(config, "_name_or_path", "")
    )
    if not base_model:
        raise ValueError("Qwen3 QTIP config does not identify its base model")
    return model, base_model
