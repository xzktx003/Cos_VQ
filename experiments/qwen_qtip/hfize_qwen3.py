"""Convert upstream per-layer QTIP artifacts into a reloadable Qwen3 model."""

from __future__ import annotations

import argparse
from operator import attrgetter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.qwen_qtip.qtip_artifacts import (
    QTIP_LINEAR_SUFFIXES,
    audit_qtip_layer_artifacts,
)
from experiments.qwen_qtip.qtip_runtime import (
    import_qtip_runtime,
    install_qwen3_tp_plan_compat,
    install_torch24_dtensor_compat,
    load_qwen3_qtip_model,
)
from experiments.qwen_qtip.qwen3_qtip_adapter import (
    QWEN3_QTIP_LINEAR_SPECS,
    replace_qwen3_linears,
    validate_qwen3_qtip_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtip-repo", required=True)
    parser.add_argument("--quantized-path", required=True)
    parser.add_argument("--hf-output-path", required=True)
    parser.add_argument(
        "--base-model",
        default="",
        help="Override the base model recorded in config.pt.",
    )
    return parser.parse_args()


def _torch_load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    args = parse_args()
    install_torch24_dtensor_compat()
    quantized_path = Path(args.quantized_path).resolve()
    output_path = Path(args.hf_output_path).resolve()
    saved_config = _torch_load(quantized_path / "config.pt")
    config = saved_config["model_config"]
    install_qwen3_tp_plan_compat(config)
    quip_params = validate_qwen3_qtip_config(config)
    base_model = args.base_model or str(getattr(config, "_name_or_path", ""))
    if not base_model:
        raise ValueError("config.pt does not identify the original Qwen3 model")
    quip_params["base_model"] = base_model
    quip_params["skip_list"] = list(quip_params.get("skip_list") or [])
    config.architectures = ["Qwen3QTIPForCausalLM"]

    quantized_linear_cls, qtip_utils = import_qtip_runtime(args.qtip_repo)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=config,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    replace_qwen3_linears(
        model,
        config,
        quantized_linear_cls=quantized_linear_cls,
    )
    audit_qtip_layer_artifacts(
        quantized_path,
        expected_layers=len(model.model.layers),
    )

    skipped = set(quip_params["skip_list"])
    spec_by_suffix = {
        spec.save_name: spec for spec in QWEN3_QTIP_LINEAR_SPECS
    }
    for layer_index, layer in enumerate(model.model.layers):
        layernorm = _torch_load(quantized_path / f"{layer_index}_layernorm.pt")
        layer.input_layernorm.weight.copy_(
            layernorm["input_layernorm"].to(layer.input_layernorm.weight.dtype)
        )
        layer.post_attention_layernorm.weight.copy_(
            layernorm["post_attention_layernorm"].to(
                layer.post_attention_layernorm.weight.dtype
            )
        )
        for suffix in QTIP_LINEAR_SUFFIXES:
            if f"{layer_index}_{suffix}" in skipped:
                continue
            spec = spec_by_suffix[suffix]
            module = attrgetter(spec.attribute)(layer)
            state = _torch_load(quantized_path / f"{layer_index}_{suffix}.pt")
            qtip_utils.unpack_quip(module, state)

    output_path.mkdir(parents=True, exist_ok=True)
    model.config._name_or_path = base_model
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.save_pretrained(output_path)
    del model

    reloaded, reloaded_base = load_qwen3_qtip_model(
        str(output_path),
        qtip_repo=args.qtip_repo,
        device_map="cpu",
    )
    if reloaded_base != base_model:
        raise ValueError(
            f"Reloaded base model mismatch: {reloaded_base!r} != {base_model!r}"
        )
    expected_quantized = len(reloaded.model.layers) * len(QTIP_LINEAR_SUFFIXES)
    observed_quantized = sum(
        isinstance(module, quantized_linear_cls) for module in reloaded.modules()
    )
    if observed_quantized != expected_quantized - len(skipped):
        raise ValueError(
            "Reloaded QTIP Linear count mismatch: "
            f"{observed_quantized} != {expected_quantized - len(skipped)}"
        )
    print(
        f"Qwen3 QTIP HF checkpoint ready: {output_path} "
        f"({observed_quantized} QuantizedLinear modules)"
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
