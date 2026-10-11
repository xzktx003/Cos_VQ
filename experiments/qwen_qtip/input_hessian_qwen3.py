"""Generate official-protocol QTIP input Hessians for dense Qwen3."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)

from experiments.qwen_qtip.qtip_runtime import (
    import_qtip_runtime,
    install_qwen3_tp_plan_compat,
)
from experiments.qwen_qtip.qwen3_layer import Qwen3QTIPDecoderLayer
from experiments.qwen_qtip.calibration_data import load_qtip_calibration_cache


HESSIAN_KEYS = (
    ("qkv", "self_attn.q_proj"),
    ("o", "self_attn.o_proj"),
    ("up", "mlp.up_proj"),
    ("down", "mlp.down_proj"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtip-repo", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--large-batch-size", default=512, type=int)
    parser.add_argument("--devset-size", default=8192, type=int)
    parser.add_argument("--ctx-size", default=4096, type=int)
    return parser.parse_args()


def register_input_hessian_hook(module, save_prefix: str, device: int):
    """Accumulate the same X^T X statistic directly in its stored FP32 dtype."""
    width = module.in_features
    hessian = torch.zeros(width, width, dtype=torch.float32, device=device)
    count = 0

    def hook(_module, inputs):
        nonlocal hessian, count
        values = inputs[0].reshape(-1, width).to(torch.float32)
        hessian.addmm_(values.T, values)
        count += len(values)

    handle = module.register_forward_pre_hook(hook)

    def done() -> None:
        nonlocal hessian, count, handle
        torch.save(
            {"H": hessian, "n": width, "ct": count, "accumulation_dtype": "float32_tf32"},
            f"{save_prefix}_{device}.pt",
        )
        handle.remove()
        del hessian, handle

    return done


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _merge_hessian(
    utils,
    *,
    destination: Path,
    temporary: Path,
    split_index: int,
) -> None:
    new_data = torch.load(temporary, map_location="cpu", weights_only=False)
    if destination.exists():
        current = torch.load(destination, map_location="cpu", weights_only=False)
        completed = list(current.get("completed_splits", []))
        if split_index in completed:
            temporary.unlink(missing_ok=True)
            return
        flat_sum = current["flatH"].to(torch.float32) * int(current["ct"])
        flat_sum += utils.sym_to_flat(new_data["H"])
        count = int(current["ct"]) + int(new_data["ct"])
        completed.append(split_index)
        n = int(current["n"])
    else:
        flat_sum = utils.sym_to_flat(new_data["H"])
        count = int(new_data["ct"])
        completed = [split_index]
        n = int(new_data["n"])
    merged = {
        "flatH": (flat_sum / count).float(),
        "n": n,
        "ct": count,
        "completed_splits": sorted(completed),
        "accumulation_dtype": "float32_tf32",
    }
    _atomic_torch_save(destination, merged)
    temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.devset_size % args.large_batch_size:
        raise ValueError("--devset-size must be divisible by --large-batch-size")
    if args.large_batch_size % args.batch_size:
        raise ValueError("--large-batch-size must be divisible by --batch-size")
    output = Path(args.save_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {
            "base_model": str(Path(args.base_model).resolve()),
            "completed_splits": [],
            "completed": False,
        }
    )
    if progress["base_model"] != str(Path(args.base_model).resolve()):
        raise ValueError("Hessian resume base model mismatch")

    _, utils = import_qtip_runtime(args.qtip_repo)
    config = AutoConfig.from_pretrained(args.base_model)
    install_qwen3_tp_plan_compat(config)
    cache = load_qtip_calibration_cache(
        args.token_cache,
        model_type=config.model_type,
        vocab_size=int(config.vocab_size),
        ctx_size=args.ctx_size,
        hessian_devset_size=args.devset_size,
        finetune_devset_size=384,
    )
    devset = cache["hessian_tokens"]
    splits = torch.split(devset, args.large_batch_size)

    for split_index, token_split in enumerate(splits):
        if split_index in progress["completed_splits"]:
            print(f"HESSIAN_SPLIT_SKIP {split_index}", flush=True)
            continue
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            config=config,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        if model.config.model_type != "qwen3":
            raise ValueError(f"Expected qwen3, got {model.config.model_type!r}")
        hidden_batches = model.model.embed_tokens(
            token_split.view(-1, args.batch_size, args.ctx_size)
        )
        position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :]
        position_ids = position_ids + torch.zeros(
            args.batch_size, args.ctx_size, dtype=torch.int64
        )
        attention_mask = _prepare_4d_causal_attention_mask(
            None,
            (args.batch_size, args.ctx_size),
            hidden_batches[0],
            0,
            sliding_window=getattr(model.config, "sliding_window", None),
        ).cuda()
        position_ids = position_ids.cuda()

        layer_index = 0
        while len(model.model.layers) > 0:
            layer = Qwen3QTIPDecoderLayer(
                model.model.layers[0],
                copy.deepcopy(model.model.rotary_emb),
            ).cuda()
            callbacks = {}
            for key, attribute in HESSIAN_KEYS:
                module = layer.get_submodule(attribute)
                prefix = f"/dev/shm/qtip_qwen3_{os.getpid()}_{layer_index}_{key}"
                callbacks[key] = (
                    register_input_hessian_hook(module, prefix, 0),
                    Path(f"{prefix}_0.pt"),
                )
            for batch_index in range(len(hidden_batches)):
                hidden_batches[batch_index] = layer(
                    hidden_batches[batch_index].cuda(),
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=False,
                )[0].cpu()
            layer.cpu()
            for key, (done, temporary) in callbacks.items():
                done()
                _merge_hessian(
                    utils,
                    destination=output / f"{layer_index}_{key}.pt",
                    temporary=temporary,
                    split_index=split_index,
                )
            del layer, model.model.layers[0]
            utils.clean()
            print(
                f"HESSIAN_LAYER_DONE split={split_index} layer={layer_index}",
                flush=True,
            )
            layer_index += 1

        del attention_mask, position_ids, hidden_batches, model
        utils.clean()
        progress["completed_splits"].append(split_index)
        progress["completed_splits"] = sorted(set(progress["completed_splits"]))
        progress["completed"] = len(progress["completed_splits"]) == len(splits)
        _atomic_json(progress_path, progress)

    print(json.dumps(progress, indent=2), flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    main()
