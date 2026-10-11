"""Run upstream QTIP layerwise quantization/fine-tuning on dense Qwen3."""

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

import glog
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)

from experiments.qwen_qtip.qtip_artifacts import QTIP_LAYER_SUFFIXES
from experiments.qwen_qtip.qtip_runtime import (
    import_qtip_runtime,
    install_qwen3_tp_plan_compat,
)
from experiments.qwen_qtip.qwen3_layer import Qwen3QTIPDecoderLayer
from experiments.qwen_qtip.calibration_data import load_qtip_calibration_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtip-repo", required=True)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-cpu-threads", default=8, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--devset-size", default=384, type=int)
    parser.add_argument("--ctx-size", default=4096, type=int)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--in-hess-path", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--sigma-reg", default=1e-2, type=float)
    parser.add_argument("--sigma-reg2", default=1e-2, type=float)
    parser.add_argument("--scale-override", default=0.9, type=float)
    parser.add_argument("--codebook", default="bitshift")
    parser.add_argument("--use-fp64", action="store_true")
    parser.add_argument("--no-use-buffered", action="store_true")
    parser.add_argument("--sample-proc", default=1, type=int)
    parser.add_argument("--lowmem-ldlq", action="store_true")
    parser.add_argument("--ft-lr", default=3e-6, type=float)
    parser.add_argument("--ft-bs", default=4, type=int)
    parser.add_argument("--ft-update-freq", default=1, type=int)
    parser.add_argument("--ft-epochs", default=5, type=int)
    parser.add_argument("--ft-valid-freq", default=1, type=int)
    parser.add_argument("--ft-valid-size", default=128, type=float)
    parser.add_argument("--ft-early-stop", default=5, type=int)
    parser.add_argument("--ft-grad-ckpt", action="store_true")
    parser.add_argument("--td-x", default=16, type=int)
    parser.add_argument("--td-y", default=16, type=int)
    parser.add_argument("--L", default=16, type=int)
    parser.add_argument("--K", default=2, type=int)
    parser.add_argument("--V", default=2, type=int)
    parser.add_argument("--tlut-bits", default=9, type=int)
    parser.add_argument("--decode-mode", default="quantlut_sym")
    parser.add_argument("--ft-train-lut", action="store_true")
    parser.add_argument("--split-for-tp", action="store_true")
    parser.add_argument("--tp-rank", default=8, type=int)
    parser.add_argument("--skip-list", default=None)
    return parser.parse_args()


def _complete_layer(root: Path, index: int) -> bool:
    return all((root / f"{index}_{suffix}.pt").is_file() for suffix in QTIP_LAYER_SUFFIXES)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.tlut_bits = args.tlut_bits
    args.num_cpu_threads = args.num_cpu_threads
    args.no_use_buffered = args.no_use_buffered
    args.lowmem_ldlq = args.lowmem_ldlq
    args.ft_grad_ckpt = args.ft_grad_ckpt
    args.ft_train_lut = args.ft_train_lut
    args.split_for_tp = args.split_for_tp
    args.skip_list = args.skip_list.split(",") if args.skip_list else []
    output = Path(args.save_path).resolve()
    output.mkdir(parents=True, exist_ok=True)

    _, utils = import_qtip_runtime(args.qtip_repo)
    from lib.algo import finetune  # type: ignore[import-not-found]
    from lib.codebook import bitshift  # type: ignore[import-not-found]

    codebook = bitshift.bitshift_codebook(
        L=args.L,
        K=args.K,
        V=args.V,
        tlut_bits=args.tlut_bits,
        decode_mode=args.decode_mode,
    )
    config = AutoConfig.from_pretrained(args.base_model)
    install_qwen3_tp_plan_compat(config)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    if model.config.model_type != "qwen3":
        raise ValueError(f"Expected qwen3, got {model.config.model_type!r}")
    quip_params = {
        "codebook": args.codebook,
        "codebook_version": codebook.version,
        "L": args.L,
        "K": args.K,
        "V": args.V,
        "tlut_bits": args.tlut_bits,
        "decode_mode": args.decode_mode,
        "td_x": args.td_x,
        "td_y": args.td_y,
        "split_for_tp": args.split_for_tp,
        "skip_list": args.skip_list,
    }
    model.config.update({"quip_params": quip_params})
    torch.save(
        {"quant_args": args, "model_config": model.config},
        output / "config.pt",
    )

    cache = load_qtip_calibration_cache(
        args.token_cache,
        model_type=model.config.model_type,
        vocab_size=int(model.config.vocab_size),
        ctx_size=args.ctx_size,
        hessian_devset_size=8192,
        finetune_devset_size=args.devset_size,
    )
    devset = cache["finetune_tokens"]
    original = model.model.embed_tokens(devset)
    propagated = torch.zeros_like(original)
    position_ids = torch.arange(args.ctx_size, dtype=torch.int32)[None, :]
    position_ids = position_ids + torch.zeros(
        args.batch_size, args.ctx_size, dtype=torch.int32
    )
    attention_mask = _prepare_4d_causal_attention_mask(
        None,
        (args.batch_size, args.ctx_size),
        original[: args.batch_size],
        0,
        sliding_window=getattr(model.config, "sliding_window", None),
    )

    for index in range(len(model.model.layers)):
        started = time.time()
        layer = Qwen3QTIPDecoderLayer(
            model.model.layers[index],
            copy.deepcopy(model.model.rotary_emb),
        )
        position_ids = position_ids.cuda()
        attention_mask = attention_mask.cuda()
        layer.cuda()
        for offset in range(0, args.devset_size, args.batch_size):
            propagated[offset : offset + args.batch_size] = layer(
                original[offset : offset + args.batch_size].cuda(),
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
            )[0].cpu()
        layer.cpu()
        position_ids = position_ids.cpu()
        attention_mask = attention_mask.cpu()
        utils.clean()
        glog.info("computed original embedding for layer %d in %.3fs", index, time.time() - started)

        if _complete_layer(output, index):
            glog.info("resume: layer %d already has all eight artifacts", index)
        else:
            quant_order = []
            for item in (
                ("self_attn.v_proj", 'v', "qkv", 'v', "col"),
                ("self_attn.q_proj", 'q', "qkv", 'q', "col"),
                ("self_attn.k_proj", 'k', "qkv", 'k', "col"),
                ("self_attn.o_proj", 'o', 'o', 'o', "row"),
                ("mlp.up_proj", 'up', 'up', 'up', "col"),
                ("mlp.gate_proj", 'gate', 'up', 'gate', "col"),
                ("mlp.down_proj", 'down', 'down', 'down', "row"),
            ):
                if f"{index}_{item[1]}" not in args.skip_list:
                    quant_order.append(item)
                else:
                    layer.get_submodule(item[0]).weight.requires_grad = False
            finetune.quantize_finetune_decoder_layer(
                layer,
                quant_order,
                index,
                codebook,
                args,
                0,
                original,
                propagated,
            )
            torch.save(
                {
                    "input_layernorm": layer.input_layernorm.weight,
                    "post_attention_layernorm": layer.post_attention_layernorm.weight,
                },
                output / f"{index}_layernorm.pt",
            )
            if not _complete_layer(output, index):
                raise RuntimeError(f"Layer {index} did not produce all QTIP artifacts")

        original.copy_(propagated)
        model.model.layers[index] = None
        del layer
        utils.clean()
        print(f"QTIP_LAYER_DONE {index}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    torch.set_grad_enabled(False)
    main()
