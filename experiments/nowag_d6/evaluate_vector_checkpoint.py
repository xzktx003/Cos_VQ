"""Independently reconstruct and evaluate a logical NoWag-d6 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from experiments.nowag_d6.logical_state import logical_state_bits, residual_metadata
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.nowag_d6.diagnose_block_vector_gsq import (
    LINEAR_NAMES,
    get_linear_modules,
)
from experiments.nowag_d6.run_prefix_vector_gsq import (
    evaluate_wikitext2,
    evaluate_ppl,
    get_dataset,
    prepare_test_dataloader,
    reconstruct_logical_state,
)


def checkpoint_ppl_experiment_name(model_path: str) -> str:
    """Build an auditable result label from the model selected at runtime."""
    model_name = Path(model_path).name or "model"
    model_slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    return f"{model_slug}_nowag_d6_checkpoint_independent_ppl"


def load_logical_checkpoint(
    checkpoint_path: str,
    *,
    trusted: bool,
    torch_load=torch.load,
):
    """Load a checkpoint safely unless legacy trusted storage is explicit."""
    return torch_load(
        checkpoint_path,
        map_location="cpu",
        weights_only=not trusted,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--trusted-checkpoint",
        action="store_true",
        help=(
            "Allow PyTorch's legacy pickle loader for a checkpoint produced and "
            "audited by this project. Required for uint16 storage on PyTorch 2.3."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ppl-seqlen", type=int, default=2048)
    parser.add_argument("--ppl-batch-size", type=int, default=8)
    parser.add_argument(
        "--ppl-split",
        choices=("train", "validation", "test"),
        default="test",
        help="WikiText2 split. Test remains the formal final metric.",
    )
    return parser.parse_args()


def evaluate_wikitext2_split(
    model,
    tokenizer,
    *,
    split: str,
    sequence_length: int,
    batch_size: int,
) -> float:
    if split == "test":
        return evaluate_wikitext2(model, tokenizer, sequence_length, batch_size)
    if split not in {"train", "validation"}:
        raise ValueError(f"unsupported WikiText2 split: {split}")
    dataset = get_dataset("wikitext2", tokenizer)
    loader = prepare_test_dataloader(
        dataset[split],
        tokenizer,
        seqlen=sequence_length,
        batch_size=batch_size,
    )
    return evaluate_ppl(model, model.config.pad_token_id, loader)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    checkpoint = load_logical_checkpoint(
        args.checkpoint,
        trusted=args.trusted_checkpoint,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()

    expected_layers = len(model.model.layers)
    layer_keys = sorted(int(key) for key in checkpoint["layers"])
    if layer_keys != list(range(expected_layers)):
        raise ValueError(
            f"Checkpoint layers {layer_keys} do not cover 0..{expected_layers - 1}"
        )

    total_bits = 0
    total_weights = 0
    per_linear = []
    for layer_index in layer_keys:
        layer_state = checkpoint["layers"][str(layer_index)]
        if set(layer_state) != set(LINEAR_NAMES):
            raise ValueError(
                f"Layer {layer_index} Linear coverage mismatch: {sorted(layer_state)}"
            )
        modules = get_linear_modules(model, layer_index)
        for name in LINEAR_NAMES:
            state = layer_state[name]
            module = modules[name]
            assignments = state["assignments"].long()
            codebook_size = int(state["codebook"].shape[0])
            assignment_bits = int(state["assignment_bits"])
            if int(assignments.min()) < 0 or int(assignments.max()) >= codebook_size:
                raise ValueError(f"Out-of-range assignment in layer {layer_index} {name}")
            metadata_keys = ["codebook", "row_norm", "col_norm"]
            if "group_scale" in state:
                metadata_keys.append("group_scale")
            tensors_finite = all(
                bool(torch.isfinite(state[key].float()).all())
                for key in metadata_keys
            )
            if not tensors_finite:
                raise ValueError(f"Non-finite metadata in layer {layer_index} {name}")
            weight = reconstruct_logical_state(state, device)
            module.weight.data.copy_(weight.to(module.weight.dtype))
            reconstructed = weight.to(module.weight.dtype)
            exact = torch.equal(module.weight.detach(), reconstructed)
            if not exact:
                raise AssertionError(
                    f"Fresh reconstruction mismatch in layer {layer_index} {name}"
                )
            bits = logical_state_bits(state)
            total_bits += bits
            total_weights += module.weight.numel()
            per_linear.append({
                "layer": layer_index,
                "name": name,
                "shape": list(module.weight.shape),
                "assignment_bits": assignment_bits,
                **residual_metadata(state),
                "codebook_size": codebook_size,
                "used_codewords": int(torch.unique(assignments).numel()),
                "assignment_min": int(assignments.min()),
                "assignment_max": int(assignments.max()),
                "fresh_reconstruction_exact": exact,
                "group_scale_count": int(
                    state["group_scale"].numel()
                    if "group_scale" in state
                    else 0
                ),
                "group_size": int(state["group_size"])
                if "group_size" in state
                else None,
            })

    started = time.time()
    ppl = evaluate_wikitext2_split(
        model,
        tokenizer,
        split=args.ppl_split,
        sequence_length=args.ppl_seqlen,
        batch_size=args.ppl_batch_size,
    )
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment": checkpoint_ppl_experiment_name(args.model),
        "arguments": vars(args),
        "quantized_layers": expected_layers,
        "quantized_linears": len(per_linear),
        "expected_linears": expected_layers * len(LINEAR_NAMES),
        "linear_names": list(LINEAR_NAMES),
        "q_norm_k_norm_float_count": sum(
            1
            for name, _ in model.named_parameters()
            if name.endswith("q_norm.weight") or name.endswith("k_norm.weight")
        ),
        "lm_head_in_logical_checkpoint": any(
            "lm_head" in name
            for layer in checkpoint["layers"].values()
            for name in layer
        ),
        "all_fresh_reconstructions_exact": all(
            item["fresh_reconstruction_exact"] for item in per_linear
        ),
        "logical_effective_bpp": total_bits / max(total_weights, 1),
        "residual_parameters": sum(item["residual_parameters"] for item in per_linear),
        "residual_bits": sum(item["residual_bits"] for item in per_linear),
        "ppl": ppl,
        "ppl_split": args.ppl_split,
        "evaluation_seconds": time.time() - started,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "per_linear": per_linear,
        "completed": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, output)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
