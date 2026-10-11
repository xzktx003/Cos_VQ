"""Build a deterministic RP1T token cache with a Qwen3 or LLaMA tokenizer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from experiments.qwen_qtip.calibration_data import (
    CACHE_FORMAT,
    load_qtip_calibration_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hessian-devset-size", default=8192, type=int)
    parser.add_argument("--finetune-devset-size", default=384, type=int)
    parser.add_argument("--ctx-size", default=4096, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--documents-per-batch", default=128, type=int)
    parser.add_argument("--dataset-revision", default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    config = AutoConfig.from_pretrained(args.tokenizer_model)
    if config.model_type not in {"qwen3", "llama"}:
        raise ValueError(f"Expected qwen3 or llama tokenizer model, got {config.model_type!r}")
    expected = dict(
        model_type=config.model_type,
        vocab_size=int(config.vocab_size),
        ctx_size=args.ctx_size,
        hessian_devset_size=args.hessian_devset_size,
        finetune_devset_size=args.finetune_devset_size,
    )
    if output.is_file():
        load_qtip_calibration_cache(output, **expected)
        print(f"CALIBRATION_CACHE_SKIP {output}", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=True)
    dataset = load_dataset(
        "togethercomputer/RedPajama-Data-1T-Sample",
        split="train",
        revision=args.dataset_revision,
    )
    generator = torch.Generator().manual_seed(args.seed)
    concat_chunks: list[torch.Tensor] = []
    finetune_rows: list[torch.Tensor] = []
    concat_tokens = 0
    target_tokens = args.hessian_devset_size * args.ctx_size
    sampled_documents = 0

    while concat_tokens < target_tokens or len(finetune_rows) < args.finetune_devset_size:
        indices = torch.randint(
            len(dataset),
            (args.documents_per_batch,),
            generator=generator,
        ).tolist()
        texts = dataset[indices]["text"]
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False)
        for token_ids in encoded["input_ids"]:
            if not token_ids:
                continue
            tensor = torch.tensor(token_ids, dtype=torch.int32)
            if concat_tokens < target_tokens:
                concat_chunks.append(tensor)
                concat_tokens += tensor.numel()
            if len(finetune_rows) < args.finetune_devset_size and tensor.numel() >= args.ctx_size:
                finetune_rows.append(tensor[: args.ctx_size].clone())
        sampled_documents += len(texts)
        print(
            "CALIBRATION_PROGRESS "
            f"concat={min(concat_tokens, target_tokens)}/{target_tokens} "
            f"finetune={len(finetune_rows)}/{args.finetune_devset_size}",
            flush=True,
        )

    hessian_tokens = torch.cat(concat_chunks)[:target_tokens]
    hessian_tokens = hessian_tokens.view(args.hessian_devset_size, args.ctx_size)
    finetune_tokens = torch.stack(finetune_rows)
    metadata = {
        "format": CACHE_FORMAT,
        "dataset": "togethercomputer/RedPajama-Data-1T-Sample",
        "dataset_split": "train",
        "dataset_revision": args.dataset_revision,
        "tokenizer_model": str(Path(args.tokenizer_model).resolve()),
        "tokenizer_model_type": config.model_type,
        "tokenizer_vocab_size": int(config.vocab_size),
        "ctx_size": args.ctx_size,
        "hessian_devset_size": args.hessian_devset_size,
        "finetune_devset_size": args.finetune_devset_size,
        "seed": args.seed,
        "sampled_documents": sampled_documents,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(
        {
            "metadata": metadata,
            "hessian_tokens": hessian_tokens,
            "finetune_tokens": finetune_tokens,
        },
        temporary,
    )
    os.replace(temporary, output)
    load_qtip_calibration_cache(output, **expected)
    print(f"CALIBRATION_CACHE_DONE {output}", flush=True)


if __name__ == "__main__":
    main()
