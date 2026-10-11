"""Build a model-specific FineWeb-Edu token cache for LLaMA quantization."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from experiments.llama_qtip.protocol import infer_model_family
from src.data.dataset import make_concat_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequences", type=int, default=4736)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=1234)
    parser.add_argument("--shuffle-buffer-size", type=int, default=100000)
    parser.add_argument("--dataset-revision", default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sequences <= 0 or args.sequence_length <= 0:
        raise ValueError("Cache shape must be positive")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if infer_model_family(config) != "llama":
        raise ValueError("The token cache builder only accepts LLaMA models")
    identity = (
        f"sample-10BT|{args.model}|len={args.sequence_length}|n={args.sequences}|"
        f"seed={args.seed}|shuffle={args.shuffle_seed}|"
        f"buffer={args.shuffle_buffer_size}|revision={args.dataset_revision}"
    )
    output = Path(args.output).resolve()
    if output.exists():
        payload = torch.load(output, map_location="cpu", weights_only=True)
        if payload.get("identity") != identity:
            raise ValueError(f"Existing cache identity mismatch: {output}")
        expected_shape = (args.sequences, args.sequence_length)
        if tuple(payload["input_ids"].shape) != expected_shape:
            raise ValueError(
                f"Existing cache shape mismatch: got {tuple(payload['input_ids'].shape)}, "
                f"expected {expected_shape}"
            )
        print(output)
        return

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
        revision=args.dataset_revision,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
    chunks = make_concat_chunks(
        dataset,
        tokenizer,
        args.sequence_length,
        args.sequences,
        seed=args.shuffle_seed,
    )
    if len(chunks) != args.sequences:
        raise RuntimeError(f"FineWeb stream produced only {len(chunks)} chunks")
    payload = {
        "identity": identity,
        "input_ids": torch.stack(chunks),
        "metadata": {
            "dataset": "HuggingFaceFW/fineweb-edu",
            "dataset_config": "sample-10BT",
            "dataset_split": "train",
            "dataset_revision": args.dataset_revision,
            "tokenizer_model": str(Path(args.model).resolve()),
            "sequence_length": args.sequence_length,
            "sequences": args.sequences,
            "seed": args.seed,
            "shuffle_seed": args.shuffle_seed,
            "shuffle_buffer_size": args.shuffle_buffer_size,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    print(output)


if __name__ == "__main__":
    main()
