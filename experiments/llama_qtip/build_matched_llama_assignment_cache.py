"""Build a matched Llama-3.1 cache from official FineWeb tokens plus C4 guards.

The source Vector-GSQ phase must see exactly the same first 4736 FineWeb rows as
the existing official-scale Llama-3 cache. Reuse is allowed only after proving
that the reference and target tokenizers have identical raw tokenization. Extra
rows come from the public C4 English validation split and are reserved for
downstream assignment validation/audit, so the source quantizer has never seen
them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src.data.dataset import _load_c4_from_arrow_cache, make_concat_chunks


INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def verify_raw_tokenizer_equivalence(reference: Any, target: Any) -> int:
    """Verify raw-text equivalence while ignoring unused reserved-token names."""
    if reference.bos_token_id != target.bos_token_id:
        raise ValueError("reference and target tokenizer BOS IDs differ")
    regular_vocab_size = int(reference.bos_token_id)
    reference_vocab = {
        token: token_id
        for token, token_id in reference.get_vocab().items()
        if token_id < regular_vocab_size
    }
    target_vocab = {
        token: token_id
        for token, token_id in target.get_vocab().items()
        if token_id < regular_vocab_size
    }
    if reference_vocab != target_vocab:
        raise ValueError("reference and target regular tokenizer vocabularies differ")
    reference_backend = json.loads(reference.backend_tokenizer.to_str())
    target_backend = json.loads(target.backend_tokenizer.to_str())
    reference_backend.pop("added_tokens", None)
    target_backend.pop("added_tokens", None)
    if reference_backend != target_backend:
        raise ValueError("reference and target raw tokenizer backends differ")
    probes = (
        "The quick brown fox jumps over the lazy dog.",
        "Quantization: 4096 vectors, α=0.25, 中文与English。",
        "def f(x):\n    return x * x\n",
    )
    for text in probes:
        for add_special_tokens in (False, True):
            if reference.encode(
                text, add_special_tokens=add_special_tokens
            ) != target.encode(text, add_special_tokens=add_special_tokens):
                raise ValueError("reference and target tokenizer encodings differ")
    return regular_vocab_size


def verify_cache_token_domain(
    input_ids: torch.Tensor, *, regular_vocab_size: int, bos_token_id: int
) -> None:
    """Allow shared regular IDs and BOS, but reject tokenizer-specific specials."""
    unsupported = input_ids[
        (input_ids >= regular_vocab_size) & (input_ids != bos_token_id)
    ]
    if unsupported.numel():
        values = torch.unique(unsupported).tolist()
        raise ValueError(f"cache contains unsupported special token IDs: {values}")


def combine_fineweb_and_guard_tokens(
    fineweb_payload: dict[str, Any],
    guard_chunks: Iterable[torch.Tensor],
    *,
    target_model: str,
    guard_dataset: str,
) -> dict[str, Any]:
    fineweb = fineweb_payload.get("input_ids")
    if not isinstance(fineweb, torch.Tensor) or fineweb.ndim != 2:
        raise ValueError("FineWeb cache must contain a rank-two input_ids tensor")
    if fineweb.dtype not in INTEGER_DTYPES:
        raise ValueError("FineWeb input_ids must use an integer dtype")
    sequence_length = int(fineweb.shape[1])
    chunks = list(guard_chunks)
    if not chunks:
        raise ValueError("at least one guard sequence is required")
    for chunk in chunks:
        if not isinstance(chunk, torch.Tensor) or chunk.ndim != 1:
            raise ValueError("each guard sequence must be a rank-one tensor")
        if int(chunk.numel()) != sequence_length:
            raise ValueError(
                f"guard sequence length {chunk.numel()} does not match {sequence_length}"
            )
        if chunk.dtype not in INTEGER_DTYPES:
            raise ValueError("guard tokens must use an integer dtype")
    guard = torch.stack([chunk.to(dtype=torch.int64) for chunk in chunks])
    combined = torch.cat([fineweb.to(dtype=torch.int64), guard], dim=0)
    fineweb_count = int(fineweb.shape[0])
    guard_count = int(guard.shape[0])
    return {
        "identity": (
            f"matched-llama-assignment|{target_model}|len={sequence_length}|"
            f"fineweb={fineweb_count}|guard={guard_count}|dataset={guard_dataset}"
        ),
        "input_ids": combined,
        "target_model": target_model,
        "source_fineweb_identity": fineweb_payload.get("identity"),
        "guard_dataset": guard_dataset,
        "fineweb_rows": [0, fineweb_count - 1],
        "guard_rows": [fineweb_count, fineweb_count + guard_count - 1],
    }


def verify_prefix_extension(
    prefix_payload: dict[str, Any], extended_payload: dict[str, Any]
) -> int:
    """Require every previously exposed cache row to remain bit-exact."""
    prefix = prefix_payload.get("input_ids")
    extended = extended_payload.get("input_ids")
    if not isinstance(prefix, torch.Tensor) or prefix.ndim != 2:
        raise ValueError("prefix cache must contain rank-two input_ids")
    if not isinstance(extended, torch.Tensor) or extended.ndim != 2:
        raise ValueError("extended cache must contain rank-two input_ids")
    if prefix.dtype != extended.dtype or int(prefix.shape[1]) != int(extended.shape[1]):
        raise ValueError("extended cache must preserve prefix dtype and sequence length")
    prefix_rows = int(prefix.shape[0])
    if int(extended.shape[0]) <= prefix_rows:
        raise ValueError("extended cache must strictly extend the prefix cache")
    if not torch.equal(prefix, extended[:prefix_rows]):
        raise ValueError("extended cache prefix is not bit-exact")
    return prefix_rows


def load_c4_validation(*, source: str, revision: str):
    """Load all public C4 English validation shards or an explicitly local cache."""
    if source == "cache":
        _train, validation = _load_c4_from_arrow_cache()
        if validation is None:
            raise RuntimeError("local C4 validation Arrow cache is unavailable")
        return validation
    if source != "download":
        raise ValueError(f"Unsupported C4 source: {source!r}")
    return load_dataset(
        "allenai/c4",
        "en",
        split="validation",
        revision=revision,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--fineweb-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--guard-sequences", type=int, default=640)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--guard-seed", type=int, default=20260825)
    parser.add_argument("--expected-prefix-cache")
    parser.add_argument("--c4-source", choices=("download", "cache"), default="download")
    parser.add_argument("--c4-revision", default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.guard_sequences, args.sequence_length) <= 0:
        raise ValueError("guard count and sequence length must be positive")
    reference = AutoTokenizer.from_pretrained(
        args.reference_model, trust_remote_code=True, local_files_only=True
    )
    target = AutoTokenizer.from_pretrained(
        args.target_model, trust_remote_code=True, local_files_only=True
    )
    regular_vocab_size = verify_raw_tokenizer_equivalence(reference, target)

    fineweb_payload = torch.load(
        args.fineweb_cache, map_location="cpu", weights_only=True
    )
    fineweb = fineweb_payload.get("input_ids")
    if not isinstance(fineweb, torch.Tensor) or tuple(fineweb.shape) != (
        4736,
        args.sequence_length,
    ):
        raise ValueError(
            "source FineWeb cache must have exact official shape "
            f"(4736, {args.sequence_length})"
        )
    verify_cache_token_domain(
        fineweb,
        regular_vocab_size=regular_vocab_size,
        bos_token_id=int(target.bos_token_id),
    )

    validation = load_c4_validation(source=args.c4_source, revision=args.c4_revision)
    validation = validation.shuffle(seed=args.guard_seed)
    guard_chunks = make_concat_chunks(
        validation,
        target,
        args.sequence_length,
        args.guard_sequences,
        seed=args.guard_seed,
        add_eos=False,
    )
    if len(guard_chunks) != args.guard_sequences:
        raise RuntimeError(
            f"local C4 produced {len(guard_chunks)} guard sequences, "
            f"expected {args.guard_sequences}"
        )
    payload = combine_fineweb_and_guard_tokens(
        fineweb_payload,
        guard_chunks,
        target_model=str(Path(args.target_model).resolve()),
        guard_dataset=f"allenai/c4/en/validation@{args.c4_revision}",
    )
    if args.expected_prefix_cache is not None:
        prefix_payload = torch.load(
            args.expected_prefix_cache, map_location="cpu", weights_only=True
        )
        verify_prefix_extension(prefix_payload, payload)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = torch.load(output, map_location="cpu", weights_only=True)
        if existing.get("identity") != payload["identity"] or not torch.equal(
            existing.get("input_ids"), payload["input_ids"]
        ):
            raise ValueError(f"existing output has a different cache contract: {output}")
        print(output)
        return
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    print(output)


if __name__ == "__main__":
    main()
