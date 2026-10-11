"""Compare full-model structure-aware Vector-GSQ with original scalar GSQ."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _positive_finite(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def compare_full_results(
    vector_quantization,
    vector_ppl_result,
    original_ppl_result,
    original_bpp,
):
    """Return a strict full-model rate-and-PPL verdict."""
    for name, result in (
        ("Vector quantization", vector_quantization),
        ("Vector PPL", vector_ppl_result),
        ("original GSQ PPL", original_ppl_result),
    ):
        if not result.get("completed"):
            raise ValueError(f"{name} result is not completed")

    vector_arguments = vector_quantization.get("arguments", {})
    ppl_arguments = vector_ppl_result.get("arguments", {})
    vector_model = vector_arguments.get("model")
    if not vector_model or ppl_arguments.get("model") != vector_model:
        raise ValueError("Vector quantization and PPL model paths do not match")
    if original_ppl_result.get("model") != vector_model:
        raise ValueError("Vector and original GSQ model paths do not match")
    if ppl_arguments.get("checkpoint") != vector_arguments.get("checkpoint"):
        raise ValueError("Vector PPL does not evaluate the selected quantization checkpoint")

    expected_method = {
        "seed": 0,
        "stage": "gsq",
        "gsq_objective": "staged",
        "staged_joint_group_scales": True,
        "scale_group_size": 160,
        "scale_finetune_steps": 0,
        "block_polish_steps": 0,
        "vector_dim": 6,
        "bits_per_weight": 2,
    }
    mismatches = {
        key: {"expected": expected, "observed": vector_arguments.get(key)}
        for key, expected in expected_method.items()
        if vector_arguments.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"Vector result is not the pre-registered staged joint group160 method: "
            f"{mismatches}"
        )

    if int(vector_arguments.get("max_layers", -1)) != 32:
        raise ValueError("Vector quantization must request all 32 layers")
    vector_layers = vector_quantization.get("layers", [])
    if len(vector_layers) != 32:
        raise ValueError("Vector quantization must contain all 32 layer results")
    dense_teacher_layers = sum(
        layer.get("gsq", {}).get("teacher_reference") == "original_dense_network"
        for layer in vector_layers
    )
    if dense_teacher_layers != 32:
        raise ValueError(
            "Vector quantization must use the original dense teacher in all 32 layers"
        )
    if int(vector_ppl_result.get("quantized_layers", -1)) != 32:
        raise ValueError("Vector PPL must reconstruct all 32 layers")
    if int(original_ppl_result.get("quantized_prefix_layers", -1)) != 32:
        raise ValueError("Original GSQ PPL must load all 32 layers")
    if vector_ppl_result.get("quantized_linears") != vector_ppl_result.get(
        "expected_linears"
    ):
        raise ValueError("Vector PPL did not reconstruct every expected Linear")
    if not vector_ppl_result.get("all_fresh_reconstructions_exact"):
        raise ValueError("Vector logical checkpoint reconstruction is not exact")

    if int(ppl_arguments.get("ppl_seqlen", -1)) != 2048:
        raise ValueError("Vector PPL must use seqlength 2048")
    if int(original_ppl_result.get("ppl_seqlength", -1)) != 2048:
        raise ValueError("Original GSQ PPL must use seqlength 2048")
    if ppl_arguments.get("ppl_split") != "test" or vector_ppl_result.get(
        "ppl_split"
    ) != "test":
        raise ValueError("Vector PPL must use WikiText2 test")
    if original_ppl_result.get("dataset") != "wikitext2_test":
        raise ValueError("Original GSQ PPL must use WikiText2 test")

    vector_bpp = _positive_finite(
        "Vector logical bpp", vector_quantization.get("logical_effective_bpp")
    )
    reconstructed_bpp = _positive_finite(
        "Vector reconstructed logical bpp",
        vector_ppl_result.get("logical_effective_bpp"),
    )
    if not math.isclose(vector_bpp, reconstructed_bpp, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Vector quantization and reconstructed logical bpp differ")
    original_bpp = _positive_finite("Original GSQ logical bpp", original_bpp)
    vector_ppl = _positive_finite("Vector PPL", vector_ppl_result.get("ppl"))
    original_ppl = _positive_finite(
        "Original GSQ PPL", original_ppl_result.get("ppl")
    )

    rate_pass = vector_bpp <= original_bpp + 1e-12
    quality_pass = vector_ppl < original_ppl
    return {
        "comparison": "full_structure_aware_vector_gsq_vs_original_scalar_gsq",
        "model": vector_model,
        "dataset": "wikitext2_test",
        "ppl_seqlength": 2048,
        "quantized_layers": 32,
        "protocol_matched": True,
        "vector_method": "dense_teacher_staged_joint_group160_full_logical_gate",
        "vector_method_protocol": expected_method,
        "dense_teacher_layers": dense_teacher_layers,
        "vector_logical_bpp": vector_bpp,
        "original_gsq_logical_bpp": original_bpp,
        "vector_ppl": vector_ppl,
        "original_gsq_ppl": original_ppl,
        "ppl_delta_vector_minus_original": vector_ppl - original_ppl,
        "relative_ppl_delta_percent": (vector_ppl / original_ppl - 1.0) * 100.0,
        "vector_rate_not_higher": rate_pass,
        "vector_ppl_better": quality_pass,
        "vector_exceeds_original_gsq": rate_pass and quality_pass,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-quantization", required=True)
    parser.add_argument("--vector-ppl", required=True)
    parser.add_argument("--original-ppl", required=True)
    parser.add_argument("--original-bpp", type=float, default=2.125)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    result = compare_full_results(
        load(args.vector_quantization),
        load(args.vector_ppl),
        load(args.original_ppl),
        args.original_bpp,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
