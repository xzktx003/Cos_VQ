"""Build a rate-and-protocol matched Vector-GSQ versus scalar-GSQ Gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def compare_gate_results(vector_summary, original_result, original_bpp):
    if not original_result.get("completed"):
        raise ValueError("Original GSQ PPL result is not completed")
    if vector_summary.get("model") != original_result.get("model"):
        raise ValueError("Vector and original GSQ model paths do not match")
    vector_seqlen = vector_summary.get("dataset", {}).get("ppl_seqlength")
    original_seqlen = original_result.get("ppl_seqlength")
    if vector_seqlen != 2048 or original_seqlen != 2048:
        raise ValueError("Both PPL results must use seqlength 2048")
    if original_result.get("dataset") != "wikitext2_test":
        raise ValueError("Original GSQ result must use WikiText2 test")
    if original_result.get("quantized_prefix_layers") != 1:
        raise ValueError("Original GSQ Gate must quantize exactly one prefix layer")
    vector_layers = vector_summary.get("shared_vector_config", {}).get("max_layers", 1)
    if vector_layers != 1:
        raise ValueError("Vector-GSQ Gate must quantize exactly one prefix layer")

    candidate = vector_summary["rate_fair_candidate"]
    vector_ppl = float(candidate["prefix1_ppl"])
    scalar_ppl = float(original_result["ppl"])
    vector_bpp = float(candidate["logical_bpp"])
    original_bpp = float(original_bpp)
    if not all(
        math.isfinite(value) and value > 0
        for value in (vector_ppl, scalar_ppl, vector_bpp, original_bpp)
    ):
        raise ValueError("PPL and bpp values must be finite and positive")

    rate_pass = vector_bpp <= original_bpp + 1e-12
    quality_pass = vector_ppl < scalar_ppl
    return {
        "comparison": "structure_aware_vector_gsq_vs_original_scalar_gsq",
        "model": vector_summary["model"],
        "dataset": "wikitext2_test",
        "ppl_seqlength": 2048,
        "quantized_prefix_layers": 1,
        "protocol_matched": True,
        "vector_method": candidate["method"],
        "vector_logical_bpp": vector_bpp,
        "original_gsq_logical_bpp": original_bpp,
        "vector_ppl": vector_ppl,
        "original_gsq_ppl": scalar_ppl,
        "ppl_delta_vector_minus_original": vector_ppl - scalar_ppl,
        "relative_ppl_delta_percent": (vector_ppl / scalar_ppl - 1.0) * 100.0,
        "vector_rate_not_higher": rate_pass,
        "vector_ppl_better": quality_pass,
        "vector_exceeds_original_gsq": rate_pass and quality_pass,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-summary", required=True)
    parser.add_argument("--original-ppl", required=True)
    parser.add_argument("--original-bpp", type=float, default=2.125)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    vector = json.loads(Path(args.vector_summary).read_text(encoding="utf-8"))
    original = json.loads(Path(args.original_ppl).read_text(encoding="utf-8"))
    result = compare_gate_results(vector, original, args.original_bpp)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
