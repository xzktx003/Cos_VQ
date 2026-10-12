"""Build JSON and Markdown comparisons for matched Vector-GSQ/QTIP runs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from experiments.llama_qtip.protocol import build_comparison_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-ppl", required=True)
    parser.add_argument("--vector-lm-eval", required=True)
    parser.add_argument("--qtip-summary", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--max-bpp-gap", type=float, default=0.10)
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def render_report(summary: dict[str, Any], sources: dict[str, str]) -> str:
    protocol = summary["protocol"]
    rate = summary["rate"]
    ppl = summary["ppl"]
    accuracy = summary["accuracy"]
    task_rows = "\n".join(
        f"| {task} | {values['vector_gsq']:.6f} | {values['qtip']:.6f} | "
        f"{values['vector_gsq_minus_qtip']:+.6f} |"
        for task, values in accuracy["per_task"].items()
    )
    return f"""# LLaMA Vector-GSQ and QTIP comparison under a matched protocol

Generated at: {time.strftime('%Y-%m-%d %H:%M:%S %z')}

## Experiment objective

Compare the bit rate and quality of Vector-GSQ and QTIP using the same LLaMA base model, WikiText-2 test split, and six zero-shot tasks.

## Experiment principle

Vector-GSQ reconstructs the seven linear modules in each layer from the logical checkpoint; QTIP uses its official Hugging Face loader. Validate the base model, PPL sequence length, data split, and task set before reporting signed differences (Vector-GSQ minus QTIP). Lower PPL and higher accuracy indicate better quality.

## Experiment configuration

- Base model: `{protocol['model']}`
- PPL: WikiText-2 `{protocol['ppl_split']}`; sequence length = {protocol['ppl_seqlen']}
- lm-eval: zero-shot tasks {', '.join(protocol['tasks'])}
- Maximum comparable bit-rate gap: {rate['max_bpp_gap']:.6f} bit/weight

## Experiment results

| Metric | Vector-GSQ | QTIP | Vector-GSQ - QTIP |
|---|---:|---:|---:|
| Effective bit/weight | {rate['vector_gsq_effective_bpp']:.6f} | {rate['qtip_effective_bpp']:.6f} | {rate['vector_gsq_effective_bpp'] - rate['qtip_effective_bpp']:+.6f} |
| WikiText2 PPL | {ppl['vector_gsq']:.6f} | {ppl['qtip']:.6f} | {ppl['vector_gsq_minus_qtip']:+.6f} |
| Six-task macro accuracy | {accuracy['vector_gsq_macro_average']:.6f} | {accuracy['qtip_macro_average']:.6f} | {accuracy['vector_gsq_minus_qtip']:+.6f} |

Comparable bit rates: **{rate['comparable']}**. Quality verdict: `{summary['verdict']}`.

## Per-task results

| Task | Vector-GSQ | QTIP | Vector-GSQ - QTIP |
|---|---:|---:|---:|
{task_rows}

## Result sources

- Vector-GSQ PPL: `{sources['vector_ppl']}`
- Vector-GSQ lm-eval: `{sources['vector_lm_eval']}`
- QTIP summary: `{sources['qtip_summary']}`
"""


def main() -> None:
    args = parse_args()
    sources = {
        "vector_ppl": str(Path(args.vector_ppl).resolve()),
        "vector_lm_eval": str(Path(args.vector_lm_eval).resolve()),
        "qtip_summary": str(Path(args.qtip_summary).resolve()),
    }
    summary = build_comparison_summary(
        load_json(args.vector_ppl),
        load_json(args.vector_lm_eval),
        load_json(args.qtip_summary),
        max_bpp_gap=args.max_bpp_gap,
    )
    summary["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary["sources"] = sources
    atomic_text(
        Path(args.output_json),
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_text(Path(args.output_report), render_report(summary, sources))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
