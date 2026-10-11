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
    return f"""# LLaMA Vector-GSQ 与 QTIP 同协议对比报告

生成时间：{time.strftime('%Y-%m-%d %H:%M:%S %z')}

## 实验目的

在同一 LLaMA 基座、WikiText2 测试集和六项零样本任务上，比较 Vector-GSQ 与 QTIP 的码率和质量。

## 实验原理

Vector-GSQ 重建项目逻辑 checkpoint 中每层七个 Linear；QTIP 使用其官方 Hugging Face 加载器。先校验基座模型、PPL 序列长度、数据划分和任务集合，再报告有符号差值（Vector-GSQ 减 QTIP）。PPL 越低越好，准确率越高越好。

## 实验配置

- 基座模型：`{protocol['model']}`
- PPL：WikiText2 `{protocol['ppl_split']}`，sequence length = {protocol['ppl_seqlen']}
- lm-eval：0-shot，任务为 {', '.join(protocol['tasks'])}
- 最大可比码率差：{rate['max_bpp_gap']:.6f} bit/weight

## 实验结果

| 指标 | Vector-GSQ | QTIP | Vector-GSQ - QTIP |
|---|---:|---:|---:|
| Effective bit/weight | {rate['vector_gsq_effective_bpp']:.6f} | {rate['qtip_effective_bpp']:.6f} | {rate['vector_gsq_effective_bpp'] - rate['qtip_effective_bpp']:+.6f} |
| WikiText2 PPL | {ppl['vector_gsq']:.6f} | {ppl['qtip']:.6f} | {ppl['vector_gsq_minus_qtip']:+.6f} |
| 六任务宏平均准确率 | {accuracy['vector_gsq_macro_average']:.6f} | {accuracy['qtip_macro_average']:.6f} | {accuracy['vector_gsq_minus_qtip']:+.6f} |

码率可比：**{rate['comparable']}**。质量判定：`{summary['verdict']}`。

## 分任务结果

| 任务 | Vector-GSQ | QTIP | Vector-GSQ - QTIP |
|---|---:|---:|---:|
{task_rows}

## 结果来源

- Vector-GSQ PPL：`{sources['vector_ppl']}`
- Vector-GSQ lm-eval：`{sources['vector_lm_eval']}`
- QTIP 汇总：`{sources['qtip_summary']}`
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
