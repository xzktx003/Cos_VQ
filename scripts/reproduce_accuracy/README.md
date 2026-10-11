# Per-model accuracy reproduction

Each launcher runs the frozen V19 + E2E pipeline for one published model and
finishes with WikiText-2 test perplexity (`seqlength=2048`) plus zero-shot
ARC-Challenge, ARC-Easy, HellaSwag, LAMBADA, PIQA, and WinoGrande accuracy.

Run from any directory. Previewing a run is non-destructive:

```bash
./scripts/reproduce_accuracy/qwen3_4b.sh \
  --run-stamp 20260920_120000 --dry-run
```

Remove `--dry-run` to execute. The launchers accept every option exposed by
`python -m scripts.reproduce_v19_e2e`, including `--model-path`,
`--cuda-visible-devices`, and pinned dataset revisions. For Llama-3.1, use
`--reference-model-path` when the reference tokenizer snapshot is local.

| Published model | Launcher |
| --- | --- |
| Qwen3-4B | `qwen3_4b.sh` |
| Qwen3-8B | `qwen3_8b.sh` |
| Qwen3-14B | `qwen3_14b.sh` |
| Qwen3-32B | `qwen3_32b.sh` |
| Llama-2-7B | `llama2_7b.sh` |
| Llama-3.1-8B-Instruct | `llama3_1_8b_instruct.sh` |

Generated caches, checkpoints, logs, raw evaluation JSON, and reports remain
under `code/Cos_VQ/experiments/results/` and `code/Cos_VQ/report/`.
Historical experiment artifacts elsewhere in the parent research repository
are source records: these launchers never move, overwrite, or delete them.
