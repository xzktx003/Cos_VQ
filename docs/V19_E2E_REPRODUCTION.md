# V19 + E2E Experiment And Accuracy Reproduction Guide

This document explains how to regenerate V19 + E2E checkpoints from the public
code and reproduce WikiText-2 PPL plus six zero-shot accuracy tasks under the
paper protocol. The only machine-readable source of the protocol is
`configs/reproduction/v19_e2e_six_models.json`; the only command orchestration
implementation is `scripts/reproduce_v19_e2e.py`.

## 1. Reproduction Scope

| Model key | Hugging Face model | Layers | Standalone launcher |
| --- | --- | ---: | --- |
| `qwen3-4b` | `Qwen/Qwen3-4B` | 36 | `scripts/reproduce_accuracy/qwen3_4b.sh` |
| `qwen3-8b` | `Qwen/Qwen3-8B` | 36 | `scripts/reproduce_accuracy/qwen3_8b.sh` |
| `qwen3-14b` | `Qwen/Qwen3-14B` | 40 | `scripts/reproduce_accuracy/qwen3_14b.sh` |
| `qwen3-32b` | `Qwen/Qwen3-32B` | 64 | `scripts/reproduce_accuracy/qwen3_32b.sh` |
| `llama2-7b` | `meta-llama/Llama-2-7b-hf` | 32 | `scripts/reproduce_accuracy/llama2_7b.sh` |
| `llama3.1-8b-instruct` | `meta-llama/Llama-3.1-8B-Instruct` | 32 | `scripts/reproduce_accuracy/llama3_1_8b_instruct.sh` |

Each full run performs quantization, V19 search, E2E fine-tuning, endpoint
selection, and final evaluation. It does not read values from the paper tables.

## 2. Method Pipeline

Each model runs the following stages in order:

1. **Freeze inputs and data versions**: resolve the model snapshot, freeze the
   Hugging Face dataset revisions required for the run, and write the resolved
   commits into `campaign_manifest.json`.
2. **Build calibration caches**: Qwen3 and Llama-2 use RedPajama; Llama-3.1 uses
   FineWeb-Edu and an independent C4 guard to check tokenizer/cache identity.
3. **Vector-GSQ baseline**: run NoWag-d6, Hessian-aware Vector-GPTQ, and
   block-level Vector-GSQ to create the baseline checkpoint. Llama-3.1 also runs
   task-local assignment calibration.
4. **V19 candidate optimization**: run `scale_only`, `standard`, and `wide`
   branches independently from the same baseline; select the V19 endpoint
   between standard and wide using development NLL that was not used for
   optimization.
5. **Fixed discrete-assignment E2E**: train baseline and V19 for 3 epochs with
   learning rates `5e-5` and `1e-4`; Qwen3-8B and Llama-3.1 also run a
   scale-to-E2E branch. The initial endpoints are included in development NLL
   selection to guard against fine-tuning regressions.
6. **Final evaluation**: evaluate endpoints only after selection is complete,
   then write raw JSON files, sample logs, and summaries.

The main implementation map is:

| Stage | Code entry point |
| --- | --- |
| Orchestration, manifest, generated report | `scripts/reproduce_v19_e2e.py` |
| Qwen/Llama cache adapters | `experiments/qwen_qtip/`, `experiments/llama_qtip/` |
| Vector-GSQ baseline | `experiments/nowag_d6/run_prefix_vector_gsq.py` |
| Llama-3.1 task assignment | `experiments/nowag_d6/calibrate_task_local_assignments.py` |
| V19 model-level optimization | `experiments/nowag_d6/v19_model_driver.py` |
| V19 independent development-set selection | `experiments/nowag_d6/v19_validation_score.py` |
| Fixed-assignment E2E | `experiments/nowag_d6/finetune_hard_codebook_e2e.py` |
| E2E endpoint selection | `experiments/nowag_d6/v19_e2e_selection.py` |
| PPL and six-task accuracy | `experiments/remote_runner/evaluate_logical_checkpoint_lm_eval.py` |

## 3. Frozen Experiment Protocol

- Quantization: d=6, 2 bits/weight, BF16, seed 0.
- V19: 128x2048 calibration, block-Frobenius majorizer, 1 pass.
- Standard: 8 active positions, 4 candidates per ranking.
- Wide: 16 active positions, 8 candidates per ranking.
- E2E: context 4096, batch 4, 3 epochs, gradient clip 1.0.
- PPL: WikiText-2 raw test, `seqlength=2048`.
- Accuracy: 0-shot, batch size 1, seed 0, 1000 bootstrap iterations.
- Tasks: ARC-Challenge, ARC-Easy, HellaSwag, LAMBADA OpenAI, PIQA, WinoGrande.
- Aggregate accuracy: macro average over the six tasks; raw task metrics are
  preserved in the evaluation JSON files.

All model-specific differences in calibration counts, concurrency, E2E data
volume, scale layout, and related parameters are recorded in
`configs/reproduction/v19_e2e_six_models.json`. Do not copy or hand-edit them
in shell scripts.

## 4. Environment And Data Preparation

```bash
cd Cos_VQ
uv sync --frozen
source .venv/bin/activate
```

You need Python 3.10+, CUDA GPUs, enough local storage for models and
checkpoints, and Hugging Face Hub access. Llama models require accepting the
upstream license before use. Access tokens should be supplied only through local
Hugging Face login or environment variables; do not write them into
configuration files, scripts, manifests, or logs.

When `--model-path` is not provided, the orchestrator resolves model snapshots
with `huggingface_hub.snapshot_download(...)`, so source model downloads use the
default Hugging Face cache unless your environment overrides it.

Run a dry run first. It prints the complete command plan without downloading
data, creating outputs, or starting GPU work:

```bash
./scripts/reproduce_accuracy/qwen3_4b.sh \
  --run-stamp 20260920_120000 \
  --dry-run
```

## 5. Full Runs

Single model, automatically resolving the fixed snapshot:

```bash
./scripts/reproduce_accuracy/qwen3_8b.sh \
  --run-stamp 20260920_120000 \
  --cuda-visible-devices 0
```

Single model, using a local snapshot:

```bash
./scripts/reproduce_accuracy/qwen3_14b.sh \
  --run-stamp 20260920_130000 \
  --model-path /absolute/path/to/Qwen3-14B \
  --cuda-visible-devices 0
```

Llama-3.1 with a local model and reference tokenizer:

```bash
./scripts/reproduce_accuracy/llama3_1_8b_instruct.sh \
  --run-stamp 20260920_140000 \
  --model-path /absolute/path/to/Llama-3.1-8B-Instruct \
  --reference-model-path /absolute/path/to/Meta-Llama-3-8B \
  --cuda-visible-devices 0
```

All six models, sequentially:

```bash
python -m scripts.reproduce_v19_e2e \
  --model all \
  --run-stamp 20260920_150000 \
  --cuda-visible-devices 0
```

Training stages always use logical `cuda:0`; use `--cuda-visible-devices` to
select the physical GPU. `--split-gpus` requires exactly two visible GPUs and
splits only V19 development scoring and final evaluation. It does not
automatically distribute quantization, V19 training, or E2E training across two
GPUs.

## 6. Output Structure And Success Criteria

The default run directory is
`experiments/results/v19_e2e/<RUN_STAMP>/`:

```text
experiments/results/v19_e2e/<RUN_STAMP>/
├── campaign_manifest.json
└── <MODEL_KEY>/
    ├── data/
    ├── baseline/
    ├── checkpoints/
    ├── v19/{scale_only,standard,wide}/
    ├── training/<BRANCH>/<LEARNING_RATE>/
    ├── selections/{v19,baseline_e2e,v19_e2e,...}.json
    ├── evaluation/<ENDPOINT>/summary.json
    └── logs/
```

A successful run must satisfy all of the following:

1. The top-level `status` in `campaign_manifest.json` is `completed`.
2. The corresponding model record has `status` set to `completed`.
3. Every selection JSON has `completed` set to `true` and points to an existing
   checkpoint.
4. `evaluation/<ENDPOINT>/summary.json` exists.
5. The generated report `report/<RUN_STAMP>_v19_e2e_reproduction.md` exists.

Regular models evaluate five endpoints: `baseline`, `scale`, `v19`,
`baseline_e2e`, and `v19_e2e`. Qwen3-8B and Llama-3.1 also include
`scale_e2e`. Paper numbers should be read from summaries or raw JSON files, not
copied manually from terminal logs.

## 7. Failure, Rerun, And Storage Rules

- The orchestrator refuses to overwrite an existing `<RUN_STAMP>`, so success
  or failure records are not silently lost.
- The current public orchestrator does not promise in-place resume for a
  partially completed campaign. After a failure, keep the manifest and logs,
  fix the issue, rerun with a new timestamp, and record the previous failed
  directory in the report.
- `--output-root` is allowed, but the target must still be under this package's
  `experiments/results/` directory.
- Do not commit base models, dataset caches, large checkpoints, or complete run
  logs to Git.
- Do not delete raw JSON files, selections, manifests, configuration hashes, or
  generated reports; together they form the audit trail for the paper results.
- Activation caches can be rebuilt, but keep them until the relevant run is
  complete and archived if you need checkpoint-resume support.

## 8. Verify The Public Package

```bash
bash -n scripts/reproduce_accuracy/*.sh
python -m pytest -q tests/test_reproduction_pipeline.py
python -m compileall -q scripts experiments src
```

Full CUDA reproduction is long-running and depends on model authorization, data
network access, and GPUs. The CPU contract tests above do not replace it. They
verify the command graph, protocol parameters, module packaging, and safety
boundaries; final paper metrics still come from full run artifacts.
