# CoS-VQ

CoS-VQ is a self-contained reproduction package for shared-scale codeword optimization and end-to-end LLM quantization. The implementation uses the internal method name **V19 + E2E**. It includes GSQ/NoWag baselines, calibration-cache adapters, V19 scale and codeword-assignment optimization, fine-tuning with fixed discrete assignments, checkpoint selection, and evaluation on WikiText-2 and six `lm-eval` tasks.

## Documentation

- [V19 + E2E reproduction guide](docs/V19_E2E_REPRODUCTION.md): method stages, frozen protocol, commands for all six models, outputs, and completion criteria.
- [Per-model reproduction launchers](scripts/reproduce_accuracy/README.md) (English): a launcher for each supported model.
- [Artifacts and historical results](docs/ARTIFACTS_AND_RESULTS.md): package contents, generated artifacts, and retention rules.
- [Six-model reproduction protocol](configs/reproduction/v19_e2e_six_models.json): the single source of truth for experiment parameters.

## Get the code

```bash
git clone https://github.com/xzktx003/Cos_VQ.git
cd Cos_VQ
```

The repository is public and can be cloned without repository access approval. In the parent research workspace, this package is located at `code/Cos_VQ/`. Run the commands below from the package root.

## Environment setup

You need Python 3.10 or newer, a CUDA GPU with BF16 support, `uv`, and access to Hugging Face Hub. Llama models may require approval on Hugging Face: accept the applicable model terms and authenticate locally before running. The package does not store access tokens.

```bash
uv sync --frozen
```

The included `pyproject.toml` and `uv.lock` describe the current code snapshot. Historical experiments used Torch 2.4, Transformers 4.52.4, and `lm-eval` 0.4.4; the current lockfile comes from a newer environment. Bitwise reproduction of historical metrics has not been established with this lockfile. Release validation covered static checks, command previews, and CPU unit tests; it did not install the runtime dependencies or rerun the full CUDA experiments.

## Preview the pipeline

Start with a single-model dry run:

```bash
uv run --frozen bash scripts/reproduce_accuracy/qwen3_4b.sh \
  --run-stamp 20261012_120000 --dry-run
```

The pipeline's `--dry-run` option prints the complete command plan without downloading models or datasets, launching training or evaluation, or creating result directories. When used through `uv run`, `uv` may first prepare or synchronize the Python environment. Once that environment is installed, a preview without environment synchronization can be run with:

```bash
.venv/bin/python -m scripts.reproduce_v19_e2e \
  --model qwen3-4b --run-stamp 20261012_120000 --dry-run
```

To preview all six models:

```bash
uv run --frozen python -m scripts.reproduce_v19_e2e \
  --model all --run-stamp 20261012_130000 --dry-run
```

## Run the full pipeline

After checking the command plan, model access, and GPU resources, remove `--dry-run` to execute the pipeline. For example:

```bash
uv run --frozen bash scripts/reproduce_accuracy/qwen3_4b.sh \
  --run-stamp 20261012_140000 --cuda-visible-devices 0
```

Choose a fresh timestamp in `YYYYMMDD_HHMMSS` format for each run. The orchestrator refuses to overwrite an existing campaign directory.

Each model follows the frozen protocol in `configs/reproduction/v19_e2e_six_models.json`:

1. Resolve model snapshots and freeze dataset revisions.
2. Build calibration and training caches.
3. Generate the Vector-GSQ baseline checkpoint.
4. Run scale-only and V19 candidate optimization, then select the V19 checkpoint using held-out development NLL.
5. Fine-tune the baseline and V19 checkpoints with fixed discrete assignments, and select an endpoint using development NLL.
6. Evaluate the selected endpoints and generate a reproduction report.

Models run sequentially when `--model all` is selected. The orchestrator records model and dataset revisions, configuration hashes, commands, logs, and endpoint results.

### Supported models and options

The package provides launchers for Qwen3-4B, Qwen3-8B, Qwen3-14B, Qwen3-32B, Llama-2-7B, and Llama-3.1-8B-Instruct. See the [launcher table](scripts/reproduce_accuracy/README.md) for their filenames. The launchers can be invoked from any working directory and forward their arguments to the orchestrator.

- `--model`: select `all` or one model key from the configuration.
- `--model-path`: use a local model snapshot for a single-model run.
- `--reference-model-path`: supply the reference tokenizer snapshot for Llama-3.1.
- `--cuda-visible-devices`: select physical GPU indices.
- `--split-gpus`: use exactly two visible GPUs for V19 development scoring and final evaluation. Quantization, V19 optimization, and E2E fine-tuning use one visible GPU according to the protocol.

Cache sizes and training budgets are defined per model. Llama-3.1 additionally builds a FineWeb-Edu cache with a C4 guard, calibrates task-local assignments, and enables group-scale E2E fine-tuning. Qwen3-8B and Llama-3.1 also include a scale-to-E2E branch. V19 selection compares the standard and wide candidates using development NLL. E2E selection compares two learning rates and includes the initial checkpoint as an eligible endpoint.

### Evaluation protocol

Final evaluation uses WikiText-2 **test** perplexity with `seqlength=2048` and zero-shot accuracy on:

- ARC-Challenge
- ARC-Easy
- HellaSwag
- LAMBADA OpenAI
- PIQA
- WinoGrande

The six-model configuration defines the evaluation parameters. Raw task metrics and endpoint summaries are written as JSON.

### Models and dataset revisions

RedPajama, FineWeb-Edu, C4, and WikiText-2 are fetched from Hugging Face dataset repositories by default. At the start of an actual run, mutable dataset references are resolved to immutable commit SHAs and recorded in the campaign manifest. Explicit revisions can be provided with `--redpajama-revision`, `--fineweb-revision`, `--c4-revision`, and `--wikitext-revision`.

Qwen3-4B and Qwen3-8B model revisions are pinned in the configuration. Other model references currently use `main`; their resolved snapshot commits are recorded in the manifest. To reuse the same model snapshot, pass its local path through `--model-path` for a single-model run.

## Outputs and historical results

By default, generated calibration and training caches, checkpoints, evaluation JSON, and logs are stored under:

```text
experiments/results/v19_e2e/<RUN_STAMP>/
```

The automatic report is written to:

```text
report/<RUN_STAMP>_v19_e2e_reproduction.md
```

Downloaded source models and datasets use the Hugging Face cache settings.

Large runtime artifacts are excluded from Git. Compact reference results and release-validation records are included separately. The [historical six-model result snapshot](experiments/results/reference/20260914_122219_v19_e2e_six_model_core_aggregate.json) contains archived results from September 14, 2026; it was not generated by the release-validation run. Historical evidence paths in that JSON are provenance records, not prerequisites for a new pipeline run.

Existing experiment artifacts in the parent research workspace are retained in their original locations. New pipeline runs do not move, overwrite, or delete them.

## Checks and tests

Run CPU unit tests from a development environment with `pytest` installed:

```bash
python -m pytest -q
```

For the reproduction-orchestration checks only:

```bash
python -m pytest -q tests/test_reproduction_pipeline.py
```

`pytest` is a development tool and is not included in the runtime dependencies. Four tests that depend on one-off shell launchers from the original research workspace are skipped in this standalone package. The six-model orchestration is covered by `test_reproduction_pipeline.py`.

CPU tests and dry runs validate the command plan and package structure. Full CUDA training, dataset downloads, WikiText-2 evaluation, and `lm-eval` accuracy measurements require a separate experiment run.

## License

The root [LICENSE](LICENSE) applies to the corresponding project files. Vendored code under `third_party/NoWag/` is covered by its own GPL-3.0 license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Models and datasets retain their upstream licenses and access terms.
