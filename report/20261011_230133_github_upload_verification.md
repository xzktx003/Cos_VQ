# Cos_VQ Private Repository Release Verification

Date and time: 2026-10-11T23:01:34.948172+08:00.

## Purpose

Rename the standalone reproduction package from `code/code2open` to
`code/Cos_VQ`, prepare it for release to the private `xzktx003/Cos_VQ`
repository, and confirm that the renamed end-to-end entry point can generate a
complete command plan.

## Principle

Paths are derived dynamically from the package root, so renaming the directory
does not require changing algorithm module names. The Python distribution name
is standardized as `cos-vq`; the editable entry in the lock file is synchronized
without reselecting dependency versions.

## Configuration And Tools

- Python: 3.11.0 | packaged by conda-forge | (main, Jan 14 2023, 12:28:47)
  [GCC 11.3.0].
- Models: Qwen3-4B/8B/14B/32B, Llama-2-7B, and Llama-3.1-8B-Instruct; commands
  were previewed only, with no weights loaded.
- Configuration: `configs/reproduction/v19_e2e_six_models.json`.
- Tools: pytest, bash syntax checks, Python AST/tomllib, uv offline lockfile
  check, Git, and GitHub CLI.
- Verification parameters: `--dry-run` for all six launchers, executed from
  outside the package directory; PPL2048 and six-task evaluation parameters are
  checked by the orchestration tests.
- CUDA training and accuracy evaluation were not executed in this verification.

## Steps And Results

1. Synchronized the directory name, README paths, Python distribution name, and
   lock file.
2. `tests/test_reproduction_pipeline.py`: 12 tests passed.
3. Syntax checks passed for seven shell scripts, and dry runs passed for all six
   model launchers.
4. Python source files under `scripts/`, `experiments/`, and `src/` parsed
   successfully as AST.
5. `uv lock --check --offline` passed; the project name and lock file are
   consistent.
6. The historical six-model result JSON was recorded as matching the original
   file's SHA256; the result belongs to the historical experiment from
   2026-09-14.

Verification JSON:
`experiments/results/upload_validation/20261011_230133_github_upload_validation.json`.

## Release Scope

Algorithm source, six-model configuration, orchestrator and single-model
launcher scripts, tests, dependency lock file, license, documentation,
historical compact results, and this verification record. Runtime caches,
weights, checkpoints, test temporary directories, and complete logs are ignored
by Git. GitHub remote privacy attributes and commit hashes are recorded in
`docs/version_uploads.md`.

## Remote Release Check

Check time: 2026-10-11T23:04:40.736901+08:00. The GitHub repository
`xzktx003/Cos_VQ` was private, with default branch `main`. The first content
commit `5e3fd3dd17c33259e85dd59f7e75644f35b24e38` matched the remote, and the
remote file tree contained 281 files. The reproduction entry points,
configuration, tests, lock file, and historical compact result were present.
See `docs/version_uploads.md` for commit and provenance records.
