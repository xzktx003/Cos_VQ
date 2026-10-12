# Code And Result Retention Notes

## Public Package Boundary

`Cos_VQ` contains the algorithm implementation, configuration, vendored
NoWag code, tests, model-level launchers, and evaluation entry points needed
to reproduce V19 + E2E. `tests/test_reproduction_pipeline.py` traverses the
six-model command graph and verifies that every invoked Python module is
packaged.

## New Run Artifacts

Pipeline-generated experiment artifacts are kept inside this package. Source model
and dataset downloads use the configured Hugging Face cache:

- `experiments/results/v19_e2e/<RUN_STAMP>/`: caches, checkpoints,
  selections, raw evaluation JSON files, and logs.
- `report/<RUN_STAMP>_v19_e2e_reproduction.md`: the report generated
  automatically from the manifest and summaries.
- `campaign_manifest.json`: configuration hashes, resolved model and dataset
  revisions, step-by-step commands, status, and timing.

Git ignore rules exclude large caches, checkpoints, training directories, and logs;
compact JSON and Markdown summaries remain trackable. Archive at least
the manifest, frozen configuration, selections, final checkpoints, raw
evaluation JSON files, summaries, logs, and generated report.

## Historical Results From The Parent Research Repository

Historical results under the parent repository paths
`code/GSQ_nowag_d1_20260716_015355/experiments/results/`, root-level
`experiments/results/`, and `report/` remain in place. This public package does
not move, overwrite, or delete them, and a fresh reproduction does not depend on
absolute paths from the parent repository.

Activation caches that were removed under an explicit cleanup instruction are
not part of the current retained set and cannot be recovered from Git. Existing
code, checkpoints, result JSON files, and reports should continue to be kept.
Before removing rebuildable caches in the future, first confirm that final
checkpoints, evaluation JSON files, manifests, and reports have been fully
archived.

## Historical Compact Result Snapshot

[Six-model core result JSON](../experiments/results/reference/20260914_122219_v19_e2e_six_model_core_aggregate.json)
is a byte-preserving copy of the 2026-09-14 aggregate file, retaining the
original evidence paths and SHA256 values. Historical absolute paths inside the
JSON are provenance records only; new end-to-end runs do not depend on those
paths. This snapshot is not a result from rerunning the experiment during this
release.
