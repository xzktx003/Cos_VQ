# Cos_VQ Release Record

## Public repository release (2026-10-12T18:13:06.987153+08:00)

- The user requested public visibility, superseding the earlier private-repository setting.
- Repository: `xzktx003/Cos_VQ`, branch `main`; GitHub API confirms `private=false` and `visibility=public`.
- Base commit: `137d12b515276e8269c81a8cf7b3be71cd9235cc`. The main README now describes public cloning access.
- Verification: GitHub visibility and documentation whitespace checked; algorithm code is unchanged.

## English README update (2026-10-12T17:47:52.455865+08:00)

- Algorithm repository: `xzktx003/Cos_VQ`, branch `main`, private.
- README content commit: `c0bda69ab1201d87f5774be8abc3157a752d7c35`.
- Parent source repository: `xzktx003/tmp_stom`, branch `main`, base commit `8f54a6a652e2df25be987007d9b53cc925533210`; the parent workspace changes are kept local for this update.
- Uploaded changes: the English main README and this release record. The README covers setup, previews, execution, supported models, evaluation, artifacts, and licensing.
- Verification: no Chinese text in the main README; local Markdown links and code fences checked; the documented single-model CLI dry-run passed; Git whitespace checks passed. Full GPU experiments were not run for this documentation update.

## Initial Standalone Private Cos_VQ Release (2026-10-11T23:04:40.736901+08:00)

- Version: first standalone CoS-VQ reproduction package. The original
  `code/code2open` directory was renamed to `code/Cos_VQ`; the Python
  distribution name is `cos-vq`.
- Algorithm repository: `https://github.com/xzktx003/Cos_VQ`, branch `main`;
  GitHub API verified `private=true` and `visibility=private`.
- Algorithm content commit: `5e3fd3dd17c33259e85dd59f7e75644f35b24e38`;
  remote `refs/heads/main` was confirmed to match that content commit. This
  release record and the remote verification record were added afterward.
- Release-record commit: `4b17011a76ab79ab26cfa25e627bca591473cad6`, pushed to
  `main`; after adding the release record, the remote file tree contained 282
  files.
- Paper source repository: `xzktx003/tmp_stom`, branch `main`, source commit
  `0b28b1df8eca9aaf7417ee038d3b6059e550b382` (local rename and release
  preparation commit; the paper repository was not pushed in this update). This
  was not a new paper version.
- Uploaded contents: complete method source, six-model frozen protocol, the
  orchestrator and six single-model launcher scripts, tests, dependency lock
  file, license, documentation, historical six-model compact result JSON, and
  the timestamped release verification record.
- Historical result:
  `experiments/results/reference/20260914_122219_v19_e2e_six_model_core_aggregate.json`,
  recorded during the release as matching the original file's SHA256; it is not
  an accuracy result rerun during this release.
- Verification: 12 orchestration tests passed; dry runs for all six model
  launchers passed from outside the package directory; syntax checks for seven
  shell scripts, Python AST parsing, project/lockfile name consistency, and
  `uv lock --check --offline` passed; 281 content files on the GitHub remote
  were verified.
- Full CUDA training and accuracy evaluation were not run in this release.
  Weights, checkpoints, dataset caches, test temporary directories, and full
  run logs were not committed.
