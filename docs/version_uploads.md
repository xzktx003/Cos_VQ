# Cos_VQ 发布记录

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

## Cos_VQ 独立私有仓库首发（2026-10-11T23:04:40.736901+08:00）

- 版本：CoS-VQ 独立复现包首发，原目录 `code/code2open` 已改名为 `code/Cos_VQ`；Python 分发名为 `cos-vq`。
- 算法仓库：`https://github.com/xzktx003/Cos_VQ`，分支 `main`，GitHub API 验证 `private=true`、`visibility=private`。
- 算法内容提交：`5e3fd3dd17c33259e85dd59f7e75644f35b24e38`；远端 `refs/heads/main` 已确认与该内容提交一致。随后增加本发布记录与远端验证记录。
- 发布记录提交：`4b17011a76ab79ab26cfa25e627bca591473cad6`，已推送到 `main`；加入发布记录后，远端文件总数为 282。
- 论文来源仓库：`xzktx003/tmp_stom`，分支 `main`，来源提交 `0b28b1df8eca9aaf7417ee038d3b6059e550b382`（本地改名与发布准备提交，本次未推送论文仓库）。本次不是新增论文版本。
- 上传内容：完整方法源码、六模型冻结协议、总控及六个单模型启动脚本、测试、依赖锁文件、许可证、文档、历史六模型紧凑结果 JSON 和带时间戳的发布验证记录。
- 历史结果：`experiments/results/reference/20260914_122219_v19_e2e_six_model_core_aggregate.json`，与原文件 SHA256 一致；不是本次重新测得的精度。
- 验证：12 项编排测试通过；六个模型从包外目录启动 dry-run 通过；七个 shell 脚本语法、Python AST、项目/锁文件名称一致性与 `uv lock --check --offline` 通过；GitHub 远端 281 个内容文件核对通过。
- 本次未运行完整 CUDA 训练与精度评测。权重、checkpoint、数据集缓存、测试临时目录及完整运行日志未提交。
