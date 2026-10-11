# Cos_VQ 发布记录

## Cos_VQ 独立私有仓库首发（2026-10-11T23:04:40.736901+08:00）

- 版本：CoS-VQ 独立复现包首发，原目录 `code/code2open` 已改名为 `code/Cos_VQ`；Python 分发名为 `cos-vq`。
- 算法仓库：`https://github.com/xzktx003/Cos_VQ`，分支 `main`，GitHub API 验证 `private=true`、`visibility=private`。
- 算法内容提交：`5e3fd3dd17c33259e85dd59f7e75644f35b24e38`；远端 `refs/heads/main` 已确认与该内容提交一致。随后增加本发布记录与远端验证记录。
- 论文来源仓库：`xzktx003/tmp_stom`，分支 `main`，来源提交 `0b28b1df8eca9aaf7417ee038d3b6059e550b382`（本地改名与发布准备提交，本次未推送论文仓库）。本次不是新增论文版本。
- 上传内容：完整方法源码、六模型冻结协议、总控及六个单模型启动脚本、测试、依赖锁文件、许可证、文档、历史六模型紧凑结果 JSON 和带时间戳的发布验证记录。
- 历史结果：`experiments/results/reference/20260914_122219_v19_e2e_six_model_core_aggregate.json`，与原文件 SHA256 一致；不是本次重新测得的精度。
- 验证：12 项编排测试通过；六个模型从包外目录启动 dry-run 通过；七个 shell 脚本语法、Python AST、项目/锁文件名称一致性与 `uv lock --check --offline` 通过；GitHub 远端 281 个内容文件核对通过。
- 本次未运行完整 CUDA 训练与精度评测。权重、checkpoint、数据集缓存、测试临时目录及完整运行日志未提交。

