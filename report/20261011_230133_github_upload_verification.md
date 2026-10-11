# Cos_VQ 私有仓库发布验证

日期时间：2026-10-11T23:01:34.948172+08:00。

## 目的

将独立复现包从 `code/code2open` 改名为 `code/Cos_VQ`，准备发布到 `xzktx003/Cos_VQ` 私有仓库，并确认改名后的端到端入口可生成完整命令计划。

## 原理

路径由代码包根目录动态推导；目录改名不需要修改算法模块名。Python 分发名统一为 `cos-vq`，同步锁文件中的 editable 项，不重新选择依赖版本。

## 配置与工具

- Python：3.11.0 | packaged by conda-forge | (main, Jan 14 2023, 12:28:47) [GCC 11.3.0]。
- 模型：Qwen3-4B/8B/14B/32B、Llama-2-7B、Llama-3.1-8B-Instruct；仅预览命令，没有加载权重。
- 配置：`configs/reproduction/v19_e2e_six_models.json`。
- 工具：pytest、bash 语法检查、Python AST/tomllib、uv 离线锁文件检查、Git 与 GitHub CLI。
- 验证参数：六个 launcher 的 `--dry-run`，从包外目录运行；PPL2048 与六任务评测参数由编排测试检查。
- 本次没有执行 CUDA 训练或精度评测。

## 步骤与结果

1. 同步目录名、README 路径、Python 分发名及锁文件。
2. `tests/test_reproduction_pipeline.py`：12 项通过。
3. 七个 shell 脚本语法检查通过，六个模型 launcher dry-run 全部通过。
4. `scripts/`、`experiments/`、`src/` 中 Python 源文件 AST 解析通过。
5. `uv lock --check --offline` 通过；项目名称与锁文件一致。
6. 历史六模型结果 JSON 与原始文件 SHA256 一致，结果属于 2026-09-14 的历史实验。

验证 JSON：`experiments/results/upload_validation/20261011_230133_github_upload_validation.json`。

## 发布范围

算法源码、六模型配置、总控与单模型启动脚本、测试、依赖锁文件、许可证、文档、历史紧凑结果与本次验证记录。运行缓存、权重、checkpoint、测试临时目录和完整日志由 Git 忽略。GitHub 的远端私有属性与提交号将在 `docs/version_uploads.md` 登记。
