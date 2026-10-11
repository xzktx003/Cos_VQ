# CoS-VQ

本目录是 CoS-VQ 的独立复现代码包（代码中的方法版本名为 V19 + E2E），包含 GSQ/NoWag 基线、数据缓存适配器、V19 尺度与码字分配优化、固定分配端到端微调、端点选优，以及 WikiText-2 PPL 和六项 `lm-eval` 评测入口。

## 文档导航

- [V19 + E2E 完整实验与精度复现指南](docs/V19_E2E_REPRODUCTION.md)：方法阶段、冻结协议、六模型命令、输出和验收标准。
- [每个模型的独立精度复现入口](scripts/reproduce_accuracy/README.md)：六个模型 launcher 对照表。
- [代码与实验结果保留说明](docs/ARTIFACTS_AND_RESULTS.md)：公开包边界、运行产物和历史结果归档规则。
- [机器可读六模型协议](configs/reproduction/v19_e2e_six_models.json)：所有正式参数的唯一配置来源。

最短安全检查命令：

```bash
cd Cos_VQ
./scripts/reproduce_accuracy/qwen3_4b.sh \
  --run-stamp 20260920_120000 --dry-run
```

`--dry-run` 不下载模型或数据、不启动训练或评测，也不创建结果目录。确认命令图和本机路径后，去掉该参数执行正式复现。

六模型协议冻结在 `configs/reproduction/v19_e2e_six_models.json`。总控脚本按模型顺序执行整个流水线，记录模型与数据集修订号、配置哈希、命令、日志和端点结果；脚本拒绝覆盖已有 campaign 目录。模型权重、数据缓存、checkpoint 和日志只写到本目录下的 `experiments/results/`，汇总复现报告写到 `report/`。这些运行产物不随代码包提交。

## 获取代码

```bash
git clone https://github.com/xzktx003/Cos_VQ.git
cd Cos_VQ
```

在父研究仓库中，本代码包位于 `code/Cos_VQ/`。

## 环境准备

需要 Python 3.10+、支持 BF16 的 CUDA GPU、`uv` 和 Hugging Face Hub 访问权限。Llama 模型可能受 Hugging Face 授权限制，运行者必须先在 Hugging Face 接受对应模型条款并完成本机登录；本包不会保存访问令牌。

```bash
cd Cos_VQ
uv sync --frozen
```

此目录中的 `pyproject.toml` / `uv.lock` 随当前代码快照提供；它们并非对历史论文运行环境的逐项重建。历史实验记录使用过 Torch 2.4、Transformers 4.52.4 和 `lm-eval` 0.4.4，而当前锁文件来自较新的代码环境。因此，在没有独立环境对照验证前，不应声称仅凭当前锁文件即可逐位复现历史指标。本次整理没有安装依赖、下载模型/数据或启动训练和评测。

## 运行

先只预览命令计划：

```bash
uv run --frozen python -m scripts.reproduce_v19_e2e \
  --model all --run-stamp 20260916_120000 --dry-run
```

确认本机环境、模型访问权限和 GPU 资源后，按同一命令去掉 `--dry-run` 才会真正运行。`--model` 可选 `all` 或配置中的单个模型名；单模型可用 `--model-path` 提供已下载模型快照，Llama-3.1 可额外用 `--reference-model-path` 指定参考 tokenizer 快照。可用 `--cuda-visible-devices` 选择物理 GPU；`--split-gpus` 仅将 V19 开发集评分和最终评测拆到恰好两张可见 GPU，量化、V19 优化和 E2E 微调仍按协议在一张可见 GPU 上运行。

不同模型使用其配置规定的缓存与训练预算。Llama-3.1 还会构建 FineWeb-Edu + C4 guard 缓存、执行 task-local assignment 校准，并启用 group-scale E2E；Qwen3-8B 与 Llama-3.1 包含 scale→E2E 支路。V19 在 standard / wide 两组候选间按独立 development NLL 选择，E2E 在两个学习率和初始端点间按 development NLL 选择。最终评测固定 WikiText-2 test、seqlength 2048 与六项 0-shot 任务。

每个论文模型也有独立入口，位于 `scripts/reproduce_accuracy/`。这些脚本可从任意目录调用，参数原样传递给总控脚本。例如：

```bash
./scripts/reproduce_accuracy/qwen3_8b.sh \
  --run-stamp 20260920_120000 --dry-run
```

模型与脚本的完整对应表见 [`scripts/reproduce_accuracy/README.md`](scripts/reproduce_accuracy/README.md)。所有新运行的 checkpoint、原始精度 JSON、日志和报告均保存在本包内；父研究仓库已有实验结果保持原位，不移动、不覆盖、不删除。

RedPajama、FineWeb-Edu、C4 与 WikiText-2 默认从 Hugging Face 数据集仓库读取。总控脚本会在真实运行开始时将配置里的可变数据集分支解析为不可变 commit SHA，并写入 campaign manifest；也可通过 `--redpajama-revision`、`--fineweb-revision`、`--c4-revision`、`--wikitext-revision` 显式指定数据集 revision。Qwen3-4B/8B 模型 revision 已固定；其余 `main` 模型引用的实际 snapshot commit 会记录在 manifest 中。若需再次使用同一模型快照，可对单模型运行传入 `--model-path` 指向该本地 snapshot。

## 检查与测试

以下命令只做语法/单元测试，不执行论文实验：

```bash
python -m pytest -q
```

`pytest` 是测试专用工具，不属于运行依赖；请在开发环境预先安装。代码整理验收只运行静态检查、命令 dry-run 和 CPU 单元测试；完整 CUDA 训练、数据集下载、WikiText-2 与 `lm-eval` 结果均未在本次运行。
原研究仓库中 4 项依赖一次性本地 shell launcher 的合同测试标记为 skip；这些 launcher 未复制到独立包，六模型编排合同由 `test_reproduction_pipeline.py` 覆盖。

## 许可

本包根目录 `LICENSE` 适用于相应项目文件。`third_party/NoWag/` 包含 GPL-3.0 第三方代码，适用其自带许可；另见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。模型和数据集另受各自上游许可与访问条款约束。
