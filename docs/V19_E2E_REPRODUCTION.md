# V19 + E2E 实验与精度复现指南

本文档说明如何从公开代码重新生成 V19 + E2E checkpoint，并按照论文协议复现
WikiText-2 PPL 和六项零样本准确率。协议的唯一机器可读来源是
`configs/reproduction/v19_e2e_six_models.json`；命令编排的唯一实现是
`scripts/reproduce_v19_e2e.py`。

## 1. 复现范围

| 模型键 | Hugging Face 模型 | 层数 | 独立入口 |
| --- | --- | ---: | --- |
| `qwen3-4b` | `Qwen/Qwen3-4B` | 36 | `scripts/reproduce_accuracy/qwen3_4b.sh` |
| `qwen3-8b` | `Qwen/Qwen3-8B` | 36 | `scripts/reproduce_accuracy/qwen3_8b.sh` |
| `qwen3-14b` | `Qwen/Qwen3-14B` | 40 | `scripts/reproduce_accuracy/qwen3_14b.sh` |
| `qwen3-32b` | `Qwen/Qwen3-32B` | 64 | `scripts/reproduce_accuracy/qwen3_32b.sh` |
| `llama2-7b` | `meta-llama/Llama-2-7b-hf` | 32 | `scripts/reproduce_accuracy/llama2_7b.sh` |
| `llama3.1-8b-instruct` | `meta-llama/Llama-3.1-8B-Instruct` | 32 | `scripts/reproduce_accuracy/llama3_1_8b_instruct.sh` |

每次完整运行包含量化、V19 搜索、E2E 微调、端点选择和最终评测，不是只读取论文表格。

## 2. 方法流水线

每个模型依次执行以下阶段：

1. **冻结输入与数据版本**：解析模型 snapshot，冻结本次所需 Hugging Face 数据集
   revision，并将实际 commit 写入 `campaign_manifest.json`。
2. **构建校准缓存**：Qwen3/Llama-2 使用 RedPajama；Llama-3.1 使用 FineWeb-Edu，
   并以独立 C4 guard 检查 tokenizer/cache 身份。
3. **Vector-GSQ baseline**：运行 NoWag-d6、Hessian-aware Vector-GPTQ 和
   block-level Vector-GSQ，生成 baseline checkpoint。Llama-3.1 额外执行 task-local
   assignment 校准。
4. **V19 候选优化**：从同一 baseline 独立运行 `scale_only`、`standard`、`wide`
   三条支路；用未参与优化的 development NLL 在 standard/wide 间选择 V19 端点。
5. **固定离散指派 E2E**：对 baseline 和 V19 分别使用 `5e-5`、`1e-4` 两个学习率
   训练 3 epochs；Qwen3-8B 与 Llama-3.1 还运行 scale→E2E 支路。初始端点也参与
   development NLL 选择，防止微调退化。
6. **最终评测**：仅在选择完成后评测各端点，写出原始 JSON、样本日志和 summary。

主要实现映射如下：

| 阶段 | 代码入口 |
| --- | --- |
| 总控、manifest、自动报告 | `scripts/reproduce_v19_e2e.py` |
| Qwen/Llama 缓存适配 | `experiments/qwen_qtip/`、`experiments/llama_qtip/` |
| Vector-GSQ baseline | `experiments/nowag_d6/run_prefix_vector_gsq.py` |
| Llama-3.1 task assignment | `experiments/nowag_d6/calibrate_task_local_assignments.py` |
| V19 模型级优化 | `experiments/nowag_d6/v19_model_driver.py` |
| V19 独立开发集选择 | `experiments/nowag_d6/v19_validation_score.py` |
| 固定指派 E2E | `experiments/nowag_d6/finetune_hard_codebook_e2e.py` |
| E2E 端点选择 | `experiments/nowag_d6/v19_e2e_selection.py` |
| PPL 与六任务精度 | `experiments/remote_runner/evaluate_logical_checkpoint_lm_eval.py` |

## 3. 冻结实验协议

- 量化：d=6，2 bits/weight，BF16，seed 0。
- V19：128×2048 校准，block-Frobenius majorizer，1 pass。
- standard：8 个 active positions，每个 ranking 4 个 candidates。
- wide：16 个 active positions，每个 ranking 8 个 candidates。
- E2E：context 4096，batch 4，3 epochs，gradient clip 1.0。
- PPL：WikiText-2 raw test，`seqlength=2048`。
- Accuracy：0-shot、batch size 1、seed 0、1000 bootstrap iterations。
- 任务：ARC-Challenge、ARC-Easy、HellaSwag、LAMBADA OpenAI、PIQA、WinoGrande。
- 汇总准确率：六项任务 macro average；原始任务指标保存在评测 JSON 中。

各模型的 calibration 数量、并发度、E2E 数据量、scale layout 等差异全部记录在
`configs/reproduction/v19_e2e_six_models.json`，不要在 shell 脚本中复制或手改。

## 4. 环境与数据准备

```bash
cd Cos_VQ
uv sync --frozen
```

需要 Python 3.10+、CUDA GPU、足够的本地模型/checkpoint 空间和 Hugging Face Hub
访问权限。Llama 模型需要提前接受上游许可。访问令牌只应通过 Hugging Face 本机登录或
环境变量提供，不要写入配置、脚本、manifest 或日志。

先运行 dry-run。它只打印完整命令，不下载数据、不创建输出、不启动 GPU：

```bash
./scripts/reproduce_accuracy/qwen3_4b.sh \
  --run-stamp 20260920_120000 \
  --dry-run
```

## 5. 正式运行

单模型、自动获取固定 snapshot：

```bash
./scripts/reproduce_accuracy/qwen3_8b.sh \
  --run-stamp 20260920_120000 \
  --cuda-visible-devices 0
```

单模型、本地 snapshot：

```bash
./scripts/reproduce_accuracy/qwen3_14b.sh \
  --run-stamp 20260920_130000 \
  --model-path /absolute/path/to/Qwen3-14B \
  --cuda-visible-devices 0
```

Llama-3.1 本地模型与参考 tokenizer：

```bash
./scripts/reproduce_accuracy/llama3_1_8b_instruct.sh \
  --run-stamp 20260920_140000 \
  --model-path /absolute/path/to/Llama-3.1-8B-Instruct \
  --reference-model-path /absolute/path/to/Meta-Llama-3-8B \
  --cuda-visible-devices 0
```

全部六模型顺序运行：

```bash
python -m scripts.reproduce_v19_e2e \
  --model all \
  --run-stamp 20260920_150000 \
  --cuda-visible-devices 0
```

训练阶段始终使用逻辑 `cuda:0`；用 `--cuda-visible-devices` 选择物理卡。
`--split-gpus` 要求恰好两张可见 GPU，只拆分 V19 development scoring 和最终评测，
不会把量化/V19/E2E 训练自动分布到两张卡。

## 6. 输出结构与结果判定

运行目录固定在 `experiments/results/v19_e2e/<RUN_STAMP>/`：

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

运行成功必须同时满足：

1. `campaign_manifest.json` 的顶层 `status` 为 `completed`；
2. 对应模型记录的 `status` 为 `completed`；
3. 每个选优 JSON 的 `completed` 为 `true` 且指向存在的 checkpoint；
4. `evaluation/<ENDPOINT>/summary.json` 存在；
5. 自动报告 `report/<RUN_STAMP>_v19_e2e_reproduction.md` 已生成。

普通模型评测 `baseline`、`scale`、`v19`、`baseline_e2e`、`v19_e2e` 五个端点；
Qwen3-8B 与 Llama-3.1 另有 `scale_e2e`。论文数值应从 summary/原始 JSON读取，
不要从终端日志人工抄录。

## 7. 失败、重跑与存储规则

- 总控脚本拒绝覆盖已存在的 `<RUN_STAMP>`，成功或失败记录都不会静默丢失。
- 当前公开总控不承诺对半完成 campaign 原地续跑；失败后先保留 manifest 与日志，修复后
  使用新的时间戳重跑，并在报告中记录前一次失败目录。
- 可使用 `--output-root`，但目标必须仍位于本包 `experiments/results/` 下。
- 不提交基础模型、数据集缓存、大型 checkpoint 或完整运行日志到 Git。
- 不删除原始 JSON、selection、manifest、配置哈希和自动报告；这些是论文结果的审计链。
- activation cache 可重建，但若要支持断点恢复，应在对应运行结束并完成归档前保留。

## 8. 验证公开包

```bash
bash -n scripts/reproduce_accuracy/*.sh
python -m pytest -q tests/test_reproduction_pipeline.py
python -m compileall -q scripts experiments src
```

完整 CUDA 复现耗时长且依赖模型授权、数据网络与 GPU，不能用上述 CPU 合同测试替代。
CPU 测试证明命令图、协议参数、模块打包和安全边界正确；最终论文指标仍以正式运行产物为准。
