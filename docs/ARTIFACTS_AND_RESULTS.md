# 代码与实验结果保留说明

## 公开代码包边界

`code/Cos_VQ` 包含复现 V19 + E2E 所需的算法实现、配置、第三方 NoWag 代码、测试、
模型级 launcher 和评测入口。`tests/test_reproduction_pipeline.py` 会遍历六模型命令图，
确认所有被调用的 Python 模块都已打包。

## 新运行产物

所有新实验产物保存在本包内：

- `experiments/results/v19_e2e/<RUN_STAMP>/`：缓存、checkpoint、selection、原始评测 JSON、日志；
- `report/<RUN_STAMP>_v19_e2e_reproduction.md`：从 manifest 与 summary 自动生成的报告；
- `campaign_manifest.json`：配置哈希、实际模型/数据 revision、逐步命令、状态和时间。

这些目录被 Git 忽略是为了避免误提交多 GB 文件，不代表运行后可以随意删除。归档时至少保留
manifest、冻结配置、selection、最终 checkpoint、原始评测 JSON、summary、日志和报告。

## 父研究仓库历史结果

父仓库 `code/GSQ_nowag_d1_20260716_015355/experiments/results/`、根目录
`experiments/results/` 和 `report/` 中的历史结果保持原位。公开包不会移动、覆盖或删除它们，
也不会依赖父仓库的绝对路径完成一轮新的复现。

此前按明确清理指令删除的 activation cache 不属于当前保留集合，无法从 Git 恢复；其余现存
代码、checkpoint、结果 JSON 和报告应继续保留。若今后清理可重建缓存，应先确认最终
checkpoint、评测 JSON、manifest 和报告已经完整归档。

## 历史紧凑结果快照

[六模型核心结果 JSON](../experiments/results/reference/20260914_122219_v19_e2e_six_model_core_aggregate.json) 是 2026-09-14 汇总文件的原样副本，保留原始证据路径和 SHA256。JSON 中的历史绝对路径仅用于来源追踪，新的端到端运行不依赖这些路径。这份快照不是本次上传重新执行实验的结果。
