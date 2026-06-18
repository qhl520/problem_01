# 余额宝资金申购赎回预测项目报告：131 分 STL 集成版

## 1. 当前结论

当前项目只保留 131 分逻辑，最终提交文件为：

```text
output/tc_comp_predict_table.csv
```

固化副本为：

```text
output/best_131/tc_comp_predict_table.csv
```

旧的 120/121 分方案、旧 high_score 输出、旧 original baseline、传统 ML 训练产物和 probe 文件已经清理。

## 2. 131 主流程

运行入口：

```bash
python run_131_pipeline.py
```

核心脚本：

```bash
python src/optimize_ensemble_v2.py
```

该流程使用：

- STL/PPT 对齐的时间序列分解；
- 2014 年 6/7/8 月回测；
- STL、weekday rule、recent14 rule 三类分量融合；
- `output/best_131/target_totals.json` 记录的 131 总量作为当前项目锚点。

## 3. 保留的源码

当前保留源码只服务 131 流程：

```text
src/config.py
src/data_utils.py
src/evaluate.py
src/rule_models.py
src/stl_model.py
src/optimize_ensemble_v2.py
src/final_check.py
src/check_official_requirements.py
```

## 4. 已删除的旧逻辑

已删除：

- 旧 baseline：`baseline_v0.py`、`baseline_v1_weekday.py`；
- 旧 121 生成器：`generate_121_submissions.py`；
- 传统 ML 训练/预测：`train_*`、`predict_*`、`search_ensemble_weights.py`；
- 旧 STL enhanced 实验入口；
- EDA、诊断、模型 pkl、旧交叉验证、旧 feature importance；
- `output/high_score_final/`、`output/original/`、`output/best_130/`、`output/probes_130/`。

## 5. 提交与检查

提交前运行：

```bash
python src/final_check.py
```

本地完整数据存在时可运行：

```bash
python src/check_official_requirements.py
```

## 6. 风险说明

当前项目已经非常聚焦，优点是不会再误提交旧 120/121 文件。需要注意的是，`src/optimize_ensemble_v2.py` 会重新生成：

```text
output/tc_comp_predict_table.csv
output/best_131/tc_comp_predict_table.csv
output/stl_ensemble_config.json
```

如果后续又进行新的线上试验，应先备份当前 131 文件，再修改主流程。
