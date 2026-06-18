# 余额宝资金申购赎回预测：131 分 STL 集成版

本项目保留当前有效的 **131 分逻辑**，即你运行的：

```bash
python src/optimize_ensemble_v2.py
```

旧的 120/121 分 baseline、传统随机森林/LightGBM 训练入口、旧 high_score 输出、probe 输出和模型缓存都已移除。现在项目只有一条主流程：**STL/PPT 对齐分解 + weekday rule + recent14 rule 集成**。

## 最终提交文件

需要提交的文件是：

```text
output/tc_comp_predict_table.csv
```

项目也会保存一份 131 固化副本：

```text
output/best_131/tc_comp_predict_table.csv
```

两者应保持一致。

当前 131 文件月总量记录在：

```text
output/best_131/target_totals.json
```

## 一键运行

推荐运行：

```bash
python run_131_pipeline.py
```

该命令会依次执行：

```text
src/optimize_ensemble_v2.py
src/final_check.py
```

也可以单独运行核心脚本：

```bash
python src/optimize_ensemble_v2.py
```

## 131 模型思路

核心脚本：

```text
src/optimize_ensemble_v2.py
```

主要依赖：

```text
src/stl_model.py
src/rule_models.py
src/data_utils.py
src/evaluate.py
```

流程概要：

1. 从 `data/raw/user_balance_table.csv` 聚合日级 purchase/redeem；如果 raw 不存在，则回退到 `data/processed/daily_balance.csv`。
2. 使用 `stl_model.py` 构建 STL/PPT 对齐分解模型。
3. 对 2014 年 6/7/8 月进行回测，网格搜索 STL、weekday rule、recent14 rule 的融合权重。
4. 预测 2014 年 9 月 30 天。
5. 输出官方无表头提交文件 `output/tc_comp_predict_table.csv`。
6. 同步固化到 `output/best_131/tc_comp_predict_table.csv`。

## 数据目录

本地完整数据放在：

```text
data/raw/
```

GitHub/课程精简版不提交 `data/raw/`。如果没有 raw 数据，脚本会尝试使用：

```text
data/processed/daily_balance.csv
```

## 提交格式

官方提交文件必须是无表头 CSV：

```csv
20140901,purchase,redeem
20140902,purchase,redeem
...
20140930,purchase,redeem
```

金额单位为分，必须是非负整数。

格式检查：

```bash
python src/final_check.py
```

完整赛题数据检查：

```bash
python src/check_official_requirements.py
```

## 当前保留结构

```text
fund_forecast/
├── data/
│   ├── raw/                         # 本地原始数据，不提交
│   └── processed/daily_balance.csv
├── output/
│   ├── best_131/
│   │   ├── tc_comp_predict_table.csv
│   │   ├── target_totals.json
│   │   └── best_131_summary.md
│   ├── stl_ensemble_config.json
│   └── tc_comp_predict_table.csv
├── report/
│   └── fund_forecast_report.md
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── evaluate.py
│   ├── rule_models.py
│   ├── stl_model.py
│   ├── optimize_ensemble_v2.py
│   ├── final_check.py
│   └── check_official_requirements.py
├── run_131_pipeline.py
├── requirements.txt
└── README.md
```

## 注意

不要再提交或引用旧文件：

```text
output/high_score_final/
output/original/
output/best_130/
output/probes_130/
output/tc_comp_predict_table_best.csv
```

这些旧逻辑已经从当前项目中删除。
