# 余额宝资金申购赎回预测

本项目是天池余额宝资金流入流出预测竞赛的工程化版本，目标是预测 2014 年 9 月每天的平台总申购额 `purchase` 和总赎回额 `redeem`。金额单位为分，最终提交文件为无表头 CSV。

当前默认提交文件：

```text
output/tc_comp_predict_table.csv
```

当前保留的最终方案是线上反馈驱动的 121 分左右方案：

```text
initial 原始模型基线
+ 锁定 purchase/redeem 全月总量
+ 日级形状重分配
+ 主方案 / 保守备选方案
```

## 赛题目标

预测区间：

```text
2014-09-01 至 2014-09-30
```

提交格式：

```csv
20140901,295000000,331000000
20140902,287000000,298000000
```

要求：

- 文件名为 `tc_comp_predict_table.csv`
- 无表头
- 30 行，日期覆盖 `20140901` 到 `20140930`
- `purchase` 和 `redeem` 为非负整数，单位为分

## 数据说明

原始数据需要放在：

```text
data/raw/
```

需要的赛题文件包括：

```text
user_balance_table.csv
user_profile_table.csv
mfd_day_share_interest.csv
mfd_bank_shibor.csv
comp_predict_table.csv
```

GitHub 精简版本默认不包含 `data/raw/`，请从赛题官网下载原始数据后自行放入该目录。

## 目录结构

```text
fund_forecast/
├─ data/
│  ├─ raw/                         # 本地原始赛题数据，GitHub 版本不提交
│  └─ processed/
│     └─ daily_balance.csv          # 日级汇总数据，可由 make_daily_data.py 再生成
├─ output/
│  ├─ tc_comp_predict_table.csv     # 当前默认最终提交，等于 121 主方案
│  ├─ original/
│  │  └─ tc_initial.csv             # 原始模型基线提交
│  └─ high_score_final/
│     ├─ tc_comp_predict_table.csv  # 121 主方案
│     └─ tc_comp_predict_table_conservative.csv
├─ report/
├─ src/
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ run_all.py
└─ run_original_plus_121.py
```

## 环境安装

建议使用 Python 3.10 或 3.11。

```bash
pip install -r requirements.txt
```

主要依赖：

```text
pandas
numpy
scikit-learn
matplotlib
joblib
lightgbm
tqdm
```

如果 `lightgbm` 不可用，项目会在部分模型工具中降级到 scikit-learn 的 `HistGradientBoostingRegressor`。

## 一键运行

原始工程完整流程：

```bash
python run_all.py
```

当前推荐复现流程：

```bash
python run_original_plus_121.py
```

默认会在生成最终文件后清理可再生成的模型、EDA、诊断、日志和历史候选输出目录；调试时可使用：

```bash
python run_original_plus_121.py --keep-intermediate
```

该流程会：

1. 运行原始建模链路，生成 initial 基线提交；
2. 保存原始提交到 `output/original/tc_initial.csv`；
3. 基于 initial 总量生成 121 主方案和保守备选方案；
4. 将主方案复制到 `output/tc_comp_predict_table.csv`。

如果已经有 `output/original/tc_initial.csv`，只想重新生成两个 121 文件：

```bash
python src/generate_121_submissions.py --initial output/original/tc_initial.csv
```

## 当前最终方案

线上反馈表明：

- `initial` 原始版本线上约 120 分，是当前可靠基线；
- `redeem * 1.015` 约 118 分；
- `redeem * 1.030` 约 117 分；
- `rule_bank_best` 本地验证更高，但线上低于 initial。

因此最终结论是：`redeem` 全月总量不宜整体上调，也不宜大幅下调。当前最终方案锁定 initial 的 `purchase/redeem` 全月总量，只做温和的日级形状重分配。

保留输出：

```text
output/original/tc_initial.csv
output/high_score_final/tc_comp_predict_table.csv
output/high_score_final/tc_comp_predict_table_conservative.csv
output/tc_comp_predict_table.csv
```

其中 `output/tc_comp_predict_table.csv` 等于 `output/high_score_final/tc_comp_predict_table.csv`。

## 检查命令

```bash
python src/final_check.py
python src/check_official_requirements.py
```

检查内容包括：

- 30 行
- 无表头
- 日期完整且升序
- 日期不重复
- `purchase/redeem` 为非负整数
- 金额单位为分

## GitHub 提交注意事项

建议保留：

```text
README.md
requirements.txt
.gitignore
run_all.py
run_original_plus_121.py
src/
report/
data/processed/daily_balance.csv
output/original/tc_initial.csv
output/high_score_final/tc_comp_predict_table.csv
output/high_score_final/tc_comp_predict_table_conservative.csv
output/tc_comp_predict_table.csv
```

不要提交：

```text
data/raw/
output/models/
output/eda/
output/diagnostics/
.git/
__pycache__/
*.pyc
*.pyo
```

## 文件清理状态

当前项目已清理历史实验候选池、模型缓存、EDA 图片、诊断旧文件和 Python 缓存。`output/` 目录只保留当前交付相关提交文件和工程整理报告。
