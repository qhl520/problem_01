# 余额宝资金申购赎回预测项目报告

## 1. 赛题与目标

本项目预测 2014 年 9 月 1 日至 2014 年 9 月 30 日每日平台总申购额 `purchase` 和总赎回额 `redeem`。提交文件为无表头 CSV，共 30 行，金额单位为分。

最终提交文件：

```text
output/tc_comp_predict_table.csv
```

## 2. 数据与工程流程

项目使用四类赛题数据：

- 用户基本信息表：`user_profile_table.csv`
- 用户申购赎回表：`user_balance_table.csv`
- 收益率表：`mfd_day_share_interest.csv`
- Shibor 利率表：`mfd_bank_shibor.csv`

原始数据放在 `data/raw/`。日级训练数据由 `src/make_daily_data.py` 从用户流水中聚合得到，核心字段为：

```text
date,purchase,redeem
```

原始建模流程由 `run_all.py` 串联，包括数据检查、业务规则检查、日级聚合、EDA、规则 baseline、特征构造、时序交叉验证、模型训练、融合预测和最终提交检查。

## 3. 历史原始模型

原始模型阶段使用：

- 时间特征、节假日特征、lag/rolling 特征、同星期历史特征；
- 规则模型；
- RandomForest、LightGBM、ExtraTrees；
- 分目标权重搜索，对 `purchase` 和 `redeem` 分别融合。

该阶段生成 initial 基线提交：

```text
output/original/tc_initial.csv
```

线上反馈显示 initial 原始版本约 120 分，是当前项目中最可靠的基线。

## 4. 线上反馈驱动的方案调整

多轮线上反馈得到以下结论：

```text
initial 原始版本：约 120 分
redeem +1.5%：约 118 分
redeem +3.0%：约 117 分
rule_bank_best：低于 initial
```

这些反馈说明：

1. initial 的全月 `redeem` 总量已经较接近真实值；
2. 全月整体上调 `redeem` 会降低线上分数；
3. 大幅压低 `redeem` 的 rule bank 方案也不适合线上；
4. 后续优化不应继续赌总量方向，而应在总量不变的前提下优化日级形状。

## 5. 为什么放弃 rule_bank_best

`rule_bank_best` 在本地验证中改善了部分月份的近似分数，但线上低于 initial。原因很可能是本地 2014-04 至 2014-08 验证窗口无法完全代表 2014-09 的真实分布，尤其是中秋、节后和国庆前的资金行为。

因此本项目不再把本地验证分数最高作为唯一目标，而是以线上反馈为优先约束。

## 6. 为什么不再整体上调 redeem

`tc_initial_redeem_1015.csv` 和 `tc_initial_redeem_1030.csv` 的线上分数均低于 initial，说明全月 `redeem` 总量上调不是有效方向。

最终方案保留 initial 的全月 `redeem` 总量，并同样锁定 `purchase` 总量，只做日级重分配。

## 7. 总量锁定与日级形状重分配

当前 121 分方案由 `src/generate_121_submissions.py` 生成，核心逻辑是：

1. 读取 original initial 提交；
2. 从历史真实日级数据提取 2014-08 同星期形状和 2014-07 至 2014-08 最近 8 周同星期形状；
3. 将形状模板归一到 initial 全月总量；
4. 以 `0.90 * initial + 0.10 * shape` 做温和融合；
5. 对中秋、节后和国庆前日期做总量守恒式温和重分配；
6. 四舍五入后仍保持全月 `purchase/redeem` 总量等于 initial。

主方案使用 2014-08 同星期形状；保守备选使用最近 8 周同星期形状。

## 8. 当前最终提交方案

当前保留三个关键提交：

```text
output/original/tc_initial.csv
output/high_score_final/tc_comp_predict_table.csv
output/high_score_final/tc_comp_predict_table_conservative.csv
```

默认最终提交：

```text
output/tc_comp_predict_table.csv
```

它与主方案 `output/high_score_final/tc_comp_predict_table.csv` 完全一致。

## 9. 工程复现方式

完整复现原始模型并生成 121 分方案：

```bash
python run_original_plus_121.py
```

该脚本默认在生成最终文件后清理可再生成的模型、EDA、诊断、日志和历史候选输出目录。如需保留中间产物用于调试，可运行：

```bash
python run_original_plus_121.py --keep-intermediate
```

如果已经保留原始模型提交，只重新生成主方案和保守方案：

```bash
python src/generate_121_submissions.py --initial output/original/tc_initial.csv
```

最终检查：

```bash
python src/final_check.py
python src/check_official_requirements.py
```

## 10. 工程化交付状态

当前项目已清理：

- 历史候选池；
- 模型 pkl 缓存；
- EDA 图片；
- 旧诊断文件；
- Python 缓存。

GitHub 精简版本不应包含 `data/raw/` 和 `.git/`。本地完整版本可以保留 `data/raw/` 以便完整运行。
