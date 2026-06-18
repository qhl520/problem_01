# 余额宝资金申购赎回预测：131 基线强候选版

本项目当前只保留已经验证过的 **131 分 STL ensemble 基线**，并在其上生成少量、高信息量、可回滚的冲分候选。候选策略吸收了公开方案中的有效思路：总量锁定、节假日/调休单独修正、redeem 单独激进调参、purchase 拆分为 direct_purchase + share、calendar 弱模型小权重融合，但不直接搬用旧代码。

## 绝对保护的 131 基线

```text
output/baseline_131/tc_comp_predict_table.csv
```

该文件是保底锚点。候选生成脚本只会在它缺失时初始化一次，之后默认不覆盖。不要手动覆盖这个文件。

当前最终提交文件是：

```text
output/tc_comp_predict_table.csv
```

只有当某个候选线上分数高于 131 时，才手动晋级为最终提交。

## 重新生成 131 主流程

```bash
python run_131_pipeline.py
```

主模型仍是：

- STL/PPT 对齐分解；
- weekday rule；
- recent14 rule；
- redeem = consume + transfer；
- 2014 年 6/7/8 月回测搜索 ensemble 权重；
- 输出官方无表头 CSV。

## 生成 3 个强候选

```bash
python run_131_candidate_pipeline.py
```

或：

```bash
python src/generate_131_candidates.py
```

输出目录：

```text
output/candidates_131/
```

摘要文件：

```text
output/candidates_131/candidates_summary.csv
output/candidates_131/candidates_summary.md
```

每次生成会清理旧候选目录里的旧候选文件，只保留当前 3 个候选。

## 当前推荐提交候选

优先级从高到低：

```text
04_bigfeast_2013_shape_w18_plus_0_6pct.csv
05_bigfeast_2013_shape_w26_plus_0_6pct.csv
```

策略说明：

- `04_bigfeast_2013_shape_w18_plus_0_6pct.csv`：吸收 200+ 答辩 PPT 的 `f_bigfeast` 思路，用 2013 年 9 月 25-30 日真实 redeem 分布作为国庆前异常项参考，将 2014 年 9 月 25-30 日形状向历史分布靠拢 18%，并保持已验证有效方向的全月 redeem +0.6%。
- `05_bigfeast_2013_shape_w26_plus_0_6pct.csv`：同样使用 2013 国庆前真实分布，但靠拢强度提高到 26%，更激进地测试 9/25 和 9/28 偏高、9/29 和 9/30 偏低的历史形状。

已有线上反馈：

- `01_redeem_total_plus_1_1pct.csv` = 131，说明单纯抬 redeem 总量不够。
- `02_national_early_shift_plus_0_6pct.csv` = 133，说明国庆前 redeem 前移方向有效。
- `03_purchase_decomp_linear_0909.csv` = 126，暂时不继续押 purchase 分解/9 月 9 日组合候选。

## 验证格式

检查当前最终提交：

```bash
python src/final_check.py
```

检查官方数据与字段要求：

```bash
python src/check_official_requirements.py
```

所有候选生成时都会自动校验：

- 无表头；
- 30 行；
- 日期为 20140901-20140930；
- purchase/redeem 为非负整数；
- 金额单位为分。

## 手动晋级候选

只有线上分数高于 131 才晋级：

```bash
python src/promote_candidate.py --candidate output/candidates_131/04_bigfeast_2013_shape_w18_plus_0_6pct.csv
```

晋级会备份当前：

```text
output/tc_comp_predict_table.csv
```

备份目录：

```text
output/backups/
```

晋级不会覆盖：

```text
output/baseline_131/tc_comp_predict_table.csv
```

## 可选分析脚本

用户行为日统计：

```bash
python src/build_user_behavior_stats.py
```

purchase 分解实验：

```bash
python src/experiment_purchase_decomposition.py
```

calendar 弱模型：

```bash
python src/simple_linear_date_model.py
```

这些脚本只产出 analysis/experiments 文件，不替换主模型。
