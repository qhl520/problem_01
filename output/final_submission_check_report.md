# 131 Pipeline Check Report

## Final Submission

| item | value |
| --- | ---: |
| file | output/tc_comp_predict_table.csv |
| rows | 30 |
| first date | 20140901 |
| last date | 20140930 |
| purchase monthly total | 7561979250 |
| redeem monthly total | 7575786693 |

## Frozen Copy

`output/best_131/tc_comp_predict_table.csv` equals `output/tc_comp_predict_table.csv`: `True`

## Ensemble Config

```json
{
  "w_stl": 0.75,
  "w_weekday_rule": 0.175,
  "w_recent14": 0.075,
  "backtest_score": 4.996906588942037
}
```

## Checks

| command | status |
| --- | --- |
| python run_131_pipeline.py | pass |
| python src/final_check.py | pass |
| python src/check_official_requirements.py | pass |

## Removed Legacy Logic

The project now keeps only the 131 STL ensemble pipeline. Old 120/121 baseline scripts, old high_score/original outputs, model caches, EDA outputs, and probe files were removed.
