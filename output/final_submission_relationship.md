# Final Submission Relationship

Current default final submission:

```text
output/tc_comp_predict_table.csv
```

Relationship:

| file | role |
| --- | --- |
| output/original/tc_initial.csv | original model initial baseline |
| output/high_score_final/tc_comp_predict_table.csv | 121 main submission |
| output/high_score_final/tc_comp_predict_table_conservative.csv | 121 conservative backup submission |
| output/tc_comp_predict_table.csv | official default submission copied from the 121 main file |

Default equals 121 main: `true`

The retained final idea is not a large candidate pool. It is:

```text
original initial monthly total
+ locked purchase/redeem monthly sums
+ mild daily shape redistribution
+ main/conservative 121 outputs
```
