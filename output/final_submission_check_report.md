# Final Submission Check Report

Executed lightweight final checks after cleanup.

| check | file | status | detail |
| --- | --- | --- | --- |
| default_final | output/tc_comp_predict_table.csv | pass | rows=30, purchase_sum=7541997807, redeem_sum=7730818312 |
| original_initial | output/original/tc_initial.csv | pass | rows=30, purchase_sum=7541997807, redeem_sum=7730818312 |
| main_121 | output/high_score_final/tc_comp_predict_table.csv | pass | rows=30, purchase_sum=7541997807, redeem_sum=7730818312 |
| conservative_121 | output/high_score_final/tc_comp_predict_table_conservative.csv | pass | rows=30, purchase_sum=7541997807, redeem_sum=7730818312 |

Submission format requirements:

| check | status |
| --- | --- |
| official headerless CSV | pass |
| 30 rows | pass |
| dates 20140901-20140930 | pass |
| purchase/redeem integer fen | pass |
| purchase/redeem non-negative | pass |
| default final equals 121 main | pass |
