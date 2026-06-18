# Data Schema Report

Raw data directory: `C:\Users\35047\Desktop\problem_1\fund_forecast\data\raw`
CSV files found: 5

## comp_predict_table.csv

- Encoding: `utf-8`
- Inferred type: `submission_template`
- Shape: `3` rows x `3` columns
- Columns: `report_date, purchase, redeem`

### Date Columns

- `report_date`: 2014-09-01 to 2014-09-03

### Missing Values

- `report_date`: 0
- `purchase`: 0
- `redeem`: 0

### Head

| report_date | purchase | redeem |
| --- | --- | --- |
| 20140901 | 40000000 | 30000000 |
| 20140902 | 40000000 | 30000000 |
| 20140903 | 40000000 | 30000000 |

## mfd_bank_shibor.csv

- Encoding: `utf-8`
- Inferred type: `shibor`
- Shape: `294` rows x `9` columns
- Columns: `mfd_date, Interest_O_N, Interest_1_W, Interest_2_W, Interest_1_M, Interest_3_M, Interest_6_M, Interest_9_M, Interest_1_Y`

### Date Columns

- `mfd_date`: 2013-07-01 to 2014-08-29

### Missing Values

- `mfd_date`: 0
- `Interest_O_N`: 0
- `Interest_1_W`: 0
- `Interest_2_W`: 0
- `Interest_1_M`: 0
- `Interest_3_M`: 0
- `Interest_6_M`: 0
- `Interest_9_M`: 0
- `Interest_1_Y`: 0

### Head

| mfd_date | Interest_O_N | Interest_1_W | Interest_2_W | Interest_1_M | Interest_3_M | Interest_6_M | Interest_9_M | Interest_1_Y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20130701.0 | 4.456 | 5.423 | 6.04 | 6.88 | 5.295 | 4.239 | 4.282 | 4.4125 |
| 20130702.0 | 3.786 | 4.75 | 5.074 | 5.8 | 5.211 | 4.2344 | 4.2808 | 4.407 |
| 20130703.0 | 3.4 | 4.242 | 4.658 | 5.2 | 5.148 | 4.23 | 4.2796 | 4.4022 |
| 20130704.0 | 3.348 | 3.938 | 4.464 | 5.102 | 5.029 | 4.2287 | 4.2776 | 4.4 |
| 20130705.0 | 3.38 | 3.816 | 4.295 | 4.7885 | 4.939 | 4.2273 | 4.2749 | 4.4 |

## mfd_day_share_interest.csv

- Encoding: `utf-8`
- Inferred type: `fund_yield`
- Shape: `427` rows x `3` columns
- Columns: `mfd_date, mfd_daily_yield, mfd_7daily_yield`

### Date Columns

- `mfd_date`: 2013-07-01 to 2014-08-31

### Missing Values

- `mfd_date`: 0
- `mfd_daily_yield`: 0
- `mfd_7daily_yield`: 0

### Head

| mfd_date | mfd_daily_yield | mfd_7daily_yield |
| --- | --- | --- |
| 20130701.0 | 1.5787 | 6.307 |
| 20130702.0 | 1.5461 | 6.174 |
| 20130703.0 | 1.467 | 6.034 |
| 20130704.0 | 1.4223 | 5.903 |
| 20130705.0 | 1.3845 | 5.739 |

## user_balance_table.csv

- Encoding: `utf-8`
- Inferred type: `user_balance`
- Shape: `2840421` rows x `18` columns
- Columns: `user_id, report_date, tBalance, yBalance, total_purchase_amt, direct_purchase_amt, purchase_bal_amt, purchase_bank_amt, total_redeem_amt, consume_amt, transfer_amt, tftobal_amt, tftocard_amt, share_amt, category1, category2, category3, category4`

### Date Columns

- `report_date`: 2013-07-01 to 2014-08-31

### Missing Values

- `user_id`: 0
- `report_date`: 0
- `tBalance`: 0
- `yBalance`: 0
- `total_purchase_amt`: 0
- `direct_purchase_amt`: 0
- `purchase_bal_amt`: 0
- `purchase_bank_amt`: 0
- `total_redeem_amt`: 0
- `consume_amt`: 0
- `transfer_amt`: 0
- `tftobal_amt`: 0
- `tftocard_amt`: 0
- `share_amt`: 0
- `category1`: 2666682
- `category2`: 2666682
- `category3`: 2666682
- `category4`: 2666682

### Head

| user_id | report_date | tBalance | yBalance | total_purchase_amt | direct_purchase_amt | purchase_bal_amt | purchase_bank_amt | total_redeem_amt | consume_amt | transfer_amt | tftobal_amt | tftocard_amt | share_amt | category1 | category2 | category3 | category4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 20140805.0 | 20385.0 | 20383.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.0 | nan | nan | nan | nan |
| 1.0 | 20140808.0 | 20391.0 | 20389.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.0 | nan | nan | nan | nan |
| 1.0 | 20140811.0 | 20397.0 | 20395.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.0 | nan | nan | nan | nan |
| 1.0 | 20140814.0 | 20403.0 | 20401.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.0 | nan | nan | nan | nan |
| 1.0 | 20140817.0 | 20409.0 | 20407.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.0 | nan | nan | nan | nan |

## user_profile_table.csv

- Encoding: `utf-8`
- Inferred type: `user_profile`
- Shape: `28041` rows x `4` columns
- Columns: `user_id, sex, city, constellation`

### Date Columns

- No likely date columns detected.

### Missing Values

- `user_id`: 0
- `sex`: 0
- `city`: 0
- `constellation`: 0

### Head

| user_id | sex | city | constellation |
| --- | --- | --- | --- |
| 2 | 1 | 6411949 | 狮子座 |
| 12 | 1 | 6412149 | 摩羯座 |
| 22 | 1 | 6411949 | 双子座 |
| 23 | 1 | 6411949 | 双鱼座 |
| 25 | 1 | 6481949 | 双鱼座 |
