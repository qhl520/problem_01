from __future__ import annotations

import pandas as pd

from config import DIAGNOSTIC_DIR, RAW_DATA_DIR
from data_utils import find_user_balance_file, parse_competition_date, read_csv_safely


def main() -> None:
    path = find_user_balance_file(RAW_DATA_DIR)
    if path is None:
        raise FileNotFoundError("No user_balance_table-like file found in data/raw/.")

    required_cols = [
        "report_date",
        "tBalance",
        "yBalance",
        "total_purchase_amt",
        "total_redeem_amt",
        "consume_amt",
        "category1",
        "category2",
        "category3",
        "category4",
    ]
    df, _ = read_csv_safely(path, usecols=required_cols)
    df["date"] = parse_competition_date(df["report_date"])

    balance_expected = df["yBalance"] + df["total_purchase_amt"] - df["total_redeem_amt"]
    balance_diff = df["tBalance"] - balance_expected
    mismatch = df.loc[balance_diff != 0, required_cols].copy()
    if not mismatch.empty:
        mismatch["expected_tBalance"] = balance_expected.loc[mismatch.index]
        mismatch["balance_diff"] = balance_diff.loc[mismatch.index]
    amount_cols = ["tBalance", "yBalance", "total_purchase_amt", "total_redeem_amt", "consume_amt"]
    category_cols = ["category1", "category2", "category3", "category4"]

    consume_zero = df["consume_amt"] == 0
    category_non_empty_when_consume_zero = df.loc[consume_zero, category_cols].notna().any(axis=1).sum()
    consume_positive = df["consume_amt"] > 0
    category_all_empty_when_consume_positive = df.loc[consume_positive, category_cols].isna().all(axis=1).sum()

    rows = [
        {"check": "rows", "value": len(df), "status": "info"},
        {"check": "date_min", "value": df["date"].min().strftime("%Y-%m-%d"), "status": "info"},
        {"check": "date_max", "value": df["date"].max().strftime("%Y-%m-%d"), "status": "info"},
        {"check": "invalid_report_date_count", "value": int(df["date"].isna().sum()), "status": "pass" if df["date"].notna().all() else "fail"},
        {"check": "balance_equation_mismatch_count", "value": int((balance_diff != 0).sum()), "status": "pass" if (balance_diff == 0).all() else "warn"},
        {"check": "negative_amount_count", "value": int((df[amount_cols] < 0).sum().sum()), "status": "pass" if (df[amount_cols] >= 0).all().all() else "fail"},
        {
            "check": "category_non_empty_when_consume_zero_count",
            "value": int(category_non_empty_when_consume_zero),
            "status": "pass" if category_non_empty_when_consume_zero == 0 else "warn",
        },
        {
            "check": "category_all_empty_when_consume_positive_count",
            "value": int(category_all_empty_when_consume_positive),
            "status": "pass" if category_all_empty_when_consume_positive == 0 else "warn",
        },
    ]

    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out_path = DIAGNOSTIC_DIR / "business_rule_check.csv"
    out.to_csv(out_path, index=False)
    if not mismatch.empty:
        mismatch.to_csv(DIAGNOSTIC_DIR / "business_rule_balance_mismatch_examples.csv", index=False)
    print(out)
    print(f"Business rule check written to: {out_path}")


if __name__ == "__main__":
    main()
