from __future__ import annotations

import pandas as pd

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from data_utils import find_user_balance_file, parse_competition_date, read_csv_safely


def main() -> None:
    balance_path = find_user_balance_file(RAW_DATA_DIR)
    profile_path = RAW_DATA_DIR / "user_profile_table.csv"
    if balance_path is None or not profile_path.exists():
        print("Profile feature inputs are not available; skipped.")
        return

    balance, _ = read_csv_safely(balance_path)
    profile, _ = read_csv_safely(profile_path)
    balance["date"] = parse_competition_date(balance["report_date"])
    merged = balance.merge(profile, on="user_id", how="left")
    group = merged.groupby("date")
    out = group.agg(
        active_user_count=("user_id", "nunique"),
        purchase_user_count=("total_purchase_amt", lambda x: int((x > 0).sum())),
        redeem_user_count=("total_redeem_amt", lambda x: int((x > 0).sum())),
        avg_purchase_per_user=("total_purchase_amt", "mean"),
        avg_redeem_per_user=("total_redeem_amt", "mean"),
        median_purchase_per_user=("total_purchase_amt", "median"),
        median_redeem_per_user=("total_redeem_amt", "median"),
    ).reset_index()
    if "sex" in merged.columns:
        sex_sum = merged.pivot_table(
            index="date",
            columns="sex",
            values=["total_purchase_amt", "total_redeem_amt"],
            aggfunc="sum",
            fill_value=0,
        )
        sex_sum.columns = [f"sex_{sex}_{target}" for target, sex in sex_sum.columns]
        out = out.merge(sex_sum.reset_index(), on="date", how="left")
    out.to_csv(PROCESSED_DATA_DIR / "profile_daily_features.csv", index=False)
    print("Profile daily features written.")


if __name__ == "__main__":
    main()
