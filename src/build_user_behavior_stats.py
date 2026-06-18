from __future__ import annotations

import sys

import pandas as pd

from config import OUTPUT_DIR, RAW_DATA_DIR
from data_utils import find_user_balance_file, read_csv_safely


def build_user_behavior_stats() -> pd.DataFrame:
    path = find_user_balance_file(RAW_DATA_DIR)
    if path is None:
        raise FileNotFoundError("Missing user_balance_table.csv under data/raw/.")

    usecols = ["user_id", "report_date", "total_purchase_amt", "total_redeem_amt"]
    df, _ = read_csv_safely(path, usecols=usecols)
    df["has_purchase"] = df["total_purchase_amt"].fillna(0).gt(0)
    df["has_redeem"] = df["total_redeem_amt"].fillna(0).gt(0)
    df["has_any"] = df["has_purchase"] | df["has_redeem"]

    grouped = df.groupby("report_date", as_index=False)
    stats = grouped.agg(
        active_users=("user_id", "nunique"),
        purchase_user_count=("has_purchase", "sum"),
        redeem_user_count=("has_redeem", "sum"),
        purchase_times=("has_purchase", "sum"),
        redeem_times=("has_redeem", "sum"),
        total_purchase_amt=("total_purchase_amt", "sum"),
        total_redeem_amt=("total_redeem_amt", "sum"),
    )

    overlap = (
        df[df["has_purchase"] & df["has_redeem"]]
        .groupby("report_date")["user_id"]
        .nunique()
        .rename("purchase_redeem_overlap_users")
        .reset_index()
    )
    stats = stats.merge(overlap, on="report_date", how="left")
    stats["purchase_redeem_overlap_users"] = stats["purchase_redeem_overlap_users"].fillna(0).astype("int64")
    stats["avg_purchase_per_user"] = (
        stats["total_purchase_amt"] / stats["purchase_user_count"].replace(0, pd.NA)
    ).fillna(0)
    stats["avg_redeem_per_user"] = (
        stats["total_redeem_amt"] / stats["redeem_user_count"].replace(0, pd.NA)
    ).fillna(0)

    keep = [
        "report_date",
        "active_users",
        "purchase_user_count",
        "redeem_user_count",
        "purchase_times",
        "redeem_times",
        "purchase_redeem_overlap_users",
        "avg_purchase_per_user",
        "avg_redeem_per_user",
    ]
    return stats[keep].sort_values("report_date").reset_index(drop=True)


def main() -> None:
    out_dir = OUTPUT_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "user_behavior_daily.csv"
    stats = build_user_behavior_stats()
    stats.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved user behavior stats to: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
