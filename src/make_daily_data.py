from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import EDA_OUTPUT_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directories
from data_utils import find_user_balance_file, parse_competition_date, read_csv_safely


def make_daily_data() -> pd.DataFrame:
    ensure_directories()
    path = find_user_balance_file(RAW_DATA_DIR)
    if path is None:
        raise FileNotFoundError("No user balance CSV found in data/raw/.")

    usecols = ["report_date", "total_purchase_amt", "total_redeem_amt"]
    df, _ = read_csv_safely(path)
    missing = set(usecols) - set(df.columns)
    if missing:
        raise ValueError(f"User balance file is missing required columns: {sorted(missing)}")

    df = df[usecols].copy()
    df["date"] = parse_competition_date(df["report_date"])
    if df["date"].isna().any():
        raise ValueError("Failed to parse some report_date values.")

    daily = (
        df.groupby("date", as_index=False)
        .agg(purchase=("total_purchase_amt", "sum"), redeem=("total_redeem_amt", "sum"))
        .sort_values("date")
    )
    full_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(full_dates).rename_axis("date").reset_index()
    if daily[["purchase", "redeem"]].isna().any().any():
        missing_dates = daily.loc[daily["purchase"].isna(), "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Daily data has missing dates after aggregation: {missing_dates[:10]}")
    if (daily[["purchase", "redeem"]] < 0).any().any():
        raise ValueError("Daily data contains negative purchase/redeem values.")
    if daily["date"].duplicated().any():
        raise ValueError("Daily data contains duplicate dates.")

    output_path = PROCESSED_DATA_DIR / "daily_balance.csv"
    daily.to_csv(output_path, index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(daily["date"], daily["purchase"], label="purchase")
    plt.plot(daily["date"], daily["redeem"], label="redeem")
    plt.title("Daily Purchase and Redeem")
    plt.xlabel("Date")
    plt.ylabel("Amount")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR / "daily_purchase_redeem.png", dpi=160)
    plt.close()

    print(f"Daily data written to: {output_path}")
    print(f"Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
    print(f"Rows: {len(daily)}")
    return daily


if __name__ == "__main__":
    make_daily_data()
