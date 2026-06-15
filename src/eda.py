from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import EDA_OUTPUT_DIR, PROCESSED_DATA_DIR, ensure_directories


def _load_daily() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "daily_balance.csv"
    if not path.exists():
        from make_daily_data import make_daily_data

        return make_daily_data()
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _plot_group(df: pd.DataFrame, key: str, path_name: str, title: str) -> None:
    grouped = df.groupby(key)[["purchase", "redeem"]].mean()
    ax = grouped.plot(kind="bar", figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel(key)
    ax.set_ylabel("Mean amount")
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR / path_name, dpi=160)
    plt.close()


def run_eda() -> None:
    ensure_directories()
    df = _load_daily()
    df["weekday"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["purchase"], label="purchase")
    plt.plot(df["date"], df["redeem"], label="redeem")
    plt.title("Daily Purchase and Redeem")
    plt.xlabel("Date")
    plt.ylabel("Amount")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR / "daily_purchase_redeem.png", dpi=160)
    plt.close()

    _plot_group(df, "weekday", "weekday_mean.png", "Mean Amount by Weekday")
    _plot_group(df, "month", "month_mean.png", "Mean Amount by Month")
    _plot_group(df, "day", "dayofmonth_mean.png", "Mean Amount by Day of Month")

    for target in ["purchase", "redeem"]:
        plt.figure(figsize=(8, 5))
        plt.hist(df[target], bins=40)
        plt.title(f"{target.title()} Distribution")
        plt.xlabel(target)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(EDA_OUTPUT_DIR / f"{target}_distribution.png", dpi=160)
        plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(df["purchase"], df["redeem"], alpha=0.7)
    plt.title("Purchase Redeem Correlation")
    plt.xlabel("purchase")
    plt.ylabel("redeem")
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR / "purchase_redeem_correlation.png", dpi=160)
    plt.close()

    outlier_rows = []
    for target in ["purchase", "redeem"]:
        z = (df[target] - df[target].mean()) / df[target].std(ddof=0)
        outlier_rows.append(df.loc[np.abs(z) > 3, ["date", "purchase", "redeem"]])
    outliers = pd.concat(outlier_rows).drop_duplicates().sort_values("date")
    outliers.to_csv(EDA_OUTPUT_DIR / "outliers.csv", index=False)

    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    missing_dates = sorted(set(full_dates) - set(df["date"]))
    summary = pd.DataFrame(
        [
            {"metric": "start_date", "value": df["date"].min().strftime("%Y-%m-%d")},
            {"metric": "end_date", "value": df["date"].max().strftime("%Y-%m-%d")},
            {"metric": "total_days", "value": len(df)},
            {"metric": "missing_dates", "value": len(missing_dates)},
            {"metric": "purchase_mean", "value": df["purchase"].mean()},
            {"metric": "redeem_mean", "value": df["redeem"].mean()},
            {"metric": "purchase_redeem_corr", "value": df["purchase"].corr(df["redeem"])},
        ]
    )
    summary.to_csv(EDA_OUTPUT_DIR / "eda_summary.csv", index=False)
    print(f"EDA outputs written to: {EDA_OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda()
