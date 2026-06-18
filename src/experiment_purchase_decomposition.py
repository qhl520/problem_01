from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, PREDICT_END_DATE, PREDICT_START_DATE, RAW_DATA_DIR
from data_utils import find_user_balance_file, read_csv_safely, read_submission, validate_submission
from generate_131_candidates import BASELINE_PATH, ensure_baseline_131, round_to_total


def _same_weekday_forecast(history: pd.DataFrame, target: str, dates: pd.DatetimeIndex, n: int = 8) -> np.ndarray:
    values = []
    hist = history.copy()
    hist["weekday"] = hist["date"].dt.weekday
    for dt in dates:
        subset = hist[hist["weekday"] == dt.weekday()].tail(n)
        if subset.empty:
            subset = hist.tail(30)
        weights = np.linspace(0.7, 1.3, len(subset))
        values.append(float(np.average(subset[target], weights=weights)))
    return np.asarray(values)


def build_purchase_decomposition_candidate() -> pd.DataFrame:
    ensure_baseline_131()
    path = find_user_balance_file(RAW_DATA_DIR)
    if path is None:
        raise FileNotFoundError("Missing user_balance_table.csv under data/raw/.")

    usecols = ["report_date", "total_purchase_amt", "direct_purchase_amt", "share_amt"]
    df, _ = read_csv_safely(path, usecols=usecols)
    daily = (
        df.groupby("report_date", as_index=False)
        .agg(
            total_purchase_amt=("total_purchase_amt", "sum"),
            direct_purchase_amt=("direct_purchase_amt", "sum"),
            share_amt=("share_amt", "sum"),
        )
    )
    daily["date"] = pd.to_datetime(daily["report_date"].astype(str), format="%Y%m%d")
    daily = daily[daily["date"] <= pd.Timestamp("2014-08-31")].sort_values("date")

    future_dates = pd.date_range(PREDICT_START_DATE, PREDICT_END_DATE, freq="D")
    direct_pred = _same_weekday_forecast(daily, "direct_purchase_amt", future_dates, n=8)
    share_base = daily["share_amt"].tail(30).ewm(span=10, adjust=False).mean().iloc[-1]
    share_pred = np.full(len(future_dates), float(share_base))
    purchase_pred = np.clip(direct_pred + share_pred, 0, None)

    baseline = read_submission(BASELINE_PATH)
    target_total = int(baseline["purchase"].sum())
    baseline["purchase"] = round_to_total(purchase_pred / purchase_pred.sum() * target_total, target_total)
    return baseline[["report_date", "purchase", "redeem"]]


def main() -> None:
    out_dir = OUTPUT_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "purchase_decomposition_candidate.csv"
    candidate = build_purchase_decomposition_candidate()
    candidate.to_csv(out_path, index=False, header=False)
    validate_submission(out_path)
    print(f"Saved optional purchase decomposition candidate to: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
