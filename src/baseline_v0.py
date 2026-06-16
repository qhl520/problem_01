"""DEPRECATED: V0 baseline — copies August 2014 values one month forward.

A naive baseline that predicts each September day using the same day-of-month
from August 2014 (or the August mean when no match exists).

Superseded by the rule models in ``src/rule_models.py`` and the canonical
ensemble pipeline ``src/predict_final.py``.

Kept for reference only; not used by ``run_all.py`` or ``run_original_plus_121.py``.
"""

from __future__ import annotations

import shutil

import pandas as pd

from config import OUTPUT_DIR, PREDICT_END_DATE, PREDICT_START_DATE, PROCESSED_DATA_DIR, SUBMISSION_DIR
from data_utils import validate_submission
from rule_models import to_submission


def main() -> None:
    daily = pd.read_csv(PROCESSED_DATA_DIR / "daily_balance.csv", parse_dates=["date"])
    august = daily[(daily["date"] >= "2014-08-01") & (daily["date"] <= "2014-08-31")].copy()
    august_mean = august[["purchase", "redeem"]].mean()
    rows = []
    for date in pd.date_range(PREDICT_START_DATE, PREDICT_END_DATE, freq="D"):
        source_date = date - pd.DateOffset(months=1)
        source = august[august["date"] == source_date]
        if source.empty:
            purchase, redeem = august_mean["purchase"], august_mean["redeem"]
        else:
            purchase, redeem = source[["purchase", "redeem"]].iloc[0]
        rows.append({"date": date, "purchase": purchase, "redeem": redeem})
    sub = to_submission(pd.DataFrame(rows))
    out_path = SUBMISSION_DIR / "submission_v0_august_copy.csv"
    root_copy = OUTPUT_DIR / "submission_v0_august_copy.csv"
    sub.to_csv(out_path, index=False, header=False)
    shutil.copyfile(out_path, root_copy)
    validate_submission(out_path)
    print(f"V0 submission written to: {out_path}")


if __name__ == "__main__":
    main()
