from __future__ import annotations

import pandas as pd

from config import PREDICT_END_DATE, PREDICT_START_DATE, PROCESSED_DATA_DIR, SUBMISSION_DIR
from data_utils import validate_submission
from rule_models import predict_weekday_rule, to_submission


def main() -> None:
    daily = pd.read_csv(PROCESSED_DATA_DIR / "daily_balance.csv", parse_dates=["date"])
    pred = predict_weekday_rule(daily, PREDICT_START_DATE, PREDICT_END_DATE, n=8)
    for weekday in range(7):
        count = len(daily[daily["date"].dt.weekday == weekday].tail(8))
        print(f"weekday={weekday}, history_samples={count}")
    sub = to_submission(pred)
    out_path = SUBMISSION_DIR / "submission_v1_weekday_rule.csv"
    sub.to_csv(out_path, index=False, header=False)
    validate_submission(out_path)
    print(f"V1 submission written to: {out_path}")


if __name__ == "__main__":
    main()
