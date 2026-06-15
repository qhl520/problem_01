from __future__ import annotations

import pandas as pd

from config import OUTPUT_DIR, PROCESSED_DATA_DIR
from evaluate import mape, weighted_score
from rule_models import predict_weekday_rule


def main() -> None:
    daily = pd.read_csv(PROCESSED_DATA_DIR / "daily_balance.csv", parse_dates=["date"])
    folds = [
        ("2014-05-31", "2014-06-01", "2014-06-30"),
        ("2014-06-30", "2014-07-01", "2014-07-31"),
        ("2014-07-31", "2014-08-01", "2014-08-31"),
    ]
    rows = []
    for train_end, valid_start, valid_end in folds:
        history = daily[daily["date"] <= train_end]
        actual = daily[(daily["date"] >= valid_start) & (daily["date"] <= valid_end)]
        pred = predict_weekday_rule(history, valid_start, valid_end, n=8)
        rows.append(
            {
                "train_end": train_end,
                "valid_start": valid_start,
                "valid_end": valid_end,
                "purchase_mape": mape(actual["purchase"], pred["purchase"]),
                "redeem_mape": mape(actual["redeem"], pred["redeem"]),
                "weighted_score": weighted_score(
                    actual["purchase"], pred["purchase"], actual["redeem"], pred["redeem"]
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "validation_rule_baseline.csv", index=False)
    print(result)


if __name__ == "__main__":
    main()
