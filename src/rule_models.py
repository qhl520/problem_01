from __future__ import annotations

import pandas as pd

from config import OUTPUT_DIR, PREDICT_END_DATE, PREDICT_START_DATE
from evaluate import mape, weighted_score


def _weighted_center(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(0.6 * values.median() + 0.4 * values.mean())


def predict_weekday_rule(
    history_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    n: int = 8,
    recursive: bool = True,
) -> pd.DataFrame:
    history = history_df.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date").reset_index(drop=True)
    preds = []
    for date in pd.date_range(start_date, end_date, freq="D"):
        past = history[history["date"] < date]
        row = {"date": date}
        for target in ["purchase", "redeem"]:
            same_weekday = past[past["date"].dt.weekday == date.weekday()].tail(n)[target]
            if same_weekday.empty:
                recent = past.tail(min(28, len(past)))[target]
                value = recent.mean()
            else:
                value = _weighted_center(same_weekday)
            if date.day <= 3:
                value *= 1.03
            if date.day >= 28:
                value *= 1.02
            if pd.Timestamp("2014-09-06") <= date <= pd.Timestamp("2014-09-08"):
                value *= 0.92
            row[target] = max(0, int(round(value)))
        preds.append(row)
        if recursive:
            history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    return pd.DataFrame(preds)


def predict_recent_average_rule(history_df: pd.DataFrame, start_date: str, end_date: str, window: int) -> pd.DataFrame:
    history = history_df.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date").reset_index(drop=True)
    preds = []
    for date in pd.date_range(start_date, end_date, freq="D"):
        past = history[history["date"] < date].tail(window)
        row = {"date": date}
        for target in ["purchase", "redeem"]:
            row[target] = max(0, int(round(past[target].mean())))
        preds.append(row)
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    return pd.DataFrame(preds)


def to_submission(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out["report_date"] = pd.to_datetime(out["date"]).dt.strftime("%Y%m%d").astype(int)
    out["purchase"] = out["purchase"].round().clip(lower=0).astype("int64")
    out["redeem"] = out["redeem"].round().clip(lower=0).astype("int64")
    return out[["report_date", "purchase", "redeem"]]


def validate_rule_models() -> pd.DataFrame:
    daily = pd.read_csv("data/processed/daily_balance.csv", parse_dates=["date"])
    folds = [
        ("2014-05-31", "2014-06-01", "2014-06-30"),
        ("2014-06-30", "2014-07-01", "2014-07-31"),
        ("2014-07-31", "2014-08-01", "2014-08-31"),
    ]
    rows = []
    for train_end, valid_start, valid_end in folds:
        history = daily[daily["date"] <= train_end]
        actual = daily[(daily["date"] >= valid_start) & (daily["date"] <= valid_end)]
        for name, pred in [
            ("weekday_rule", predict_weekday_rule(history, valid_start, valid_end)),
            ("recent_7_mean", predict_recent_average_rule(history, valid_start, valid_end, 7)),
            ("recent_14_mean", predict_recent_average_rule(history, valid_start, valid_end, 14)),
            ("recent_28_mean", predict_recent_average_rule(history, valid_start, valid_end, 28)),
        ]:
            rows.append(
                {
                    "model": name,
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
    result.to_csv(OUTPUT_DIR / "rule_model_validation.csv", index=False)
    print(result)
    return result


if __name__ == "__main__":
    validate_rule_models()
