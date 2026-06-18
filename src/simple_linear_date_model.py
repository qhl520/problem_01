from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import OUTPUT_DIR, PREDICT_END_DATE, PREDICT_START_DATE, PROCESSED_DATA_DIR, RAW_DATA_DIR
from data_utils import load_daily_balance_fallback, validate_submission


MID_AUTUMN_DATES = {"2014-09-06", "2014-09-07", "2014-09-08", "2014-09-09"}
NATIONAL_DAY_PRE = {"2014-09-25", "2014-09-26", "2014-09-27", "2014-09-28", "2014-09-29", "2014-09-30"}
PROMO_DATE = "2014-09-09"


def build_calendar_features(dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    dates = pd.to_datetime(pd.Series(dates), errors="coerce")
    weekday = dates.dt.weekday
    features = pd.DataFrame(
        {
            "weekend": weekday.isin([5, 6]).astype(int),
            "day_of_month": dates.dt.day.astype(int),
            "month_start": dates.dt.day.isin([1, 2, 3, 4, 5]).astype(int),
            "month_middle": dates.dt.day.between(11, 20).astype(int),
            "month_end": dates.dt.day.ge(25).astype(int),
            "holiday": dates.dt.strftime("%Y-%m-%d").isin(MID_AUTUMN_DATES).astype(int),
            "pre_holiday": dates.dt.strftime("%Y-%m-%d").isin(["2014-09-05", "2014-09-26", "2014-09-27"]).astype(int),
            "after_holiday": dates.dt.strftime("%Y-%m-%d").isin(["2014-09-09"]).astype(int),
            "national_day_pre": dates.dt.strftime("%Y-%m-%d").isin(NATIONAL_DAY_PRE).astype(int),
            "mid_autumn": dates.dt.strftime("%Y-%m-%d").isin(MID_AUTUMN_DATES).astype(int),
            "promo_20140909": dates.dt.strftime("%Y-%m-%d").eq(PROMO_DATE).astype(int),
        }
    )
    for day in range(7):
        features[f"weekday_{day}"] = (weekday == day).astype(int)
    return features


def _round_to_total(values: np.ndarray, target_total: int) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0, None)
    floors = np.floor(values).astype("int64")
    diff = int(target_total - floors.sum())
    if diff > 0:
        order = np.argsort(-(values - floors))
        floors[order[:diff]] += 1
    elif diff < 0:
        order = np.argsort(values - floors)
        for idx in order:
            if diff == 0:
                break
            take = min(floors[idx], -diff)
            floors[idx] -= take
            diff += take
    return floors


def predict_simple_linear_date_model(
    normalize_totals: dict[str, int] | None = None,
    start: str = PREDICT_START_DATE,
    end: str = PREDICT_END_DATE,
) -> pd.DataFrame:
    """Weak calendar-only Ridge model used for low-weight candidate blending."""
    daily = load_daily_balance_fallback(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    daily = daily[daily["date"] <= pd.Timestamp("2014-08-31")].copy()
    daily["date"] = pd.to_datetime(daily["date"])

    x_train = build_calendar_features(daily["date"])
    future_dates = pd.date_range(start, end, freq="D")
    x_future = build_calendar_features(future_dates)

    pred = pd.DataFrame({"report_date": future_dates.strftime("%Y%m%d").astype(int)})
    for target in ["purchase", "redeem"]:
        y = np.log1p(daily[target].clip(lower=0).astype(float))
        model = make_pipeline(StandardScaler(), Ridge(alpha=8.0))
        model.fit(x_train, y)
        values = np.expm1(model.predict(x_future))
        values = np.clip(values, 0, None)
        if normalize_totals and target in normalize_totals and values.sum() > 0:
            values = values / values.sum() * int(normalize_totals[target])
            pred[target] = _round_to_total(values, int(normalize_totals[target]))
        else:
            pred[target] = np.rint(values).astype("int64")
    return pred[["report_date", "purchase", "redeem"]]


def main() -> None:
    output_path = OUTPUT_DIR / "experiments" / "simple_linear_date_model.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred = predict_simple_linear_date_model()
    pred.to_csv(output_path, index=False, header=False)
    validate_submission(output_path)
    print(f"Saved optional weak-model prediction to: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
