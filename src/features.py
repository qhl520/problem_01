from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from pandas.errors import PerformanceWarning

from config import DIAGNOSTIC_DIR, RAW_DATA_DIR
from data_utils import parse_competition_date, read_csv_safely


warnings.filterwarnings("ignore", category=PerformanceWarning)


HOLIDAYS = {
    "2013-09-19",
    "2013-09-20",
    "2013-09-21",
    "2013-10-01",
    "2013-10-02",
    "2013-10-03",
    "2013-10-04",
    "2013-10-05",
    "2013-10-06",
    "2013-10-07",
    "2014-01-31",
    "2014-02-01",
    "2014-02-02",
    "2014-02-03",
    "2014-02-04",
    "2014-02-05",
    "2014-02-06",
    "2014-04-05",
    "2014-04-06",
    "2014-04-07",
    "2014-05-01",
    "2014-05-02",
    "2014-05-03",
    "2014-05-31",
    "2014-06-01",
    "2014-06-02",
    "2014-09-06",
    "2014-09-07",
    "2014-09-08",
    "2014-10-01",
    "2014-10-02",
    "2014-10-03",
}
MID_AUTUMN = {"2013-09-19", "2014-09-08"}
NATIONAL_DAY = {"2013-10-01", "2014-10-01"}
SPRING_FESTIVAL = {"2014-01-31"}


def _date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date = out["date"]
    out["day"] = date.dt.day
    out["month"] = date.dt.month
    out["weekday"] = date.dt.weekday
    out["weekofyear"] = date.dt.isocalendar().week.astype(int)
    out["is_weekend"] = out["weekday"].isin([5, 6]).astype(int)
    out["is_month_start"] = date.dt.is_month_start.astype(int)
    out["is_month_end"] = date.dt.is_month_end.astype(int)
    out["days_from_month_start"] = date.dt.day - 1
    out["days_to_month_end"] = date.dt.days_in_month - date.dt.day
    out["is_first_3_days"] = (date.dt.day <= 3).astype(int)
    out["is_last_3_days"] = (out["days_to_month_end"] <= 2).astype(int)

    date_str = date.dt.strftime("%Y-%m-%d")
    holiday_dates = pd.to_datetime(sorted(HOLIDAYS))
    out["is_holiday"] = date_str.isin(HOLIDAYS).astype(int)
    out["is_before_holiday"] = date.isin(holiday_dates - pd.Timedelta(days=1)).astype(int)
    out["is_after_holiday"] = date.isin(holiday_dates + pd.Timedelta(days=1)).astype(int)
    out["is_mid_autumn"] = date_str.isin(MID_AUTUMN).astype(int)
    out["is_national_day"] = date_str.isin(NATIONAL_DAY).astype(int)
    out["is_spring_festival"] = date_str.isin(SPRING_FESTIVAL).astype(int)

    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)
    out["day_sin"] = np.sin(2 * np.pi * out["day"] / 31)
    out["day_cos"] = np.cos(2 * np.pi * out["day"] / 31)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def _yuebao_yield_display_delay_days(date: pd.Series) -> pd.Series:
    # Competition simplified rule: Mon->Wed, Tue->Thu, Wed->Fri,
    # Thu->Sat, Fri->next Tue, Sat/Sun->next Wed.
    mapping = {0: 2, 1: 2, 2: 2, 3: 2, 4: 4, 5: 4, 6: 3}
    return date.dt.weekday.map(mapping).astype(int)


def _target_history_features(out: pd.DataFrame) -> pd.DataFrame:
    for target in ["purchase", "redeem"]:
        for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]:
            out[f"{target}_lag_{lag}"] = out[target].shift(lag)
        shifted = out[target].shift(1)
        for window in [3, 7, 14, 21, 28]:
            out[f"{target}_rolling_{window}_mean"] = shifted.rolling(window).mean()
            out[f"{target}_rolling_{window}_median"] = shifted.rolling(window).median()
            out[f"{target}_rolling_{window}_std"] = shifted.rolling(window).std()
            out[f"{target}_rolling_{window}_max"] = shifted.rolling(window).max()
            out[f"{target}_rolling_{window}_min"] = shifted.rolling(window).min()
            out[f"{target}_rolling_{window}_sum"] = shifted.rolling(window).sum()
        for idx, lag in enumerate([7, 14, 21, 28], start=1):
            out[f"{target}_same_weekday_last_{idx}"] = out[target].shift(lag)
        same_weekday_4 = pd.concat([out[target].shift(lag) for lag in [7, 14, 21, 28]], axis=1)
        same_weekday_8 = pd.concat([out[target].shift(lag) for lag in [7, 14, 21, 28, 35, 42, 49, 56]], axis=1)
        out[f"{target}_same_weekday_mean_4"] = same_weekday_4.mean(axis=1)
        out[f"{target}_same_weekday_median_4"] = same_weekday_4.median(axis=1)
        out[f"{target}_same_weekday_mean_8"] = same_weekday_8.mean(axis=1)
        out[f"{target}_same_weekday_median_8"] = same_weekday_8.median(axis=1)

    out["purchase_minus_redeem_lag_1"] = out["purchase"].shift(1) - out["redeem"].shift(1)
    out["purchase_redeem_ratio_lag_1"] = out["purchase"].shift(1) / out["redeem"].shift(1).replace(0, np.nan)
    out["purchase_diff_1"] = out["purchase"].diff(1).shift(1)
    out["redeem_diff_1"] = out["redeem"].diff(1).shift(1)
    out["purchase_pct_change_1"] = out["purchase"].pct_change(1, fill_method=None).shift(1).replace([np.inf, -np.inf], np.nan)
    out["redeem_pct_change_1"] = out["redeem"].pct_change(1, fill_method=None).shift(1).replace([np.inf, -np.inf], np.nan)
    out["yuebao_yield_display_delay_days"] = _yuebao_yield_display_delay_days(out["date"])
    return out


def _load_finance_features() -> pd.DataFrame | None:
    frames = []
    for path in RAW_DATA_DIR.glob("*.csv"):
        try:
            sample, _ = read_csv_safely(path, nrows=5)
        except Exception:
            continue
        if "mfd_date" not in sample.columns:
            continue
        cols = set(sample.columns)
        if not ({"mfd_daily_yield", "mfd_7daily_yield"} & cols or any(c.startswith("Interest_") for c in cols)):
            continue
        df, _ = read_csv_safely(path)
        df["date"] = parse_competition_date(df["mfd_date"])
        df = df.drop(columns=["mfd_date"])
        frames.append(df)
    if not frames:
        return None
    finance = frames[0]
    for extra in frames[1:]:
        finance = finance.merge(extra, on="date", how="outer")
    return finance.sort_values("date")


def _add_finance_lags(out: pd.DataFrame) -> pd.DataFrame:
    finance = _load_finance_features()
    if finance is None:
        return out
    full = out[["date"]].merge(finance, on="date", how="left").sort_values("date")
    finance_cols = [c for c in full.columns if c != "date"]
    full[finance_cols] = full[finance_cols].ffill()
    report_rows = []
    for col in finance_cols:
        report_rows.append({"feature": col, "missing_rate_after_ffill": full[col].isna().mean()})
        for lag in [1, 3, 7]:
            out[f"{col}_lag_{lag}"] = full[col].shift(lag).values
    if "mfd_daily_yield" in full.columns:
        display_delay = _yuebao_yield_display_delay_days(out["date"])
        out["mfd_daily_yield_by_display_rule"] = [
            full["mfd_daily_yield"].shift(int(delay)).iloc[i] for i, delay in enumerate(display_delay)
        ]
    if "mfd_7daily_yield" in full.columns:
        display_delay = _yuebao_yield_display_delay_days(out["date"])
        out["mfd_7daily_yield_by_display_rule"] = [
            full["mfd_7daily_yield"].shift(int(delay)).iloc[i] for i, delay in enumerate(display_delay)
        ]
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(DIAGNOSTIC_DIR / "finance_feature_missing_report.csv", index=False)
    return out


def make_features(df: pd.DataFrame, use_finance: bool = True, use_profile: bool = True, outlier_mode: str = "none") -> pd.DataFrame:
    del use_profile, outlier_mode
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out = _date_features(out)
    out = _target_history_features(out)
    if use_finance:
        out = _add_finance_lags(out)
    return out.replace([np.inf, -np.inf], np.nan)


def get_training_frame(df: pd.DataFrame, use_finance: bool = True) -> pd.DataFrame:
    features = make_features(df, use_finance=use_finance)
    return features.dropna(subset=["purchase", "redeem"]).reset_index(drop=True)
