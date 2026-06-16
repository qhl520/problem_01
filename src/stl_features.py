"""STL-decomposition features for ML models.

Extracts multiplicative STL components (S_week, S_month, trend, residual I)
from historical daily data so ML models can learn the irregular component
that pure STL cannot explain.

Core PPT technique:
    Y = f1 * S_week * S_month * f_feast * I
    → I = Y / (f1 * S_week * S_month * f_feast)

ML predicts Î; final prediction Ŷ = f1 * S_week * S_month * f_feast * Î.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants (mirror stl_model.py)
# ---------------------------------------------------------------------------
HOLIDAYS_2014 = {
    "2014-04-05", "2014-04-06", "2014-04-07",   # Qingming
    "2014-05-01", "2014-05-02", "2014-05-03",    # Labour Day
    "2014-05-31", "2014-06-01", "2014-06-02",    # Duanwu
    "2014-09-06", "2014-09-07", "2014-09-08",    # Mid-Autumn
}
QINGMING_DATES = ["2014-04-05", "2014-04-06", "2014-04-07"]
DUANWU_DATES = ["2014-05-31", "2014-06-01", "2014-06-02"]
MID_AUTUMN_DATES = ["2014-09-06", "2014-09-07", "2014-09-08"]
NATIONAL_DAY_PRE_DATES = ["2014-09-25", "2014-09-26", "2014-09-27", "2014-09-28", "2014-09-29", "2014-09-30"]
JIUJIU_PROMO_DATE = "2014-09-09"
JIUJIU_CONSUME_BOOST = 1.05

STL_SEASONAL = 13
MONTHLY_MONTHS = [4, 5, 6, 7, 8]


# ---------------------------------------------------------------------------
# 30-day month alignment
# ---------------------------------------------------------------------------

def align_to_30_day_month(series: pd.Series) -> pd.Series:
    """Align months to 30 days by averaging days 30 and 31."""
    s = series.copy()
    df = s.reset_index()
    df.columns = ["date", "value"]
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    pivot = df.pivot_table(index=["year", "month"], columns="day", values="value", aggfunc="mean")
    if 31 in pivot.columns:
        has_30 = 30 in pivot.columns
        if has_30:
            pivot[30] = pivot[[30, 31]].mean(axis=1)
        else:
            pivot[30] = pivot[31]
        pivot = pivot.drop(columns=[31])

    result = pivot.stack().reset_index()
    result.columns = ["year", "month", "day", "value"]
    result["date"] = pd.to_datetime(
        result["year"].astype(str) + "-" + result["month"].astype(str).str.zfill(2) + "-01"
    ) + pd.to_timedelta(result["day"].astype(int) - 1, unit="D")
    return result.set_index("date")["value"].sort_index()


# ---------------------------------------------------------------------------
# STL decomposition
# ---------------------------------------------------------------------------

def stl_decompose(series: pd.Series, period: int) -> pd.DataFrame:
    """Run robust STL and return trend, seasonal, resid, and mult_factor."""
    clean = series.dropna()
    stl = STL(clean, period=period, seasonal=STL_SEASONAL, robust=True)
    result = stl.fit()
    trend_safe = np.where(result.trend == 0, np.nan, result.trend)
    return pd.DataFrame({
        "trend": result.trend,
        "seasonal": result.seasonal,
        "resid": result.resid,
        "mult_factor": 1.0 + result.seasonal / trend_safe,
    }, index=clean.index)


def extract_weekday_factors(stl_weekly: pd.DataFrame) -> pd.Series:
    """Median S_week factor per weekday (0=Mon ... 6=Sun)."""
    df = stl_weekly.copy()
    df["weekday"] = df.index.weekday
    return df.groupby("weekday")["mult_factor"].median()


def extract_dayofmonth_factors(stl_monthly: pd.DataFrame) -> pd.Series:
    """Median S_month factor per day-of-month (1..30)."""
    df = stl_monthly.copy()
    df["day"] = df.index.day
    return df.groupby("day")["mult_factor"].median()


# ---------------------------------------------------------------------------
# Holiday effect from STL residuals
# ---------------------------------------------------------------------------

def compute_feast_from_residuals(
    stl_weekly: pd.DataFrame,
    ref_holidays: list[list[str]],
) -> float:
    """Compute mean multiplicative holiday effect from STL residuals.

    f_feast = mean(1 + resid/trend) over reference holiday dates.
    """
    effects = []
    for holiday_list in ref_holidays:
        mask = stl_weekly.index.isin(pd.to_datetime(holiday_list))
        if mask.sum() == 0:
            continue
        resid = stl_weekly.loc[mask, "resid"]
        trend = stl_weekly.loc[mask, "trend"]
        trend_safe = trend.replace(0, np.nan)
        eff = 1.0 + resid / trend_safe
        effects.extend(eff.dropna().tolist())
    if not effects:
        return 1.0
    return float(np.mean(effects))


# ---------------------------------------------------------------------------
# f_bigfeast: National Day pre-effect coefficient table
# ---------------------------------------------------------------------------

def compute_bigfeast_table(stl_weekly: pd.DataFrame) -> dict[str, float]:
    """Compute f_bigfeast coefficients for each Sep 25-30 date.

    Uses ALL historical months' last-6-day multiplicative residuals.
    Returns dict mapping date_str → factor.
    """
    df = stl_weekly.copy()
    df["year"] = df.index.year
    df["month"] = df.index.month

    trend_safe = df["trend"].replace(0, np.nan)
    df["mult_resid"] = 1.0 + df["resid"] / trend_safe

    offsets: dict[int, list[float]] = {i: [] for i in range(6)}
    for (_year, _month), grp in df.groupby(["year", "month"]):
        grp = grp.sort_index()
        last_6 = grp.tail(6)
        if len(last_6) < 6:
            continue
        for offset, (_, row) in enumerate(last_6.iterrows()):
            offsets[offset].append(row["mult_resid"])

    result = {}
    for i, date_str in enumerate(NATIONAL_DAY_PRE_DATES):
        if offsets[i]:
            result[date_str] = float(np.median(offsets[i]))
        else:
            result[date_str] = 1.0
    return result


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------

def extract_stl_features(
    daily: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """Extract STL decomposition features for a single target series.

    Args:
        daily: DataFrame with date index, containing per-capita columns
               like purchase_per_capita, consume_per_capita, etc.
        target: Short target name (e.g. 'purchase', 'consume', 'transfer').

    Returns:
        DataFrame with columns: S_week, S_month, stl_trend, stl_resid_I,
        f_feast, f_bigfeast, and the target column.
    """
    col = f"{target}_per_capita"
    if col not in daily.columns:
        raise KeyError(f"Column {col} not found in daily data")

    series = daily[col].dropna()
    dates = series.index

    # --- Weekly STL ---
    stl_w = stl_decompose(series, period=7)
    sw_factors = extract_weekday_factors(stl_w)

    # --- Monthly STL (on 30-day aligned data) ---
    series_30 = align_to_30_day_month(series)
    stl_m = stl_decompose(series_30, period=30)
    sm_factors = extract_dayofmonth_factors(stl_m)

    # --- Holiday effect ---
    feast_mean = compute_feast_from_residuals(
        stl_w, [QINGMING_DATES, DUANWU_DATES]
    )

    # --- Bigfeast table ---
    fb_table = compute_bigfeast_table(stl_w)

    # --- ARIMA f1: monthly mean of the last available month ---
    last_month_data = series[series.index.month == series.index[-1].month]
    if len(last_month_data) > 0:
        f1 = float(last_month_data.mean())
    else:
        f1 = float(series.tail(30).mean())

    # --- Build feature DataFrame ---
    out_data = {"date": dates, target: series.values}

    # S_week factor
    out_data["S_week"] = [float(sw_factors.get(d.weekday(), 1.0)) for d in dates]

    # S_month factor
    out_data["S_month"] = [
        float(sm_factors.get(min(d.day, 30), 1.0)) for d in dates
    ]

    # f_feast
    out_data["f_feast"] = 1.0
    for holiday_list in [MID_AUTUMN_DATES]:
        for ds in holiday_list:
            dt = pd.Timestamp(ds)
            if dt in dates:
                idx = dates.get_loc(dt)
                out_data["f_feast"] = out_data["f_feast"]  # needs special handling

    out = pd.DataFrame(out_data).set_index("date")

    # f_feast per date
    feast_series = pd.Series(1.0, index=dates)
    for i, ds in enumerate(MID_AUTUMN_DATES):
        dt = pd.Timestamp(ds)
        if dt in dates:
            if i == 2:  # peak day
                feast_series[dt] = feast_mean - (feast_mean - 1.0) * 0.5
            else:
                feast_series[dt] = feast_mean
    out["f_feast"] = feast_series.values

    # f_bigfeast per date
    fb_series = pd.Series(1.0, index=dates)
    for ds, factor in fb_table.items():
        dt = pd.Timestamp(ds)
        if dt in dates:
            fb_series[dt] = factor
    out["f_bigfeast"] = fb_series.values

    # STL residuals merged back
    out["stl_resid_I"] = 1.0
    for dt in stl_w.index:
        if dt in out.index:
            trend_val = stl_w.loc[dt, "trend"]
            resid_val = stl_w.loc[dt, "resid"]
            if trend_val != 0:
                out.loc[dt, "stl_resid_I"] = 1.0 + resid_val / trend_val

    out["stl_trend"] = np.nan
    for dt in stl_w.index:
        if dt in out.index:
            out.loc[dt, "stl_trend"] = stl_w.loc[dt, "trend"]

    # f1 (constant per month, but we store as a column)
    out["f1_monthly_mean"] = f1

    # The irregular component I = Y / (f1 * S_week * S_month * f_feast)
    # For historical months, f1 is the actual monthly mean of that month
    out["month"] = out.index.month
    out["year"] = out.index.year
    for (yr, mon), grp in out.groupby(["year", "month"]):
        month_mean = grp[target].mean()
        if month_mean > 0:
            out.loc[grp.index, "f1_monthly_mean"] = month_mean

    denom = (
        out["f1_monthly_mean"]
        * out["S_week"]
        * out["S_month"]
        * out["f_feast"]
    )
    denom_safe = denom.replace(0, np.nan)
    out["I_irregular"] = out[target] / denom_safe
    out["I_irregular"] = out["I_irregular"].clip(lower=0.1, upper=10.0)

    return out


def make_stl_enhanced_features(
    daily_df: pd.DataFrame,
    use_finance: bool = True,
) -> pd.DataFrame:
    """Create ML feature matrix enhanced with STL decomposition features.

    This is the main entry point.  It:
    1. Runs STL decomposition on per-capita purchase, consume, transfer.
    2. Extracts S_week, S_month, f_feast, f_bigfeast, I_irregular.
    3. Merges them into the original feature matrix.

    Args:
        daily_df: DataFrame with date, purchase, redeem columns (raw amounts).
        use_finance: Passed through to make_features.

    Returns:
        Feature DataFrame with all original features plus STL features.
    """
    from features import make_features

    # Original features
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feat = make_features(df, use_finance=use_finance)

    # We need per-capita data for STL. Build approximate per-capita
    # from the daily data if available, otherwise use raw amounts.
    # Since we may not have per-capita in the daily_df, we use raw amounts
    # as a proxy – the STL shape factors are scale-invariant (multiplicative).
    daily_indexed = df.set_index("date")

    for target in ["purchase", "redeem"]:
        if target not in daily_indexed.columns:
            continue
        series = daily_indexed[target].dropna()

        try:
            stl_feat = extract_stl_features_raw(series, target)
        except Exception:
            continue

        # Merge STL features back
        for col in stl_feat.columns:
            if col == target:
                continue
            feat_col = f"{target}_{col}"
            stl_series = stl_feat[col]
            feat[feat_col] = np.nan
            for dt, val in stl_series.items():
                mask = feat["date"] == dt
                feat.loc[mask, feat_col] = val

    return feat


def extract_stl_features_raw(
    series: pd.Series,
    target_name: str,
) -> pd.DataFrame:
    """Extract STL features from a raw-amount series (no per-capita).

    Simplified version that works directly on raw amounts.
    """
    series = series.dropna()
    dates = series.index

    # --- Weekly STL ---
    stl_w = stl_decompose(series, period=7)
    sw_factors = extract_weekday_factors(stl_w)

    # --- Monthly STL ---
    series_30 = align_to_30_day_month(series)
    try:
        stl_m = stl_decompose(series_30, period=30)
        sm_factors = extract_dayofmonth_factors(stl_m)
    except Exception:
        sm_factors = pd.Series(1.0, index=range(1, 31))

    # --- Holiday effect ---
    feast_mean = compute_feast_from_residuals(
        stl_w, [QINGMING_DATES, DUANWU_DATES]
    )

    # --- Bigfeast table ---
    fb_table = compute_bigfeast_table(stl_w)

    # --- Build output ---
    out = pd.DataFrame(index=dates)
    out[target_name] = series.values

    # S_week
    out["S_week"] = [float(sw_factors.get(d.weekday(), 1.0)) for d in dates]

    # S_month
    out["S_month"] = [
        float(sm_factors.get(min(d.day, 30), 1.0)) for d in dates
    ]

    # f_feast
    out["f_feast"] = 1.0
    for i, ds in enumerate(MID_AUTUMN_DATES):
        dt = pd.Timestamp(ds)
        if dt in dates:
            if i == 2:
                out.loc[dt, "f_feast"] = feast_mean - (feast_mean - 1.0) * 0.5
            else:
                out.loc[dt, "f_feast"] = feast_mean

    # f_bigfeast
    out["f_bigfeast"] = 1.0
    for ds, factor in fb_table.items():
        dt = pd.Timestamp(ds)
        if dt in dates:
            out.loc[dt, "f_bigfeast"] = factor

    # STL residual I
    out["stl_resid_I"] = 1.0
    for dt in stl_w.index:
        if dt in out.index:
            trend_val = stl_w.loc[dt, "trend"]
            resid_val = stl_w.loc[dt, "resid"]
            if trend_val != 0:
                out.loc[dt, "stl_resid_I"] = 1.0 + resid_val / trend_val

    # STL trend
    out["stl_trend"] = np.nan
    for dt in stl_w.index:
        if dt in out.index:
            out.loc[dt, "stl_trend"] = stl_w.loc[dt, "trend"]

    # f1 per month
    df_tmp = out.copy()
    df_tmp["month"] = df_tmp.index.month
    df_tmp["year"] = df_tmp.index.year
    for (yr, mon), grp in df_tmp.groupby(["year", "month"]):
        month_mean = grp[target_name].mean()
        out.loc[grp.index, "f1_monthly"] = month_mean

    # I irregular = Y / (f1 * S_week * S_month * f_feast)
    denom = (
        out["f1_monthly"]
        * out["S_week"]
        * out["S_month"]
        * out["f_feast"]
    )
    denom_safe = denom.replace(0, np.nan)
    out["I_irregular"] = out[target_name] / denom_safe
    out["I_irregular"] = out["I_irregular"].clip(lower=0.1, upper=10.0)

    return out


def compute_stl_target_I(
    daily: pd.DataFrame,
    target: str,
) -> pd.Series:
    """Compute the irregular component I = Y / (f1 * S_week * S_month * f_feast).

    This is the target variable for ML models that predict the STL residual.

    Args:
        daily: DataFrame with date index and per-capita columns.
        target: Target name (e.g. 'purchase', 'consume', 'transfer').

    Returns:
        Series of I values indexed by date.
    """
    stl_feat = extract_stl_features(daily, target)
    return stl_feat["I_irregular"]
