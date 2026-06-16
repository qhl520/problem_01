"""STL-decomposition based prediction model for Yu'E Bao fund flows.

Implements the multiplicative time-series decomposition approach described in
the 2015 Tianchi competition winning solution:

    Y = f1 * S_week * S_month * f_feast * I

where:
    f1      = ARIMA-predicted September daily mean (trend level)
    S_week  = weekly seasonal factor (7-day cycle)
    S_month = monthly seasonal factor (aligned to 30-day months)
    f_feast = holiday adjustment from similar holidays' STL residuals
    I       = irregular component (unpredictable, set to 1)

The model works on *per-capita* amounts (daily total ÷ active user count),
which is more stable because the active-user-count trend is near-linear and
easy to extrapolate.

Separate models are trained for purchase, consume_amt, and transfer_amt;
redeem = consume + transfer.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL

from config import OUTPUT_DIR, RAW_DATA_DIR
from data_utils import find_user_balance_file, parse_competition_date, read_csv_safely, validate_submission

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIG_USER_REDEEM_THRESHOLD = 5_000_000  # max single redeem amount (fen)

# Holiday date ranges for residual extraction and adjustment
QINGMING_DATES = ["2014-04-05", "2014-04-06", "2014-04-07"]
DUANWU_DATES = ["2014-05-31", "2014-06-01", "2014-06-02"]
MID_AUTUMN_DATES = ["2014-09-06", "2014-09-07", "2014-09-08"]
NATIONAL_DAY_PRE = ["2014-09-25", "2014-09-26", "2014-09-27", "2014-09-28", "2014-09-29", "2014-09-30"]

# Sep 9 "九九大促" (minor Alibaba shopping festival) — modest consume boost
JIUJIU_PROMO_DATE = "2014-09-09"
JIUJIU_CONSUME_BOOST = 1.05  # +5% consume on this day

PREDICT_START = "2014-09-01"
PREDICT_END = "2014-09-30"

# Months used for monthly STL decomposition (excludes Feb due to CNY noise)
MONTHLY_STL_MONTHS = [4, 5, 6, 7, 8]
STL_SEASONAL_SMOOTHER = 13  # default statsmodels smoother


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def load_raw_balance() -> pd.DataFrame:
    """Load the raw user_balance_table and parse dates."""
    path = find_user_balance_file(RAW_DATA_DIR)
    if path is None:
        raise FileNotFoundError("No user_balance_table found in data/raw/")
    df, _ = read_csv_safely(path)
    df["date"] = parse_competition_date(df["report_date"])
    # Remove rows where all key amounts are zero
    amount_cols = ["total_purchase_amt", "total_redeem_amt", "consume_amt", "transfer_amt"]
    mask = df[amount_cols].sum(axis=1) > 0
    return df.loc[mask].copy()


def classify_users(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each user as 'big' or 'small' based on their max single-day redeem.

    Threshold: 5,000,000 fen (50,000 CNY) per the winning solution.
    """
    user_max_redeem = df.groupby("user_id")["total_redeem_amt"].max()
    big_users = user_max_redeem[user_max_redeem >= BIG_USER_REDEEM_THRESHOLD].index
    df = df.copy()
    df["user_type"] = np.where(df["user_id"].isin(big_users), "big", "small")
    return df


def compute_daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to daily level with per-capita metrics.

    Returns DataFrame with columns:
        date, purchase, consume, transfer, redeem,
        purchase_per_capita, consume_per_capita, transfer_per_capita, redeem_per_capita,
        active_users, big_users, small_users
    """
    daily = (
        df.groupby("date", as_index=False)
        .agg(
            purchase=("total_purchase_amt", "sum"),
            consume=("consume_amt", "sum"),
            transfer=("transfer_amt", "sum"),
            active_users=("user_id", "nunique"),
            big_users=("user_type", lambda x: (x == "big").sum()),
            small_users=("user_type", lambda x: (x == "small").sum()),
        )
        .sort_values("date")
    )
    daily["redeem"] = daily["consume"] + daily["transfer"]

    for col in ["purchase", "consume", "transfer", "redeem"]:
        daily[f"{col}_per_capita"] = daily[col] / daily["active_users"]

    return daily


def predict_user_count(daily: pd.DataFrame, predict_dates: pd.DatetimeIndex) -> pd.Series:
    """Extrapolate active user count via linear regression.

    The PPT notes that user count follows a near-perfect linear trend, making
    it easy to predict with high precision.
    """
    from sklearn.linear_model import LinearRegression

    hist = daily[["date", "active_users"]].dropna()
    X_hist = (hist["date"] - hist["date"].min()).dt.days.values.reshape(-1, 1)
    y_hist = hist["active_users"].values

    model = LinearRegression()
    model.fit(X_hist, y_hist)

    X_pred = (predict_dates - hist["date"].min()).days.values.reshape(-1, 1)
    pred = model.predict(X_pred)
    return pd.Series(np.maximum(pred, 1), index=predict_dates, name="active_users_pred")


# ---------------------------------------------------------------------------
# Day alignment (31-day months → 30-day months)
# ---------------------------------------------------------------------------


def align_to_30_day_month(series: pd.Series) -> pd.Series:
    """Align months to 30 days by averaging days 30 and 31.

    For months with 31 days, day 30 = mean(original day 30, day 31).
    February is excluded from the training set due to CNY noise.
    """
    df = series.reset_index()
    df.columns = ["date", "value"]
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # Build a (year, month, day) pivot
    pivot = df.pivot_table(
        index=["year", "month"], columns="day", values="value", aggfunc="mean"
    )

    # Average days 30 and 31 → new day 30
    if 31 in pivot.columns:
        has_30 = 30 in pivot.columns
        if has_30:
            pivot[30] = pivot[[30, 31]].mean(axis=1)
        else:
            pivot[30] = pivot[31]
        pivot = pivot.drop(columns=[31])

    # Melt back to long form
    result = pivot.stack().reset_index()
    result.columns = ["year", "month", "day", "value"]
    result["date"] = pd.to_datetime(
        result["year"].astype(str) + "-" + result["month"].astype(str).str.zfill(2) + "-01"
    ) + pd.to_timedelta(result["day"].astype(int) - 1, unit="D")

    return result.set_index("date")["value"].sort_index()


# ---------------------------------------------------------------------------
# STL decomposition
# ---------------------------------------------------------------------------


def stl_decompose_weekly(series: pd.Series) -> pd.DataFrame:
    """STL decomposition with 7-day period → weekly seasonal S_week.

    Returns DataFrame with columns: trend, seasonal, resid, S_week_factor
    """
    series = series.dropna()
    if len(series) < 14:
        raise ValueError(f"Need at least 14 observations for weekly STL, got {len(series)}")

    stl = STL(series, period=7, seasonal=STL_SEASONAL_SMOOTHER, robust=True)
    result = stl.fit()

    out = pd.DataFrame(
        {"trend": result.trend, "seasonal": result.seasonal, "resid": result.resid},
        index=series.index,
    )
    # S_week as multiplicative factor: seasonal / trend (additive) →
    # multiplicative factor = 1 + seasonal / trend
    trend_safe = out["trend"].replace(0, np.nan)
    out["S_week_factor"] = 1.0 + out["seasonal"] / trend_safe
    out["S_week_factor"] = out["S_week_factor"].clip(lower=0.3, upper=3.0)
    return out


def extract_weekday_factors(stl_result: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Extract weekday-specific multiplicative factors from STL weekly decomposition.

    Returns a Series indexed by weekday (0=Mon ... 6=Sun).
    """
    df = stl_result.copy()
    df["weekday"] = df.index.weekday
    return df.groupby("weekday")["S_week_factor"].median()


def stl_decompose_monthly(series_30day: pd.Series, year: int = 2014) -> pd.DataFrame:
    """STL decomposition with 30-day period on day-aligned data.

    Uses only Apr-Aug months (MONTHLY_STL_MONTHS); February excluded for CNY.

    Returns DataFrame with S_month_factor column.
    """
    # Keep only specified months
    idx = series_30day.index
    mask = idx.month.isin(MONTHLY_STL_MONTHS) & (idx.year == year)
    subset = series_30day.loc[mask].dropna()

    if len(subset) < 60:
        raise ValueError(f"Need at least 60 observations for monthly STL, got {len(subset)}")

    stl = STL(subset, period=30, seasonal=STL_SEASONAL_SMOOTHER, robust=True)
    result = stl.fit()

    out = pd.DataFrame(
        {"trend": result.trend, "seasonal": result.seasonal, "resid": result.resid},
        index=subset.index,
    )
    trend_safe = out["trend"].replace(0, np.nan)
    out["S_month_factor"] = 1.0 + out["seasonal"] / trend_safe
    out["S_month_factor"] = out["S_month_factor"].clip(lower=0.3, upper=3.0)
    return out


def extract_dayofmonth_factors(stl_result: pd.DataFrame) -> pd.Series:
    """Extract day-of-month multiplicative factors from monthly STL.

    Returns Series indexed by day-of-month (1..30).
    """
    df = stl_result.copy()
    df["day"] = df.index.day
    return df.groupby("day")["S_month_factor"].median()


# ---------------------------------------------------------------------------
# Holiday effects (f_feast)
# ---------------------------------------------------------------------------


def extract_holiday_residuals(
    stl_weekly: pd.DataFrame, holiday_dates: list[str]
) -> pd.Series:
    """Extract mean multiplicative residual for given holiday dates.

    The residual from the additive STL represents the unexplained component.
    Converting to multiplicative: holiday_factor = 1 + mean_resid / mean_trend.

    f_feast = mean of residuals during Qingming/Duanwu, used to estimate
    Mid-Autumn effect.
    """
    mask = stl_weekly.index.isin(pd.to_datetime(holiday_dates))
    if not mask.any():
        return pd.Series([1.0], dtype=float)

    holiday_resid = stl_weekly.loc[mask, "resid"]
    holiday_trend = stl_weekly.loc[mask, "trend"]
    trend_safe = holiday_trend.replace(0, np.nan)

    # Multiplicative holiday effect
    effects = 1.0 + holiday_resid / trend_safe
    return effects.dropna()


def compute_feast_factor(
    stl_weekly: pd.DataFrame,
    reference_holidays: list[list[str]],
    target_dates: list[str],
) -> pd.Series:
    """Compute f_feast for target dates from reference holiday residuals.

    reference_holidays: list of holiday date lists (e.g., [QINGMING_DATES, DUANWU_DATES])
    target_dates: dates to apply the effect to (e.g., MID_AUTUMN_DATES)

    Returns Series indexed by target date strings with multiplicative factors.
    """
    all_effects = []
    for holiday_dates in reference_holidays:
        effects = extract_holiday_residuals(stl_weekly, holiday_dates)
        if len(effects) > 0:
            all_effects.extend(effects.tolist())

    if not all_effects:
        return pd.Series(1.0, index=pd.to_datetime(target_dates))

    # The holiday effect for Mid-Autumn is the mean deviation from 1.0
    mean_effect = np.mean(all_effects)
    # For each target day, apply the deviation proportionally
    factors = {}
    for i, dt in enumerate(pd.to_datetime(target_dates)):
        # The first two days of a 3-day holiday have similar patterns,
        # the third day (peak) has stronger effect
        if i == 2:  # last holiday day → stronger
            factors[dt] = mean_effect - (mean_effect - 1.0) * 0.5
        else:
            factors[dt] = mean_effect
    return pd.Series(factors)


# ---------------------------------------------------------------------------
# National Day pre-effect (f_bigfeast)
# ---------------------------------------------------------------------------


def compute_bigfeast_factor(
    stl_weekly: pd.DataFrame,
) -> pd.Series:
    """Compute f_bigfeast as a daily coefficient table for Sep 25-30.

    Per the PPT, f_bigfeast is derived from the STL residual (remainder) pattern
    during month-end days across historical months.  The last 6 days of each
    available month form a "month-end shape" that captures the pre-holiday
    behaviour seen before long breaks (e.g. National Day).

    For each day-offset (25→30 relative to month-end), we compute the median
    multiplicative residual across all training months and apply it to the
    corresponding September date.

    The resulting Series is indexed by datetime and contains multiplicative
    factors — NOT additive adjustments — so they can be plugged directly into
    the f3 formula from the PPT:

        f3 = f2 + f_bigfeast - 1   (if f_bigfeast < f2)
        f3 = f_bigfeast             (if f_bigfeast >= f2)
    """
    df = stl_weekly.copy()
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day

    # Multiplicative residual: 1 + resid / trend
    trend_safe = df["trend"].replace(0, np.nan)
    df["mult_resid"] = 1.0 + df["resid"] / trend_safe

    # For each calendar month, find the last 6 days (days 25-30 or 26-31 etc.)
    month_end_offsets: dict[int, list[float]] = {i: [] for i in range(6)}

    for (year, month), grp in df.groupby(["year", "month"]):
        grp = grp.sort_index()
        last_6 = grp.tail(6)
        if len(last_6) < 6:
            continue
        for offset, (_, row) in enumerate(last_6.iterrows()):
            month_end_offsets[offset].append(row["mult_resid"])

    # Build the coefficient table: median multiplicative residual per offset
    dates = pd.to_datetime(NATIONAL_DAY_PRE)
    factors = np.ones(len(dates))
    for i in range(len(dates)):
        if month_end_offsets[i]:
            factors[i] = float(np.median(month_end_offsets[i]))

    return pd.Series(np.clip(factors, 0.3, 3.0), index=dates)


# ---------------------------------------------------------------------------
# ARIMA for f1 (monthly mean prediction)
# ---------------------------------------------------------------------------


def arima_predict_mean(series: pd.Series, steps: int = 30) -> float:
    """Use ARIMA to predict the daily mean for the next `steps` days.

    The winning solution used auto.arima which selected ARIMA(0,1,0) —
    effectively a random walk. The forecast is the last observed value.

    We try auto.arima first, falling back to ARIMA(0,1,0).
    """
    series = series.dropna()
    if len(series) < 10:
        return float(series.mean())

    try:
        from pmdarima import auto_arima as pm_auto_arima

        model = pm_auto_arima(
            series,
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            d=None, max_d=1,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            maxiter=10,
        )
        forecast = model.predict(n_periods=steps)
        return float(np.clip(forecast.mean(), 0, None))

    except Exception:
        # Fall back to ARIMA(0,1,0) → random walk → forecast = last value
        try:
            model = ARIMA(series, order=(0, 1, 0))
            fitted = model.fit()
            forecast = fitted.forecast(steps=steps)
            return float(np.clip(forecast.mean(), 0, None))
        except Exception:
            return float(series.tail(30).mean())


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------


@dataclass
class STLForecast:
    """Result of STL-based September 2014 forecast."""

    purchase: pd.Series   # per-capita
    consume: pd.Series    # per-capita
    transfer: pd.Series   # per-capita
    redeem: pd.Series     # per-capita (consume + transfer)
    active_users_pred: pd.Series
    dates: pd.DatetimeIndex

    def to_total_amounts(self) -> pd.DataFrame:
        """Convert per-capita predictions back to total amounts (fen)."""
        pred = pd.DataFrame({"date": self.dates})
        pred["purchase"] = (self.purchase.values * self.active_users_pred.values).round().astype("int64")
        pred["consume"] = (self.consume.values * self.active_users_pred.values).round().astype("int64")
        pred["transfer"] = (self.transfer.values * self.active_users_pred.values).round().astype("int64")
        pred["redeem"] = pred["consume"] + pred["transfer"]
        return pred

    def to_submission(self) -> pd.DataFrame:
        """Convert to competition submission format."""
        df = self.to_total_amounts()
        df["report_date"] = df["date"].dt.strftime("%Y%m%d").astype(int)
        return df[["report_date", "purchase", "redeem"]]


class STLModel:
    """Multiplicative STL decomposition model for fund flow prediction.

    Usage:
        model = STLModel()
        model.fit(daily_stats)
        forecast = model.predict()
        submission = forecast.to_submission()
    """

    def __init__(self):
        self.daily: Optional[pd.DataFrame] = None
        self.f1: dict[str, float] = {}           # ARIMA mean prediction
        self.S_week: dict[str, pd.Series] = {}    # weekday → factor
        self.S_month: dict[str, pd.Series] = {}   # day-of-month → factor
        self.f_feast: dict[str, pd.Series] = {}   # date → holiday factor
        self.f_bigfeast: dict[str, pd.Series] = {}  # date → National Day factor
        self.user_count_model: Optional[tuple] = None

    def fit(self, daily: pd.DataFrame) -> STLModel:
        """Fit the STL model on historical daily per-capita data.

        Args:
            daily: Output of compute_daily_stats() with date index.
        """
        self.daily = daily.set_index("date").sort_index()
        targets = ["purchase_per_capita", "consume_per_capita", "transfer_per_capita"]

        for target in targets:
            series = self.daily[target].dropna()
            short_name = target.replace("_per_capita", "")

            # --- f1: ARIMA mean prediction ---
            self.f1[short_name] = arima_predict_mean(series)

            # --- Weekly STL → S_week ---
            stl_w = stl_decompose_weekly(series)
            self.S_week[short_name] = extract_weekday_factors(
                stl_w, stl_w.index
            )

            # --- Monthly STL → S_month ---
            series_30 = align_to_30_day_month(series)
            stl_m = stl_decompose_monthly(series_30)
            self.S_month[short_name] = extract_dayofmonth_factors(stl_m)

            # --- f_feast: holiday effect from Qingming/Duanwu ---
            self.f_feast[short_name] = compute_feast_factor(
                stl_w,
                [QINGMING_DATES, DUANWU_DATES],
                MID_AUTUMN_DATES,
            )

            # --- f_bigfeast: National Day pre-effect ---
            self.f_bigfeast[short_name] = compute_bigfeast_factor(stl_w)

        return self

    def backtest(
        self,
        test_month: int,
        year: int = 2014,
        calibrate_f1: bool = False,
    ) -> dict:
        """Backtest the full multiplicative model on a historical month.

        Args:
            test_month: Month to backtest (6=June, 7=July, 8=August).
            year: Year (default 2014).
            calibrate_f1: If True, use leave-one-out f1 (train up to test_month-1).

        Returns:
            Dict with purchase_mape, redeem_mape, actual/pred totals, etc.
        """
        if self.daily is None:
            raise RuntimeError("Call fit() before backtest().")

        start = f"{year}-{test_month:02d}-01"
        days = pd.Timestamp(start).days_in_month
        end = f"{year}-{test_month:02d}-{days}"

        dates = pd.date_range(start, end, freq="D")
        actual = self.daily.loc[start:end]
        if actual.empty:
            return {"error": f"No actual data for {test_month}"}

        active_users = predict_user_count(
            self.daily[self.daily.index < start].reset_index().rename(columns={"index": "date"}),
            dates,
        )

        results: dict[str, dict] = {}
        targets = ["purchase", "consume", "transfer"]

        for target in targets:
            col = f"{target}_per_capita"
            f1 = self.f1.get(target, 0)
            sw = self.S_week.get(target, pd.Series())
            sm = self.S_month.get(target, pd.Series())
            ff = self.f_feast.get(target, pd.Series())
            fb = self.f_bigfeast.get(target, pd.Series())

            # If calibrating, refit f1 on data up to test_month-1
            if calibrate_f1:
                train_series = self.daily.loc[self.daily.index < start, col].dropna()
                if len(train_series) > 10:
                    f1 = arima_predict_mean(train_series)

            pred = np.ones(len(dates), dtype=float)
            for i, dt in enumerate(dates):
                value = f1
                w = dt.weekday()
                if w in sw.index:
                    value *= sw[w]
                d = dt.day
                d_aligned = min(d, 30)
                if d_aligned in sm.index:
                    value *= sm[d_aligned]
                pred[i] = max(0, value)

            pred_total = pred * active_users.values
            actual_vals = actual[col].values * actual["active_users"].values

            results[target] = {
                "pred_total": float(pred_total.sum()),
                "actual_total": float(actual_vals.sum()),
                "mape": float(np.mean(np.abs(actual_vals - pred_total) / np.where(actual_vals == 0, 1, np.abs(actual_vals)))),
                "pred_daily_mean": float(pred_total.mean()),
                "actual_daily_mean": float(actual_vals.mean()),
            }

        # redeem = consume + transfer
        redeem_pred = results["consume"]["pred_total"] + results["transfer"]["pred_total"]
        redeem_actual = float(actual["redeem"].sum())
        results["redeem"] = {
            "pred_total": redeem_pred,
            "actual_total": redeem_actual,
            "mape": float(np.mean(np.abs(
                actual["redeem"].values - (results["consume"]["pred_total"] + results["transfer"]["pred_total"]) / len(dates)
            ) / np.where(actual["redeem"].values == 0, 1, np.abs(actual["redeem"].values)))),
        }

        return {
            "month": test_month,
            "purchase": results["purchase"],
            "redeem": results["redeem"],
            "consume": results["consume"],
            "transfer": results["transfer"],
        }

    def calibrate_f1(
        self,
    ) -> dict[str, float]:
        """Cross-validate f1 using leave-one-out backtests on Jun/Jul/Aug.

        Returns correction factors per target: corrected_f1 = original_f1 * factor.
        """
        months = [6, 7, 8]
        corrections: dict[str, list[float]] = {"purchase": [], "consume": [], "transfer": []}

        print("\n  --- f1 Cross-Validation ---")
        for month in months:
            bt = self.backtest(month, calibrate_f1=True)
            if "error" in bt:
                continue
            for target in ["purchase", "consume", "transfer"]:
                r = bt[target]
                if r["actual_total"] > 0 and r["pred_total"] > 0:
                    ratio = r["actual_total"] / r["pred_total"]
                    corrections[target].append(ratio)
            print(
                f"  {bt['month']:>5}: purchase MAPE={bt['purchase']['mape']:.3f}  "
                f"redeem MAPE={bt['redeem']['mape']:.3f}  "
                f"purchase ratio={bt['purchase']['actual_total']/bt['purchase']['pred_total']:.3f}"
            )

        # Median correction (more robust than mean)
        factors = {}
        for target in ["purchase", "consume", "transfer"]:
            if corrections[target]:
                factors[target] = float(np.median(corrections[target]))
            else:
                factors[target] = 1.0

        print(f"  CV correction factors: { {k: f'{v:.4f}' for k, v in factors.items()} }")
        return factors

    def predict(
        self,
        start: str = PREDICT_START,
        end: str = PREDICT_END,
        correct_f1: bool = True,
        target_totals: Optional[dict[str, float]] = None,
    ) -> STLForecast:
        """Generate per-capita predictions for September 2014.

        Implements the PPT's multiplicative model:

            Sep  1-24:  Ŷ = f1 × f2       where f2 = S_week × S_month × f_feast
            Sep 25-30:  Ŷ = f1 × f3       where f3 combines f2 and f_bigfeast

        f3 formula (PPT exact):
            f3 = f2 + f_bigfeast - 1   if f_bigfeast < f2   (blend)
            f3 = f_bigfeast             if f_bigfeast >= f2  (replace)

        Args:
            start, end: Prediction date range.
            correct_f1: If True, apply cross-validated f1 correction factors.
            target_totals: Optional dict. If provided, re-normalise shape to
                           match these totals (hybrid mode).
        """
        if self.daily is None:
            raise RuntimeError("Call fit() before predict().")

        if correct_f1:
            f1_correction = self.calibrate_f1()
        else:
            f1_correction = {"purchase": 1.0, "consume": 1.0, "transfer": 1.0}

        dates = pd.date_range(start, end, freq="D")
        active_users = predict_user_count(
            self.daily.reset_index().rename(columns={"index": "date"}),
            dates,
        )

        results = {}
        targets = ["purchase", "consume", "transfer"]

        for target in targets:
            f1 = self.f1[target] * f1_correction.get(target, 1.0)
            sw = self.S_week[target]
            sm = self.S_month[target]
            ff = self.f_feast[target]
            fb = self.f_bigfeast[target]

            pred = np.ones(len(dates), dtype=float)

            for i, dt in enumerate(dates):
                date_str = dt.strftime("%Y-%m-%d")

                # --- Compute f2 = S_week × S_month × f_feast ---
                w = dt.weekday()
                sw_factor = float(sw.get(w, 1.0))

                d = dt.day
                d_aligned = min(d, 30)
                sm_factor = float(sm.get(d_aligned, 1.0))

                ff_factor = 1.0
                if date_str in MID_AUTUMN_DATES and dt in ff.index:
                    ff_factor = float(ff[dt])

                f2 = sw_factor * sm_factor * ff_factor

                # --- 九九大促: mild consume boost on Sep 9 ---
                if target == "consume" and date_str == JIUJIU_PROMO_DATE:
                    f2 *= JIUJIU_CONSUME_BOOST

                # --- Compute final factor ---
                if date_str in NATIONAL_DAY_PRE:
                    fb_factor = float(fb.get(dt, f2))
                    # PPT f3 formula
                    if fb_factor < f2:
                        factor = f2 + fb_factor - 1.0  # blend
                    else:
                        factor = fb_factor              # bigfeast dominates
                else:
                    factor = f2

                pred[i] = max(0, f1 * factor)

            results[target] = pd.Series(pred * active_users.values, index=dates, name=target)

        # redeem = consume + transfer
        results["redeem"] = pd.Series(
            results["consume"].values + results["transfer"].values,
            index=dates,
            name="redeem",
        )

        # Hybrid mode: normalise shapes to match target totals
        if target_totals is not None:
            for target in ["purchase", "redeem"]:
                if target in target_totals and results[target].sum() > 0:
                    results[target] = pd.Series(
                        results[target].values / results[target].sum() * target_totals[target],
                        index=dates,
                        name=target,
                    )

            # Re-derive consume/transfer ratio from scaled redeem
            if "redeem" in target_totals:
                total_ct = results["consume"].sum() + results["transfer"].sum()
                if total_ct > 0:
                    ratio = results["redeem"].sum() / total_ct
                    results["consume"] = pd.Series(
                        results["consume"].values * ratio, index=dates, name="consume"
                    )
                    results["transfer"] = pd.Series(
                        results["transfer"].values * ratio, index=dates, name="transfer"
                    )

        # Convert to per-capita for STLForecast
        purchase_pc = pd.Series(results["purchase"].values / active_users.values, index=dates)
        consume_pc = pd.Series(results["consume"].values / active_users.values, index=dates)
        transfer_pc = pd.Series(results["transfer"].values / active_users.values, index=dates)
        redeem_pc = pd.Series(results["redeem"].values / active_users.values, index=dates)

        return STLForecast(
            purchase=purchase_pc,
            consume=consume_pc,
            transfer=transfer_pc,
            redeem=redeem_pc,
            active_users_pred=active_users,
            dates=dates,
        )


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_stl_pipeline(
    hybrid: bool = True,
    ensemble: bool = False,
    ensemble_weight: float = 0.5,
) -> STLForecast:
    """Run the complete STL pipeline.

    Args:
        hybrid: If True, use STL shape + initial model totals.
        ensemble: If True, blend STL prediction with the 121 submission.
        ensemble_weight: Weight for STL in the blend (0.0 = all 121, 1.0 = all STL).
    """
    mode_str = "hybrid" if hybrid else "pure"
    if ensemble:
        mode_str += f"+ensemble(w={ensemble_weight})"

    print("=" * 60)
    print(f"STL Decomposition Pipeline ({mode_str})")
    print("=" * 60)

    # 1. Load and prepare data
    print("\n[1/6] Loading raw data and classifying users ...")
    raw = load_raw_balance()
    raw = classify_users(raw)
    daily = compute_daily_stats(raw)
    print(f"  Daily records: {len(daily)}")
    print(f"  Active users: {daily['active_users'].min():,} → {daily['active_users'].max():,}")

    # 2. Fit STL model
    print("\n[2/6] Fitting STL decomposition ...")
    model = STLModel()
    model.fit(daily)

    for target in ["purchase", "consume", "transfer"]:
        print(f"  {target}:")
        print(f"    f1 = {model.f1[target]:,.0f}  "
              f"S_week Mon={model.S_week[target].get(0, 1):.3f}  "
              f"Sun={model.S_week[target].get(6, 1):.3f}  "
              f"S_month d1={model.S_month[target].get(1, 1):.3f}  "
              f"d30={model.S_month[target].get(30, 1):.3f}")
        fb = model.f_bigfeast[target]
        if len(fb) > 0:
            print(f"    f_bigfeast (Sep25-30) = {[f'{v:.3f}' for v in fb.values]}")
        ff = model.f_feast[target]
        if len(ff) > 0:
            print(f"    f_feast (Mid-Autumn)  = {[f'{v:.3f}' for v in ff.values]}")

    # 3. Backtest with full multiplicative model
    print("\n[3/6] Backtesting full model on Jun/Jul/Aug 2014 ...")
    backtest_scores = []
    for month in [6, 7, 8]:
        bt = model.backtest(month, calibrate_f1=True)
        if "error" in bt:
            print(f"  {month}: {bt['error']}")
            continue
        backtest_scores.append(bt["purchase"]["mape"])
        print(
            f"  {bt['month']:>5}: "
            f"p MAPE={bt['purchase']['mape']:.3f}  "
            f"r MAPE={bt['redeem']['mape']:.3f}  "
            f"p ratio={bt['purchase']['actual_total']/bt['purchase']['pred_total']:.3f}  "
            f"r ratio={bt['redeem']['actual_total']/bt['redeem']['pred_total']:.3f}"
        )
    if backtest_scores:
        print(f"  Mean purchase MAPE: {np.mean(backtest_scores):.3f}")

    # 4. Predict September
    print("\n[4/6] Predicting September 2014 ...")

    if hybrid:
        from data_utils import read_submission

        initial_path = OUTPUT_DIR / "original" / "tc_initial.csv"
        if initial_path.exists():
            initial = read_submission(initial_path)
            target_totals = {
                "purchase": float(initial["purchase"].sum()),
                "redeem": float(initial["redeem"].sum()),
            }
            print(f"  Hybrid: STL shape × initial totals ({target_totals['purchase']:,.0f} / {target_totals['redeem']:,.0f})")
        else:
            print(f"  Initial model not found, using CV-corrected pure STL")
            target_totals = None
    else:
        target_totals = None

    forecast = model.predict(
        correct_f1=(not hybrid),
        target_totals=target_totals,
    )

    # If ensemble, blend with 121 submission
    if ensemble:
        path_121 = OUTPUT_DIR / "tc_comp_predict_table.csv"
        if path_121.exists():
            df_121 = pd.read_csv(path_121, header=None, names=["report_date", "purchase", "redeem"])
            sub_stl = forecast.to_submission()
            blended = sub_stl.copy()
            blended["purchase"] = (
                (1 - ensemble_weight) * df_121["purchase"].values
                + ensemble_weight * sub_stl["purchase"].values
            ).round().astype("int64")
            blended["redeem"] = (
                (1 - ensemble_weight) * df_121["redeem"].values
                + ensemble_weight * sub_stl["redeem"].values
            ).round().astype("int64")
            sub = blended
            print(f"  Ensemble: {1-ensemble_weight:.0%} 121 + {ensemble_weight:.0%} STL")
        else:
            sub = forecast.to_submission()
    else:
        sub = forecast.to_submission()

    # 5. Show results
    print("\n[5/6] Forecast (first 7 days):")
    for i in range(min(7, len(sub))):
        row = sub.iloc[i]
        print(f"  {row['report_date']}  {row['purchase']:>12,}  {row['redeem']:>12,}")

    totals = forecast.to_total_amounts()
    print(f"\n  Sep totals — purchase: {sub['purchase'].sum():,}  redeem: {sub['redeem'].sum():,}")
    print(f"  consume/transfer ratio: {totals['consume'].sum()/totals['transfer'].sum():.2f}")

    # 6. Save
    print("\n[6/6] Saving ...")
    if ensemble:
        out_path = OUTPUT_DIR / f"tc_comp_predict_table_stl_ensemble_w{int(ensemble_weight*100)}.csv"
    else:
        suffix = "hybrid" if hybrid else "pure"
        out_path = OUTPUT_DIR / f"tc_comp_predict_table_stl_{suffix}.csv"

    sub.to_csv(out_path, index=False, header=False)
    validate_submission(out_path)
    print(f"  → {out_path}")

    # Default STL path
    default_path = OUTPUT_DIR / "tc_comp_predict_table_stl.csv"
    sub.to_csv(default_path, index=False, header=False)
    validate_submission(default_path)
    print(f"  → {default_path}")

    # Detail
    detail = forecast.to_total_amounts()
    detail["active_users_pred"] = forecast.active_users_pred.values
    detail.to_csv(OUTPUT_DIR / "stl_forecast_detail.csv", index=False)

    print("\n" + "=" * 60)
    print(f"STL pipeline complete ({mode_str}).")
    print("=" * 60)

    return forecast


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pure", action="store_true", help="Pure STL mode (no initial model calibration)")
    parser.add_argument("--ensemble", type=float, default=None, help="STL weight for ensemble blend with 121 (0.0-1.0)")
    args = parser.parse_args()
    run_stl_pipeline(
        hybrid=not args.pure,
        ensemble=args.ensemble is not None,
        ensemble_weight=args.ensemble if args.ensemble is not None else 0.5,
    )
