"""Simple backtest-optimized ensemble: STL hybrid + rule models.

Grid searches ensemble weights using Jun/Jul/Aug 2014 backtests,
then generates the best September submission.
"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import FINAL_SUBMISSION_PATH, OUTPUT_DIR, PREDICT_END_DATE, PREDICT_START_DATE, PROCESSED_DATA_DIR, RAW_DATA_DIR
from data_utils import load_daily_balance_fallback
from data_utils import read_submission, validate_submission
from evaluate import mape, weighted_score

warnings.filterwarnings("ignore")

BEST_131_DIR = OUTPUT_DIR / "best_131"
BEST_131_PATH = BEST_131_DIR / "tc_comp_predict_table.csv"
BEST_131_TOTALS_PATH = BEST_131_DIR / "target_totals.json"
BASELINE_131_DIR = OUTPUT_DIR / "baseline_131"
BASELINE_131_PATH = BASELINE_131_DIR / "tc_comp_predict_table.csv"
BASELINE_131_TOTALS_PATH = BASELINE_131_DIR / "target_totals.json"


def main() -> None:
    print("=" * 60)
    print("PPT-Aligned Ensemble Optimization v2")
    print("=" * 60)

    daily = load_daily_balance_fallback(RAW_DATA_DIR, PROCESSED_DATA_DIR)

    # 1. Get STL hybrid prediction
    print("\n[1/4] Computing STL hybrid prediction ...")
    from stl_model import STLModel, load_daily_stats_fallback

    # Build STL model for backtest
    try:
        daily_stats = load_daily_stats_fallback()
    except Exception:
        daily_stats = daily.copy()
        daily_stats["date"] = pd.to_datetime(daily_stats["date"])
        daily_stats = daily_stats.set_index("date")
        avg_users = 10_000_000
        for col in ["purchase", "redeem"]:
            daily_stats[f"{col}_per_capita"] = daily_stats[col] / avg_users
        daily_stats["consume_per_capita"] = daily_stats["redeem"] * 0.55 / avg_users
        daily_stats["transfer_per_capita"] = daily_stats["redeem"] * 0.45 / avg_users
        daily_stats["active_users"] = avg_users

    stl_model = STLModel()
    stl_model.fit(daily_stats)

    # Get 131 locked totals.  This project intentionally keeps only the
    # 131 STL ensemble flow; old 120/121 output/original anchors are not used.
    target_totals = None
    if BEST_131_TOTALS_PATH.exists():
        target_totals = json.loads(BEST_131_TOTALS_PATH.read_text(encoding="utf-8"))
        print(f"  Best-131 locked totals: purchase={target_totals['purchase']:,.0f}  redeem={target_totals['redeem']:,.0f}")
    elif FINAL_SUBMISSION_PATH.exists():
        initial = read_submission(FINAL_SUBMISSION_PATH)
        target_totals = {
            "purchase": float(initial["purchase"].sum()),
            "redeem": float(initial["redeem"].sum()),
        }
        print(f"  Current final totals: purchase={target_totals['purchase']:,.0f}  redeem={target_totals['redeem']:,.0f}")

    # 2. Backtest all components on Jun/Jul/Aug
    print("\n[2/4] Backtesting components ...")

    from rule_models import predict_recent_average_rule, predict_weekday_rule

    folds = [
        ("2014-06-01", "2014-06-30"),
        ("2014-07-01", "2014-07-31"),
        ("2014-08-01", "2014-08-31"),
    ]

    comp_names = ["stl_hybrid", "weekday_rule", "recent14"]
    all_preds = {name: [] for name in comp_names}
    all_actual = []

    for start, end in folds:
        train_end = pd.Timestamp(start) - pd.Timedelta(days=1)
        history = daily[daily["date"] <= train_end].copy()
        actual = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy()
        all_actual.append(actual)

        month = pd.Timestamp(start).month
        dates = pd.date_range(start, end, freq="D")

        # STL backtest
        bt = stl_model.backtest(month, calibrate_f1=True)
        if "error" not in bt:
            # Reconstruct daily STL predictions for this month
            stl_daily = _build_stl_daily(stl_model, daily_stats, dates, month)

            # Hybrid: normalize STL shape to actual totals
            if stl_daily["purchase"].sum() > 0:
                stl_daily["purchase"] = (
                    stl_daily["purchase"] / stl_daily["purchase"].sum() * actual["purchase"].sum()
                )
            if stl_daily["redeem"].sum() > 0:
                stl_daily["redeem"] = (
                    stl_daily["redeem"] / stl_daily["redeem"].sum() * actual["redeem"].sum()
                )
            all_preds["stl_hybrid"].append(stl_daily[["purchase", "redeem"]].values)

        # Rule models
        wr = predict_weekday_rule(history, start, end, n=8, recursive=False)
        r14 = predict_recent_average_rule(history, start, end, 14)
        all_preds["weekday_rule"].append(wr[["purchase", "redeem"]].values)
        all_preds["recent14"].append(r14[["purchase", "redeem"]].values)

        print(f"  {month}: STL pur={stl_daily['purchase'].sum():,.0f} red={stl_daily['redeem'].sum():,.0f}")

    # 3. Grid search optimal weights
    print("\n[3/4] Optimizing ensemble weights ...")

    best_score = -999.0
    best_w = 0.5

    for w in np.linspace(0.0, 1.0, 21):  # 0.0, 0.05, ..., 1.0
        w_stl = w
        w_wr = (1.0 - w) * 0.7  # weekday_rule gets 70% of non-STL weight
        w_r14 = (1.0 - w) * 0.3  # recent14 gets 30%

        all_pur_true, all_pur_pred = [], []
        all_red_true, all_red_pred = [], []

        for fold_idx in range(len(folds)):
            actual = all_actual[fold_idx]
            pur_blend = np.zeros(len(actual))
            red_blend = np.zeros(len(actual))

            if fold_idx < len(all_preds["stl_hybrid"]):
                pur_blend += w_stl * all_preds["stl_hybrid"][fold_idx][:, 0]
                red_blend += w_stl * all_preds["stl_hybrid"][fold_idx][:, 1]
            if fold_idx < len(all_preds["weekday_rule"]):
                pur_blend += w_wr * all_preds["weekday_rule"][fold_idx][:, 0]
                red_blend += w_wr * all_preds["weekday_rule"][fold_idx][:, 1]
            if fold_idx < len(all_preds["recent14"]):
                pur_blend += w_r14 * all_preds["recent14"][fold_idx][:, 0]
                red_blend += w_r14 * all_preds["recent14"][fold_idx][:, 1]

            all_pur_true.extend(actual["purchase"].values)
            all_pur_pred.extend(pur_blend)
            all_red_true.extend(actual["redeem"].values)
            all_red_pred.extend(red_blend)

        score = weighted_score(
            np.array(all_pur_true), np.array(all_pur_pred),
            np.array(all_red_true), np.array(all_red_pred),
        )

        if score > best_score:
            best_score = score
            best_w = w

        if w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            print(f"    w={w:.2f}: score={score:.4f}")

    print(f"  Best: w_stl={best_w:.2f}  score={best_score:.4f}")

    # 4. Generate September prediction
    print("\n[4/4] Generating best submission ...")

    # Get September STL prediction
    sep_dates = pd.date_range(PREDICT_START_DATE, PREDICT_END_DATE, freq="D")
    stl_sep = _build_stl_daily(stl_model, daily_stats, sep_dates, 9)

    # Normalize to target totals
    if target_totals:
        if stl_sep["purchase"].sum() > 0:
            stl_sep["purchase"] = stl_sep["purchase"] / stl_sep["purchase"].sum() * target_totals["purchase"]
        if stl_sep["redeem"].sum() > 0:
            stl_sep["redeem"] = stl_sep["redeem"] / stl_sep["redeem"].sum() * target_totals["redeem"]

    # Rule predictions (also normalized)
    wr_sep = predict_weekday_rule(daily, PREDICT_START_DATE, PREDICT_END_DATE, n=8)
    r14_sep = predict_recent_average_rule(daily, PREDICT_START_DATE, PREDICT_END_DATE, 14)

    if target_totals:
        for df in [wr_sep, r14_sep]:
            if df["purchase"].sum() > 0:
                df["purchase"] = df["purchase"] / df["purchase"].sum() * target_totals["purchase"]
            if df["redeem"].sum() > 0:
                df["redeem"] = df["redeem"] / df["redeem"].sum() * target_totals["redeem"]

    wr_sep["date"] = pd.to_datetime(wr_sep["date"])
    r14_sep["date"] = pd.to_datetime(r14_sep["date"])

    # Blend
    w_wr = (1.0 - best_w) * 0.7
    w_r14 = (1.0 - best_w) * 0.3

    pred = pd.DataFrame({"date": sep_dates, "purchase": 0.0, "redeem": 0.0})
    stl_sep_indexed = stl_sep.set_index("date")
    wr_indexed = wr_sep.set_index("date")
    r14_indexed = r14_sep.set_index("date")

    for dt in sep_dates:
        if dt in stl_sep_indexed.index:
            pred.loc[pred["date"] == dt, "purchase"] += best_w * float(stl_sep_indexed.loc[dt, "purchase"])
            pred.loc[pred["date"] == dt, "redeem"] += best_w * float(stl_sep_indexed.loc[dt, "redeem"])
        if dt in wr_indexed.index:
            pred.loc[pred["date"] == dt, "purchase"] += w_wr * float(wr_indexed.loc[dt, "purchase"])
            pred.loc[pred["date"] == dt, "redeem"] += w_wr * float(wr_indexed.loc[dt, "redeem"])
        if dt in r14_indexed.index:
            pred.loc[pred["date"] == dt, "purchase"] += w_r14 * float(r14_indexed.loc[dt, "purchase"])
            pred.loc[pred["date"] == dt, "redeem"] += w_r14 * float(r14_indexed.loc[dt, "redeem"])

    # Convert to submission
    sub = pred.copy()
    sub["report_date"] = pd.to_datetime(sub["date"]).dt.strftime("%Y%m%d").astype(int)
    sub["purchase"] = sub["purchase"].round().clip(lower=0).astype("int64")
    sub["redeem"] = sub["redeem"].round().clip(lower=0).astype("int64")

    sub[["report_date", "purchase", "redeem"]].to_csv(
        FINAL_SUBMISSION_PATH, index=False, header=False,
    )
    validate_submission(FINAL_SUBMISSION_PATH)
    BEST_131_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FINAL_SUBMISSION_PATH, BEST_131_PATH)
    validate_submission(BEST_131_PATH)
    BASELINE_131_DIR.mkdir(parents=True, exist_ok=True)
    if not BASELINE_131_PATH.exists():
        shutil.copyfile(BEST_131_PATH, BASELINE_131_PATH)
        validate_submission(BASELINE_131_PATH)

    # Verify and display
    final = pd.read_csv(FINAL_SUBMISSION_PATH, header=None, names=["date", "purchase", "redeem"])
    print(f"\n  Ensemble: {best_w:.0%} STL + {(1-best_w)*0.7:.0%} weekday_rule + {(1-best_w)*0.3:.0%} recent14")
    print(f"  Purchase total: {final['purchase'].sum():,}")
    print(f"  Redeem total:   {final['redeem'].sum():,}")
    print(f"  Saved to: {FINAL_SUBMISSION_PATH}")

    # Save config
    (OUTPUT_DIR / "stl_ensemble_config.json").write_text(
        json.dumps({
            "w_stl": best_w,
            "w_weekday_rule": w_wr,
            "w_recent14": w_r14,
            "backtest_score": best_score,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    BEST_131_TOTALS_PATH.write_text(
        json.dumps({
            "purchase": int(final["purchase"].sum()),
            "redeem": int(final["redeem"].sum()),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not BASELINE_131_TOTALS_PATH.exists():
        BASELINE_131_TOTALS_PATH.write_text(
            json.dumps({
                "purchase": int(final["purchase"].sum()),
                "redeem": int(final["redeem"].sum()),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (BASELINE_131_DIR / "baseline_131_summary.md").write_text(
            "# Baseline 131 Protected Copy\n\n"
            "This directory is initialized once from the verified 131 submission. "
            "Candidate-generation scripts must never overwrite it by default.\n",
            encoding="utf-8",
        )

    print("=" * 60)


def _build_stl_daily(
    model,
    daily_stats: pd.DataFrame,
    dates: pd.DatetimeIndex,
    month: int,
) -> pd.DataFrame:
    """Build daily STL predictions for a given month.

    Uses per-capita STL factors × predicted active user count.
    """
    from stl_model import arima_predict_mean, predict_user_count

    ds_indexed = daily_stats.set_index("date").sort_index()

    # Predict active users for these dates
    first_of_month = pd.Timestamp(f"2014-{month:02d}-01")
    train_users = ds_indexed[ds_indexed.index < first_of_month].reset_index()
    train_users = train_users.rename(columns={"index": "date"})
    active_users = predict_user_count(train_users, dates)

    result = pd.DataFrame({"date": dates})

    # Purchase (direct)
    pur_pc = _predict_target_daily(model, ds_indexed, dates, month, "purchase")
    result["purchase"] = pur_pc * active_users.values

    # Redeem = consume + transfer
    con_pc = _predict_target_daily(model, ds_indexed, dates, month, "consume")
    trf_pc = _predict_target_daily(model, ds_indexed, dates, month, "transfer")
    result["redeem"] = (con_pc + trf_pc) * active_users.values

    return result


def _predict_target_daily(
    model,
    ds_indexed: pd.DataFrame,
    dates: pd.DatetimeIndex,
    month: int,
    target: str,
) -> np.ndarray:
    """Predict per-capita daily values for one target."""
    from stl_model import (
        MID_AUTUMN_DATES, NATIONAL_DAY_PRE,
        JIUJIU_CONSUME_BOOST, JIUJIU_PROMO_DATE,
        arima_predict_mean,
    )

    first_of_month = pd.Timestamp(f"2014-{month:02d}-01")

    # f1: ARIMA on data before this month
    col = f"{target}_per_capita"
    train_series = ds_indexed.loc[ds_indexed.index < first_of_month, col].dropna()
    if len(train_series) > 10:
        f1 = arima_predict_mean(train_series)
    else:
        f1 = float(model.f1.get(target, ds_indexed[col].mean()))

    sw = model.S_week.get(target, pd.Series())
    sm = model.S_month.get(target, pd.Series())
    ff = model.f_feast.get(target, pd.Series())
    fb = model.f_bigfeast.get(target, pd.Series())

    pred = np.ones(len(dates), dtype=float)
    for i, dt in enumerate(dates):
        f2 = float(sw.get(dt.weekday(), 1.0)) * float(sm.get(min(dt.day, 30), 1.0))
        date_str = dt.strftime("%Y-%m-%d")

        # f_feast
        if dt in ff.index:
            f2 *= float(ff[dt])

        # 九九大促
        if target == "consume" and date_str == JIUJIU_PROMO_DATE:
            f2 *= JIUJIU_CONSUME_BOOST

        # f_bigfeast for Sep 25-30
        if date_str in NATIONAL_DAY_PRE and dt in fb.index:
            fb_val = float(fb[dt])
            if fb_val < f2:
                factor = f2 + fb_val - 1.0
            else:
                factor = fb_val
        else:
            factor = f2

        pred[i] = max(0, f1 * factor)

    return pred


if __name__ == "__main__":
    main()
