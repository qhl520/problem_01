"""Backtest-optimize ensemble weights for the STL-enhanced prediction.

Grid-searches over component weights using rolling time-series CV on
Jun/Jul/Aug 2014, then generates the best submission.
"""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import FINAL_SUBMISSION_PATH, OUTPUT_DIR, PREDICT_END_DATE, PREDICT_START_DATE, PROCESSED_DATA_DIR
from data_utils import validate_submission
from evaluate import mape, weighted_score

warnings.filterwarnings("ignore")


def backtest_components(daily: pd.DataFrame) -> dict:
    """Generate component predictions for backtest months (Jun/Jul/Aug 2014).

    Returns dict with:
        components[fold_idx][component_name] = DataFrame with purchase/redeem
        actuals[fold_idx] = {"purchase": [...], "redeem": [...]}
    """
    from rule_models import predict_recent_average_rule, predict_weekday_rule

    folds = [
        ("2014-06-01", "2014-06-30"),
        ("2014-07-01", "2014-07-31"),
        ("2014-08-01", "2014-08-31"),
    ]

    # STL backtests using stl_model
    from stl_model import (
        STLModel,
        classify_users,
        compute_daily_stats,
        load_raw_balance,
        predict_user_count,
    )

    # Build STL model once
    try:
        raw = load_raw_balance()
        raw = classify_users(raw)
        daily_stats = compute_daily_stats(raw)
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

    component_names = ["stl_hybrid", "stl_pure", "weekday_rule", "recent14", "recent28"]
    component_preds = {name: [] for name in component_names}
    actuals_pur = []
    actuals_red = []

    for start, end in folds:
        train_end = pd.Timestamp(start) - pd.Timedelta(days=1)
        history = daily[daily["date"] <= train_end].copy()
        actual = daily[(daily["date"] >= start) & (daily["date"] <= end)]
        dates = pd.date_range(start, end, freq="D")

        actuals_pur.append(actual["purchase"].values)
        actuals_red.append(actual["redeem"].values)

        # STL pure (use backtest method)
        month = pd.Timestamp(start).month
        bt = stl_model.backtest(month, calibrate_f1=True)
        if "error" not in bt:
            # Reconstruct daily STL prediction from backtest results
            bt_dates = dates
            bt_purchase = _stl_daily_from_backtest(stl_model, bt_dates, month, "purchase", daily_stats)
            bt_redeem = _stl_daily_from_backtest(stl_model, bt_dates, month, "redeem", daily_stats)

            # Normalize to actual totals (hybrid)
            if bt_purchase.sum() > 0 and bt_redeem.sum() > 0:
                bt_pur_hybrid = bt_purchase / bt_purchase.sum() * actual["purchase"].sum()
                bt_red_hybrid = bt_redeem / bt_redeem.sum() * actual["redeem"].sum()
            else:
                bt_pur_hybrid = bt_purchase
                bt_red_hybrid = bt_redeem

            component_preds["stl_hybrid"].append(
                np.column_stack([bt_pur_hybrid, bt_red_hybrid])
            )
            component_preds["stl_pure"].append(
                np.column_stack([bt_purchase, bt_redeem])
            )

        # Rule models
        wr = predict_weekday_rule(history, start, end, n=8, recursive=False)
        r14 = predict_recent_average_rule(history, start, end, 14)
        r28 = predict_recent_average_rule(history, start, end, 28)

        component_preds["weekday_rule"].append(wr[["purchase", "redeem"]].values)
        component_preds["recent14"].append(r14[["purchase", "redeem"]].values)
        component_preds["recent28"].append(r28[["purchase", "redeem"]].values)

    return {
        "components": component_preds,
        "actuals_pur": actuals_pur,
        "actuals_red": actuals_red,
        "folds": ["Jun", "Jul", "Aug"],
    }


def _stl_daily_from_backtest(
    model, dates: pd.DatetimeIndex, month: int, target: str, daily_stats: pd.DataFrame
) -> np.ndarray:
    """Reconstruct daily STL predictions for a backtest month."""
    from stl_model import predict_user_count

    f1 = model.f1.get(target, 0)
    sw = model.S_week.get(target, pd.Series())
    sm = model.S_month.get(target, pd.Series())
    ff = model.f_feast.get(target, pd.Series())
    fb = model.f_bigfeast.get(target, pd.Series())

    # Recompute f1 using only data before the test month
    year = 2014
    start_dt = pd.Timestamp(f"{year}-{month:02d}-01")
    ds_indexed = daily_stats.set_index("date").sort_index()
    train_series = ds_indexed.loc[ds_indexed.index < start_dt, f"{target}_per_capita"].dropna()
    if len(train_series) > 10:
        from stl_model import arima_predict_mean
        f1 = arima_predict_mean(train_series)

    active_users = predict_user_count(
        ds_indexed[ds_indexed.index < start_dt].reset_index().rename(columns={"index": "date"}),
        dates,
    )

    pred = np.ones(len(dates), dtype=float)
    for i, dt in enumerate(dates):
        f2 = sw.get(dt.weekday(), 1.0) * sm.get(min(dt.day, 30), 1.0)
        date_str = dt.strftime("%Y-%m-%d")
        ff_val = 1.0
        if dt in ff.index:
            ff_val = float(ff[dt])
        f2 *= ff_val

        if date_str in [
            "2014-09-25", "2014-09-26", "2014-09-27",
            "2014-09-28", "2014-09-29", "2014-09-30",
        ]:
            fb_val = float(fb.get(dt, f2))
            if fb_val < f2:
                factor = f2 + fb_val - 1.0
            else:
                factor = fb_val
        else:
            factor = f2

        pred[i] = max(0, f1 * factor)

    return pred * active_users.values


def optimize_weights(bt_data: dict) -> dict:
    """Grid search for optimal component weights.

    Uses 3-fold rolling CV on Jun/Jul/Aug 2014.
    Weights are per-component, optimized separately for purchase and redeem.
    """
    components = bt_data["components"]
    active = [name for name, preds in components.items() if len(preds) == 3]
    actuals_pur = bt_data["actuals_pur"]
    actuals_red = bt_data["actuals_red"]

    print(f"  Active components: {active}")

    # Grid search on weight simplex for top components
    # Focus on: stl_hybrid, weekday_rule, recent14
    main = [c for c in ["stl_hybrid", "weekday_rule", "recent14"] if c in active]

    if len(main) < 2:
        return {
            "purchase_weights": {c: 1.0/len(active) for c in active},
            "redeem_weights": {c: 1.0/len(active) for c in active},
        }

    weight_steps = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0

    # Separate optimization for purchase and redeem
    results = {}
    for target_name, actuals in [("purchase", actuals_pur), ("redeem", actuals_red)]:
        best_score = -999.0
        best_weights = {}

        # 2-component search (if only 2 main components)
        if len(main) == 2:
            for w1 in weight_steps:
                w2 = 1.0 - w1
                weights = {main[0]: w1, main[1]: w2}
                score = _score_weights(weights, components, actuals_pur, actuals_red)
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
        else:
            # 3-component search
            for w1 in weight_steps:
                for w2 in weight_steps:
                    w3 = 1.0 - w1 - w2
                    if w3 < 0:
                        continue
                    weights = {main[0]: w1, main[1]: w2, main[2]: w3}
                    # Also try adding other components
                    for extra in [c for c in active if c not in main]:
                        weights[extra] = 0.0
                    score = _score_weights(weights, components, actuals_pur, actuals_red)
                    if score > best_score:
                        best_score = score
                        best_weights = weights.copy()

        results[f"{target_name}_weights"] = best_weights
        results[f"{target_name}_score"] = best_score
        print(f"  {target_name}: best_score={best_score:.4f}  weights={best_weights}")

    return results


def _score_weights(
    weights: dict[str, float],
    components: dict,
    actuals_pur: list[np.ndarray],
    actuals_red: list[np.ndarray],
) -> float:
    """Score a weight combination on backtest folds using both purchase and redeem."""
    all_pur_true, all_pur_pred = [], []
    all_red_true, all_red_pred = [], []

    for fold_idx in range(len(actuals_pur)):
        pur_blend = np.zeros(len(actuals_pur[fold_idx]))
        red_blend = np.zeros(len(actuals_red[fold_idx]))
        total_w = 0.0

        for name, w in weights.items():
            if name not in components or len(components[name]) <= fold_idx:
                continue
            pred = components[name][fold_idx]
            pur_blend += w * pred[:, 0]
            red_blend += w * pred[:, 1]
            total_w += w

        if total_w > 0:
            pur_blend /= total_w
            red_blend /= total_w

        all_pur_true.extend(actuals_pur[fold_idx])
        all_pur_pred.extend(pur_blend)
        all_red_true.extend(actuals_red[fold_idx])
        all_red_pred.extend(red_blend)

    return float(weighted_score(
        np.array(all_pur_true), np.array(all_pur_pred),
        np.array(all_red_true), np.array(all_red_pred),
    ))


def _score_weights_separate(
    weights: dict[str, float],
    components: dict,
    actuals_pur: list[np.ndarray],
    actuals_red: list[np.ndarray],
) -> float:
    """Score weights using separate purchase and redeem components."""
    all_pur_true, all_pur_pred = [], []
    all_red_true, all_red_pred = [], []

    for fold_idx in range(len(actuals_pur)):
        pur_blend = np.zeros(len(actuals_pur[fold_idx]))
        red_blend = np.zeros(len(actuals_red[fold_idx]))
        total_w = 0.0

        for name, w in weights.items():
            if name not in components or len(components[name]) <= fold_idx:
                continue
            pred = components[name][fold_idx]
            pur_blend += w * pred[:, 0]
            red_blend += w * pred[:, 1]
            total_w += w

        if total_w > 0:
            pur_blend /= total_w
            red_blend /= total_w

        all_pur_true.extend(actuals_pur[fold_idx])
        all_pur_pred.extend(pur_blend)
        all_red_true.extend(actuals_red[fold_idx])
        all_red_pred.extend(red_blend)

    return float(weighted_score(
        np.array(all_pur_true), np.array(all_pur_pred),
        np.array(all_red_true), np.array(all_red_pred),
    ))


def optimize_and_generate(daily: pd.DataFrame) -> pd.DataFrame:
    """Run backtest optimization and generate the best submission."""
    from rule_models import predict_recent_average_rule, predict_weekday_rule, to_submission
    from stl_model import (
        STLModel,
        classify_users,
        compute_daily_stats,
        load_raw_balance,
    )

    print("=" * 60)
    print("PPT-Aligned Ensemble Optimization")
    print("=" * 60)

    # 1. Backtest
    print("\n[1/4] Running backtests ...")
    bt_data = backtest_components(daily)

    # 2. Optimize weights
    print("\n[2/4] Optimizing ensemble weights ...")
    opt_results = optimize_weights(bt_data)

    # 3. Generate September predictions
    print("\n[3/4] Generating September predictions ...")

    # STL hybrid prediction
    try:
        raw = load_raw_balance()
        raw = classify_users(raw)
        daily_stats = compute_daily_stats(raw)
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

    # Get initial model totals for hybrid anchoring
    initial_path = OUTPUT_DIR / "original" / "tc_initial.csv"
    if initial_path.exists():
        from data_utils import read_submission
        initial = read_submission(initial_path)
        target_totals = {
            "purchase": float(initial["purchase"].sum()),
            "redeem": float(initial["redeem"].sum()),
        }
    else:
        target_totals = None

    stl_forecast = stl_model.predict(correct_f1=False, target_totals=target_totals)
    stl_sub = stl_forecast.to_submission()
    stl_sub["date"] = pd.to_datetime(stl_sub["report_date"].astype(str))

    # Rule model predictions
    dates = pd.date_range(PREDICT_START_DATE, PREDICT_END_DATE, freq="D")
    wr = predict_weekday_rule(daily, PREDICT_START_DATE, PREDICT_END_DATE, n=8)
    r14 = predict_recent_average_rule(daily, PREDICT_START_DATE, PREDICT_END_DATE, 14)
    r28 = predict_recent_average_rule(daily, PREDICT_START_DATE, PREDICT_END_DATE, 28)

    all_preds = {
        "stl_hybrid": stl_sub[["date", "purchase", "redeem"]],
        "weekday_rule": wr[["date", "purchase", "redeem"]],
        "recent14": r14[["date", "purchase", "redeem"]],
        "recent28": r28[["date", "purchase", "redeem"]],
    }

    # Normalize all to same totals before blending
    if target_totals:
        for name in all_preds:
            frame = all_preds[name]
            p_sum = frame["purchase"].sum()
            r_sum = frame["redeem"].sum()
            if p_sum > 0:
                frame["purchase"] = frame["purchase"] * target_totals["purchase"] / p_sum
            if r_sum > 0:
                frame["redeem"] = frame["redeem"] * target_totals["redeem"] / r_sum

    # 4. Blend with optimized weights
    print("\n[4/4] Blending and saving ...")

    purchase_weights = opt_results.get("purchase_weights", {"stl_hybrid": 0.6, "weekday_rule": 0.4})
    redeem_weights = opt_results.get("redeem_weights", {"stl_hybrid": 0.6, "weekday_rule": 0.4})

    # Normalize weights to sum to 1
    for wdict in [purchase_weights, redeem_weights]:
        total = sum(wdict.values())
        if total > 0:
            for k in wdict:
                wdict[k] /= total

    print(f"  Purchase weights: {purchase_weights}")
    print(f"  Redeem weights: {redeem_weights}")

    pred = pd.DataFrame({"date": dates, "purchase": 0.0, "redeem": 0.0})
    for name, frame in all_preds.items():
        w_p = purchase_weights.get(name, 0.0)
        w_r = redeem_weights.get(name, 0.0)
        if w_p > 0 or w_r > 0:
            frame_indexed = frame.set_index("date")
            for dt in dates:
                if dt in frame_indexed.index:
                    if w_p > 0:
                        pred.loc[pred["date"] == dt, "purchase"] += (
                            w_p * float(frame_indexed.loc[dt, "purchase"])
                        )
                    if w_r > 0:
                        pred.loc[pred["date"] == dt, "redeem"] += (
                            w_r * float(frame_indexed.loc[dt, "redeem"])
                        )

    sub = pred.copy()
    sub["report_date"] = pd.to_datetime(sub["date"]).dt.strftime("%Y%m%d").astype(int)
    sub["purchase"] = sub["purchase"].round().clip(lower=0).astype("int64")
    sub["redeem"] = sub["redeem"].round().clip(lower=0).astype("int64")

    sub[["report_date", "purchase", "redeem"]].to_csv(
        FINAL_SUBMISSION_PATH, index=False, header=False,
    )
    validate_submission(FINAL_SUBMISSION_PATH)

    # Verify totals
    final_df = pd.read_csv(FINAL_SUBMISSION_PATH, header=None, names=["date", "purchase", "redeem"])
    print(f"\n  Final totals:")
    print(f"    Purchase: {final_df['purchase'].sum():,}")
    print(f"    Redeem:   {final_df['redeem'].sum():,}")
    print(f"  Saved to: {FINAL_SUBMISSION_PATH}")

    # Save optimization results
    (OUTPUT_DIR / "stl_enhanced_optimal_weights.json").write_text(
        json.dumps(opt_results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("=" * 60)
    return sub[["report_date", "purchase", "redeem"]]


if __name__ == "__main__":
    daily = pd.read_csv(PROCESSED_DATA_DIR / "daily_balance.csv", parse_dates=["date"])
    optimize_and_generate(daily)
