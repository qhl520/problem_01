"""Train ML models enhanced with STL decomposition features.

Implements the PPT winning-solution technique:
1. STL decomposes Y = f1 * S_week * S_month * f_feast * I
2. ML models learn to predict I (the irregular component)
3. ML models also learn raw amounts directly
4. Separate consume and transfer models for redeem decomposition

Also implements the PPT's "Algorithm Step 3" — iterative holiday correction
using STL residuals.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config import MODEL_DIR, OUTPUT_DIR, PROCESSED_DATA_DIR
from features import make_features
from model_utils import get_feature_columns, save_model, save_feature_list

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _daily_to_per_capita(daily: pd.DataFrame) -> pd.DataFrame:
    """Add per-capita columns if user count data is available.

    Falls back to raw amounts if user_profile_table is not available.
    """
    from config import RAW_DATA_DIR
    from data_utils import find_user_balance_file, parse_competition_date, read_csv_safely

    path = find_user_balance_file(RAW_DATA_DIR)
    if path is None:
        return daily

    df, _ = read_csv_safely(path, usecols=["report_date", "user_id",
                                            "total_purchase_amt", "total_redeem_amt",
                                            "consume_amt", "transfer_amt"])
    df["date"] = parse_competition_date(df["report_date"])
    # Remove all-zero rows
    amount_cols = ["total_purchase_amt", "total_redeem_amt", "consume_amt", "transfer_amt"]
    mask = df[amount_cols].sum(axis=1) > 0
    df = df.loc[mask].copy()

    daily_agg = (
        df.groupby("date", as_index=False)
        .agg(
            purchase=("total_purchase_amt", "sum"),
            consume=("consume_amt", "sum"),
            transfer=("transfer_amt", "sum"),
            active_users=("user_id", "nunique"),
        )
        .sort_values("date")
    )
    daily_agg["redeem"] = daily_agg["consume"] + daily_agg["transfer"]

    for col in ["purchase", "consume", "transfer", "redeem"]:
        daily_agg[f"{col}_per_capita"] = daily_agg[col] / daily_agg["active_users"]

    return daily_agg


def _make_ml_model(model_type: str):
    """Create an ML model pipeline."""
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=400, max_depth=14, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )
    elif model_type == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=450, max_depth=14, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )
    elif model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(
                n_estimators=1500, learning_rate=0.02, num_leaves=15,
                max_depth=-1, min_child_samples=12,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.25, reg_lambda=0.75,
                random_state=42, n_jobs=-1, verbosity=-1,
            )
        except Exception:
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(
                max_iter=500, learning_rate=0.03, random_state=42,
            )
    elif model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(
                n_estimators=800, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.5,
                random_state=42, n_jobs=-1, verbosity=0,
            )
        except Exception:
            print("  XGBoost not available, falling back to ExtraTrees")
            model = ExtraTreesRegressor(
                n_estimators=450, max_depth=14, min_samples_leaf=2,
                random_state=42, n_jobs=-1,
            )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=6,
            min_samples_leaf=4, subsample=0.8,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# STL feature enrichment
# ---------------------------------------------------------------------------

def enrich_with_stl_features(
    feat: pd.DataFrame,
    daily_per_capita: pd.DataFrame,
) -> pd.DataFrame:
    """Add STL decomposition features to the ML feature matrix.

    For each row in feat, looks up STL factors by date and merges them.
    Uses stl_features module to compute S_week, S_month, I_irregular.
    """
    from stl_features import extract_stl_features_raw

    feat = feat.copy()
    feat["date_dt"] = pd.to_datetime(feat["date"])

    for col in ["purchase", "consume", "transfer", "redeem"]:

        if col not in daily_per_capita.columns:
            continue

        series = daily_per_capita.set_index("date")[col].dropna()
        if len(series) < 30:
            continue

        try:
            stl_feat = extract_stl_features_raw(series, col)
        except Exception:
            continue

        for stl_col in ["S_week", "S_month", "f_feast", "f_bigfeast",
                         "stl_resid_I", "stl_trend", "I_irregular"]:
            if stl_col not in stl_feat.columns:
                continue
            feat_col_name = f"{col}_stl_{stl_col}"
            feat[feat_col_name] = np.nan
            for dt, val in stl_feat[stl_col].items():
                mask = feat["date_dt"] == dt
                feat.loc[mask, feat_col_name] = val

    feat = feat.drop(columns=["date_dt"])
    return feat


# ---------------------------------------------------------------------------
# Rolling time-series cross-validation
# ---------------------------------------------------------------------------

def rolling_cv_score(
    model_pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    fold_months: list[int],
    target_name: str = "",
) -> dict:
    """Rolling time-series CV: train up to month M-1, validate on month M.

    Returns dict with mape, rmse, and per-fold scores.
    """
    from evaluate import mape

    scores = []
    for month in fold_months:
        train_mask = dates.dt.month < month
        val_mask = dates.dt.month == month

        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        model_pipeline.fit(X_train, y_train)
        pred = model_pipeline.predict(X_val)
        pred = np.maximum(pred, 0)

        fold_mape = mape(y_val.values, pred)
        scores.append({"month": month, "mape": fold_mape, "n_train": len(X_train), "n_val": len(X_val)})

    if not scores:
        return {"mape_mean": 999, "mape_std": 0, "folds": []}

    mapes = [s["mape"] for s in scores]
    return {
        "mape_mean": float(np.mean(mapes)),
        "mape_std": float(np.std(mapes)),
        "folds": scores,
    }


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train_stl_enhanced_models(
    model_types: Optional[list[str]] = None,
    save: bool = True,
) -> dict:
    """Train ML models enhanced with STL features.

    For each model type, trains:
    - Raw purchase, consume, transfer predictors
    - STL I_irregular predictors (for purchase, consume, transfer)

    Also trains direct redeem models for comparison.

    Returns dict with CV scores for all models.
    """
    if model_types is None:
        model_types = ["random_forest", "lightgbm", "extra_trees"]

    print("=" * 60)
    print("Training STL-Enhanced ML Models")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data ...")
    daily = pd.read_csv(PROCESSED_DATA_DIR / "daily_balance.csv", parse_dates=["date"])

    # Build features
    print("[2/4] Building features ...")
    history = daily[daily["date"] <= "2014-08-31"].copy()
    feat = make_features(history)
    feat = enrich_with_stl_features(feat, daily)  # use daily (has purchase/redeem)

    # Ensure we only use numeric features
    feature_cols_base = get_feature_columns(feat)
    # Remove any columns that are targets
    exclude = {"purchase", "redeem", "purchase_per_capita", "redeem_per_capita",
               "consume", "transfer"}
    feature_cols = [c for c in feature_cols_base if c not in exclude]

    # Add STL-specific features that might be object type
    for col in feat.columns:
        if col.startswith(("purchase_stl_", "redeem_stl_", "consume_stl_", "transfer_stl_")):
            if col not in feature_cols and pd.api.types.is_numeric_dtype(feat[col]):
                feature_cols.append(col)

    print(f"  Total features: {len(feature_cols)}")
    print(f"  Training samples: {len(feat)}")

    X = feat[feature_cols].values
    dates = pd.to_datetime(feat["date"])
    cv_months = [6, 7, 8]

    results = {}

    # Targets to model: purchase and redeem only
    targets = {
        "purchase": feat["purchase"].values,
        "redeem": feat["redeem"].values,
    }

    # STL I_irregular targets
    stl_targets = {}
    for t in ["purchase", "redeem"]:
        stl_col = f"{t}_stl_I_irregular"
        if stl_col in feat.columns:
            valid_mask = feat[stl_col].notna()
            if valid_mask.sum() > 20:
                stl_targets[t] = {
                    "y": feat[stl_col].values,
                    "mask": valid_mask.values,
                }

    print(f"\n[3/4] Training {len(model_types)} model types × "
          f"{len(targets) + len(stl_targets)} targets ...")

    for model_type in model_types:
        print(f"\n  --- {model_type} ---")
        results[model_type] = {}

        # --- Raw amount models ---
        for target_name, y in targets.items():
            pipe = _make_ml_model(model_type)
            cv = rolling_cv_score(pipe, X, pd.Series(y), dates, cv_months, target_name)
            pipe.fit(X, y)  # Final fit on all training data

            key = f"{target_name}_raw"
            results[model_type][key] = {
                "cv_mape_mean": cv["mape_mean"],
                "cv_mape_std": cv["mape_std"],
                "cv_folds": cv["folds"],
            }

            if save:
                model_path = MODEL_DIR / f"stl_enhanced_{model_type}_{target_name}_raw.pkl"
                save_model(pipe, model_path)

            print(f"    {target_name}_raw: CV MAPE={cv['mape_mean']:.4f} ± {cv['mape_std']:.4f}")

        # --- STL I_irregular models ---
        for target_name, info in stl_targets.items():
            mask = info["mask"]
            if mask.sum() < 20:
                continue

            X_stl = X[mask]
            y_stl = info["y"][mask]
            dates_stl = dates[mask]

            pipe = _make_ml_model(model_type)
            cv = rolling_cv_score(pipe, X_stl, pd.Series(y_stl), dates_stl, cv_months, target_name)
            pipe.fit(X_stl, y_stl)

            key = f"{target_name}_I"
            results[model_type][key] = {
                "cv_mape_mean": cv["mape_mean"],
                "cv_mape_std": cv["mape_std"],
                "cv_folds": cv["folds"],
            }

            if save:
                model_path = MODEL_DIR / f"stl_enhanced_{model_type}_{target_name}_I.pkl"
                save_model(pipe, model_path)

            print(f"    {target_name}_I:   CV MAPE={cv['mape_mean']:.4f} ± {cv['mape_std']:.4f}")

    # Save feature list
    if save:
        save_feature_list(feature_cols, MODEL_DIR / "stl_enhanced_features.json")

    # Save CV results
    cv_path = OUTPUT_DIR / "stl_enhanced_cv_results.json"
    cv_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\n[4/4] Results saved to {cv_path}")
    print("=" * 60)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train STL-enhanced ML models.")
    parser.add_argument("--models", nargs="+", default=["random_forest", "lightgbm", "extra_trees"],
                        help="Model types to train.")
    parser.add_argument("--no-save", action="store_true", help="Skip saving models.")
    args = parser.parse_args()
    train_stl_enhanced_models(args.models, save=not args.no_save)


if __name__ == "__main__":
    main()
