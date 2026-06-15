from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from features import make_features


warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    forbidden = {"date", "purchase", "redeem", "report_date"}
    cols = []
    for col in df.columns:
        if col in forbidden:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def make_model(model_type: str):
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=360,
            max_depth=14,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=420,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(
                n_estimators=1200,
                learning_rate=0.025,
                num_leaves=15,
                max_depth=-1,
                min_child_samples=12,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.25,
                reg_lambda=0.75,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
        except Exception as exc:
            print(f"LightGBM unavailable, falling back to HistGradientBoostingRegressor: {exc}")
            model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, random_state=42)
    elif model_type == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def train_model(model_type: str, X_train: pd.DataFrame, y_train: pd.Series):
    model = make_model(model_type)
    model.fit(X_train, y_train)
    return model


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    return joblib.load(path)


def save_feature_list(features: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")


def load_feature_list(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_importance_frame(model, feature_cols: list[str]) -> pd.DataFrame:
    estimator = model.named_steps.get("model", model)
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = np.abs(estimator.coef_)
    else:
        importance = np.zeros(len(feature_cols))
    return pd.DataFrame({"feature": feature_cols, "importance": importance}).sort_values(
        "importance", ascending=False
    )


def recursive_predict(
    history_df: pd.DataFrame,
    model_purchase,
    model_redeem,
    feature_cols: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    history = history_df.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date").reset_index(drop=True)
    rows = []
    for date in pd.date_range(start_date, end_date, freq="D"):
        candidate = pd.concat(
            [history, pd.DataFrame([{"date": date, "purchase": np.nan, "redeem": np.nan}])],
            ignore_index=True,
        )
        feat = make_features(candidate)
        x = feat.loc[feat["date"] == date, feature_cols]
        purchase = max(0, int(round(float(model_purchase.predict(x)[0]))))
        redeem = max(0, int(round(float(model_redeem.predict(x)[0]))))
        row = {"date": date, "purchase": purchase, "redeem": redeem}
        rows.append(row)
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    return pd.DataFrame(rows)


def append_experiment(log_path: Path, row: dict) -> None:
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), **row}
    df = pd.DataFrame([row])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)
