from __future__ import annotations

import argparse

import pandas as pd

from config import OUTPUT_DIR, PROCESSED_DATA_DIR
from evaluate import mape, weighted_score
from features import make_features
from model_utils import get_feature_columns, recursive_predict, train_model


def run_cv(model_type: str) -> pd.DataFrame:
    daily = pd.read_csv(PROCESSED_DATA_DIR / "daily_balance.csv", parse_dates=["date"])
    folds = [
        ("2014-05-31", "2014-06-01", "2014-06-30"),
        ("2014-06-30", "2014-07-01", "2014-07-31"),
        ("2014-07-31", "2014-08-01", "2014-08-31"),
    ]
    rows = []
    for i, (train_end, valid_start, valid_end) in enumerate(folds, start=1):
        history = daily[daily["date"] <= train_end].copy()
        feat = make_features(history)
        feature_cols = get_feature_columns(feat)
        X = feat[feature_cols]
        purchase_model = train_model(model_type, X, feat["purchase"])
        redeem_model = train_model(model_type, X, feat["redeem"])
        pred = recursive_predict(history, purchase_model, redeem_model, feature_cols, valid_start, valid_end)
        actual = daily[(daily["date"] >= valid_start) & (daily["date"] <= valid_end)]
        rows.append(
            {
                "fold": i,
                "model": model_type,
                "train_end": train_end,
                "valid_start": valid_start,
                "valid_end": valid_end,
                "purchase_mape": mape(actual["purchase"], pred["purchase"]),
                "redeem_mape": mape(actual["redeem"], pred["redeem"]),
                "weighted_score": weighted_score(actual["purchase"], pred["purchase"], actual["redeem"], pred["redeem"]),
            }
        )
    result = pd.DataFrame(rows)
    result.loc[len(result)] = {
        "fold": "mean",
        "model": model_type,
        "train_end": "",
        "valid_start": "",
        "valid_end": "",
        "purchase_mape": result["purchase_mape"].mean(),
        "redeem_mape": result["redeem_mape"].mean(),
        "weighted_score": result["weighted_score"].mean(),
    }
    output_path = OUTPUT_DIR / f"cross_validation_results_{model_type}.csv"
    result.to_csv(output_path, index=False)
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "random_forest", "extra_trees", "hist_gradient_boosting"])
    args = parser.parse_args()
    run_cv(args.model)


if __name__ == "__main__":
    main()
