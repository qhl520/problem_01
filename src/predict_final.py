from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import FINAL_SUBMISSION_PATH, MODEL_DIR, OUTPUT_DIR, PREDICT_END_DATE, PREDICT_START_DATE, PROCESSED_DATA_DIR, SUBMISSION_DIR
from data_utils import validate_submission
from model_utils import load_feature_list, load_model, recursive_predict
from rule_models import predict_recent_average_rule, predict_weekday_rule, to_submission


MODEL_COMPONENTS = ["random_forest", "lightgbm", "extra_trees"]


def _component_predictions(daily: pd.DataFrame, feature_cols: list[str]) -> dict[str, pd.DataFrame]:
    preds: dict[str, pd.DataFrame] = {
        "weekday_rule": predict_weekday_rule(daily, PREDICT_START_DATE, PREDICT_END_DATE),
        "recent14": predict_recent_average_rule(daily, PREDICT_START_DATE, PREDICT_END_DATE, 14),
        "recent28": predict_recent_average_rule(daily, PREDICT_START_DATE, PREDICT_END_DATE, 28),
    }
    for model_type in MODEL_COMPONENTS:
        purchase_path = MODEL_DIR / f"final_{model_type}_purchase_model.pkl"
        redeem_path = MODEL_DIR / f"final_{model_type}_redeem_model.pkl"
        if not purchase_path.exists() or not redeem_path.exists():
            continue
        purchase_model = load_model(purchase_path)
        redeem_model = load_model(redeem_path)
        preds[model_type] = recursive_predict(
            daily, purchase_model, redeem_model, feature_cols, PREDICT_START_DATE, PREDICT_END_DATE
        )
    return preds


def _load_weights(available_components: list[str]) -> dict:
    weights_path = OUTPUT_DIR / "best_ensemble_weights.json"
    if weights_path.exists():
        raw = json.loads(weights_path.read_text(encoding="utf-8"))
        if "purchase_weights" in raw and "redeem_weights" in raw:
            return raw

    fallback = {name: 1.0 / len(available_components) for name in available_components}
    return {
        "components": available_components,
        "purchase_weights": fallback,
        "redeem_weights": fallback,
        "purchase_score": None,
        "redeem_score": None,
        "weighted_score": None,
    }


def _normalize_weights(weights: dict[str, float], available_components: list[str]) -> dict[str, float]:
    filtered = {name: float(weights.get(name, 0.0)) for name in available_components}
    total = sum(filtered.values())
    if total <= 0:
        return {name: 1.0 / len(available_components) for name in available_components}
    return {name: value / total for name, value in filtered.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-components", action="store_true", help="Write per-component submission files for inspection.")
    args = parser.parse_args()

    daily = pd.read_csv(PROCESSED_DATA_DIR / "daily_balance.csv", parse_dates=["date"])
    feature_cols = load_feature_list(MODEL_DIR / "final_features.json")
    preds = _component_predictions(daily, feature_cols)
    available = list(preds)
    weights = _load_weights(available)
    purchase_weights = _normalize_weights(weights.get("purchase_weights", {}), available)
    redeem_weights = _normalize_weights(weights.get("redeem_weights", {}), available)

    pred = next(iter(preds.values()))[["date"]].copy()
    pred["purchase"] = 0.0
    pred["redeem"] = 0.0
    for name, frame in preds.items():
        pred["purchase"] += purchase_weights.get(name, 0.0) * frame["purchase"]
        pred["redeem"] += redeem_weights.get(name, 0.0) * frame["redeem"]

    sub = to_submission(pred)
    sub.to_csv(FINAL_SUBMISSION_PATH, index=False, header=False)
    validate_submission(FINAL_SUBMISSION_PATH)

    if args.save_components:
        SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
        sub.to_csv(SUBMISSION_DIR / "submission_optimized_ensemble.csv", index=False, header=False)
        for name, frame in preds.items():
            to_submission(frame).to_csv(SUBMISSION_DIR / f"submission_component_{name}.csv", index=False, header=False)

    (OUTPUT_DIR / "final_used_weights.json").write_text(
        json.dumps(
            {
                "purchase_weights": purchase_weights,
                "redeem_weights": redeem_weights,
                "validation_weighted_score": weights.get("weighted_score"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    plt.figure(figsize=(10, 5))
    dates = pd.to_datetime(sub["report_date"].astype(str))
    plt.plot(dates, sub["purchase"], label="purchase")
    plt.plot(dates, sub["redeem"], label="redeem")
    plt.title("Final Optimized September Prediction")
    plt.xlabel("Date")
    plt.ylabel("Amount")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "final_prediction_plot.png", dpi=160)
    plt.close()
    print(f"Optimized final submission written to: {FINAL_SUBMISSION_PATH}")
    print(f"Used purchase weights: {purchase_weights}")
    print(f"Used redeem weights: {redeem_weights}")


if __name__ == "__main__":
    main()
