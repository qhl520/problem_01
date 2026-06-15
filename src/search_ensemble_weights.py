from __future__ import annotations

import json
import argparse

import pandas as pd

from config import OUTPUT_DIR, PROCESSED_DATA_DIR
from evaluate import daily_score, mape, weighted_score
from features import make_features
from model_utils import get_feature_columns, recursive_predict, train_model
from rule_models import predict_recent_average_rule, predict_weekday_rule


FOLDS = [
    ("2014-05-31", "2014-06-01", "2014-06-30"),
    ("2014-06-30", "2014-07-01", "2014-07-31"),
    ("2014-07-31", "2014-08-01", "2014-08-31"),
]

MODEL_COMPONENTS = ["random_forest", "lightgbm", "extra_trees"]
RULE_COMPONENTS = ["weekday_rule", "recent14", "recent28"]
COMPONENTS = RULE_COMPONENTS + MODEL_COMPONENTS


def _rule_prediction(name: str, history: pd.DataFrame, valid_start: str, valid_end: str) -> pd.DataFrame:
    if name == "weekday_rule":
        return predict_weekday_rule(history, valid_start, valid_end)
    if name == "recent14":
        return predict_recent_average_rule(history, valid_start, valid_end, 14)
    if name == "recent28":
        return predict_recent_average_rule(history, valid_start, valid_end, 28)
    raise ValueError(f"Unknown rule component: {name}")


def build_cv_prediction_matrix(save_predictions: bool = False) -> pd.DataFrame:
    daily = pd.read_csv(PROCESSED_DATA_DIR / "daily_balance.csv", parse_dates=["date"])
    rows = []
    for fold_id, (train_end, valid_start, valid_end) in enumerate(FOLDS, start=1):
        history = daily[daily["date"] <= train_end].copy()
        actual = daily[(daily["date"] >= valid_start) & (daily["date"] <= valid_end)][
            ["date", "purchase", "redeem"]
        ].copy()
        fold_frame = actual.rename(columns={"purchase": "actual_purchase", "redeem": "actual_redeem"})
        fold_frame["fold"] = fold_id
        fold_frame["train_end"] = train_end

        for component in RULE_COMPONENTS:
            pred = _rule_prediction(component, history, valid_start, valid_end)
            fold_frame[f"{component}_purchase"] = pred["purchase"].values
            fold_frame[f"{component}_redeem"] = pred["redeem"].values

        feat = make_features(history)
        feature_cols = get_feature_columns(feat)
        X = feat[feature_cols]
        for model_type in MODEL_COMPONENTS:
            purchase_model = train_model(model_type, X, feat["purchase"])
            redeem_model = train_model(model_type, X, feat["redeem"])
            pred = recursive_predict(history, purchase_model, redeem_model, feature_cols, valid_start, valid_end)
            fold_frame[f"{model_type}_purchase"] = pred["purchase"].values
            fold_frame[f"{model_type}_redeem"] = pred["redeem"].values

        rows.append(fold_frame)

    cv_pred = pd.concat(rows, ignore_index=True)
    if save_predictions:
        cv_pred.to_csv(OUTPUT_DIR / "optimized_cv_predictions.csv", index=False)
    return cv_pred


def _weight_grid(components: list[str], step: float = 0.1):
    units = int(round(1 / step))

    def generate(prefix: list[int], remaining_units: int, remaining_slots: int):
        if remaining_slots == 1:
            yield prefix + [remaining_units]
            return
        for value in range(remaining_units + 1):
            yield from generate(prefix + [value], remaining_units - value, remaining_slots - 1)

    for values in generate([], units, len(components)):
        yield {component: value / units for component, value in zip(components, values)}


def _score_weights(cv_pred: pd.DataFrame, target: str, weights: dict[str, float]) -> float:
    pred = sum(weights[name] * cv_pred[f"{name}_{target}"] for name in COMPONENTS)
    return daily_score(cv_pred[f"actual_{target}"], pred)


def search_weights(cv_pred: pd.DataFrame, step: float = 0.1, top_n: int = 100) -> dict:
    best = {}
    rows = []
    for target in ["purchase", "redeem"]:
        best_score = -1.0
        best_weights = None
        for weights in _weight_grid(COMPONENTS, step=step):
            score = _score_weights(cv_pred, target, weights)
            rows.append({"target": target, "score": score, **weights})
            if score > best_score:
                best_score = score
                best_weights = weights
        best[f"{target}_weights"] = best_weights
        best[f"{target}_score"] = best_score

    best["weighted_score"] = 0.45 * best["purchase_score"] + 0.55 * best["redeem_score"]
    best["components"] = COMPONENTS
    (
        pd.DataFrame(rows)
        .sort_values(["target", "score"], ascending=[True, False])
        .groupby("target", group_keys=False)
        .head(top_n)
        .to_csv(OUTPUT_DIR / "ensemble_weight_search_top.csv", index=False)
    )
    (OUTPUT_DIR / "best_ensemble_weights.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    return best


def save_optimized_validation(cv_pred: pd.DataFrame, best: dict) -> None:
    purchase_weights = best["purchase_weights"]
    redeem_weights = best["redeem_weights"]
    rows = []
    scored = cv_pred.copy()
    scored["pred_purchase"] = sum(purchase_weights[name] * scored[f"{name}_purchase"] for name in COMPONENTS)
    scored["pred_redeem"] = sum(redeem_weights[name] * scored[f"{name}_redeem"] for name in COMPONENTS)

    for fold, part in scored.groupby("fold"):
        rows.append(
            {
                "fold": fold,
                "valid_start": part["date"].min().strftime("%Y-%m-%d"),
                "valid_end": part["date"].max().strftime("%Y-%m-%d"),
                "purchase_mape": mape(part["actual_purchase"], part["pred_purchase"]),
                "redeem_mape": mape(part["actual_redeem"], part["pred_redeem"]),
                "purchase_score": daily_score(part["actual_purchase"], part["pred_purchase"]),
                "redeem_score": daily_score(part["actual_redeem"], part["pred_redeem"]),
                "weighted_score": weighted_score(
                    part["actual_purchase"], part["pred_purchase"], part["actual_redeem"], part["pred_redeem"]
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.loc[len(result)] = {
        "fold": "mean",
        "valid_start": "",
        "valid_end": "",
        "purchase_mape": result["purchase_mape"].mean(),
        "redeem_mape": result["redeem_mape"].mean(),
        "purchase_score": result["purchase_score"].mean(),
        "redeem_score": result["redeem_score"].mean(),
        "weighted_score": result["weighted_score"].mean(),
    }
    result.to_csv(OUTPUT_DIR / "optimized_ensemble_validation.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true", help="Reuse output/optimized_cv_predictions.csv if present.")
    parser.add_argument("--save-predictions", action="store_true", help="Save the full CV prediction matrix.")
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    cached_path = OUTPUT_DIR / "optimized_cv_predictions.csv"
    if args.reuse and cached_path.exists():
        cv_pred = pd.read_csv(cached_path, parse_dates=["date"])
    else:
        cv_pred = build_cv_prediction_matrix(save_predictions=args.save_predictions)
    best = search_weights(cv_pred, step=args.step, top_n=args.top_n)
    save_optimized_validation(cv_pred, best)
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
