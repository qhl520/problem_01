from __future__ import annotations

import pandas as pd

from config import LOG_DIR, MODEL_DIR, OUTPUT_DIR, PROCESSED_DATA_DIR
from evaluate import mape, weighted_score
from features import make_features
from model_utils import append_experiment, feature_importance_frame, get_feature_columns, recursive_predict, save_feature_list, save_model, train_model


def main() -> None:
    daily = pd.read_csv(PROCESSED_DATA_DIR / "daily_balance.csv", parse_dates=["date"])
    train_end, valid_start, valid_end = "2014-07-31", "2014-08-01", "2014-08-31"
    train_history = daily[daily["date"] <= train_end].copy()
    train_feat = make_features(train_history)
    feature_cols = get_feature_columns(train_feat)
    X = train_feat[feature_cols]
    purchase_model = train_model("lightgbm", X, train_feat["purchase"])
    redeem_model = train_model("lightgbm", X, train_feat["redeem"])
    pred = recursive_predict(train_history, purchase_model, redeem_model, feature_cols, valid_start, valid_end)
    actual = daily[(daily["date"] >= valid_start) & (daily["date"] <= valid_end)]
    result = pd.DataFrame(
        [
            {
                "model": "lightgbm_or_fallback",
                "train_range": f"<= {train_end}",
                "valid_range": f"{valid_start} to {valid_end}",
                "purchase_mape": mape(actual["purchase"], pred["purchase"]),
                "redeem_mape": mape(actual["redeem"], pred["redeem"]),
                "weighted_score": weighted_score(actual["purchase"], pred["purchase"], actual["redeem"], pred["redeem"]),
            }
        ]
    )
    result.to_csv(OUTPUT_DIR / "validation_lightgbm.csv", index=False)
    save_model(purchase_model, MODEL_DIR / "lightgbm_purchase.pkl")
    save_model(redeem_model, MODEL_DIR / "lightgbm_redeem.pkl")
    save_feature_list(feature_cols, MODEL_DIR / "lightgbm_features.json")
    feature_importance_frame(purchase_model, feature_cols).to_csv(OUTPUT_DIR / "feature_importance_lgb_purchase.csv", index=False)
    feature_importance_frame(redeem_model, feature_cols).to_csv(OUTPUT_DIR / "feature_importance_lgb_redeem.csv", index=False)
    append_experiment(
        LOG_DIR / "experiment_log.csv",
        {
            "model": "lightgbm_or_fallback",
            "features_version": "v1",
            "train_range": f"<= {train_end}",
            "valid_range": f"{valid_start} to {valid_end}",
            "purchase_mape": result.loc[0, "purchase_mape"],
            "redeem_mape": result.loc[0, "redeem_mape"],
            "purchase_score": "",
            "redeem_score": "",
            "total_score": result.loc[0, "weighted_score"],
            "notes": "recursive validation",
        },
    )
    print(result)


if __name__ == "__main__":
    main()
