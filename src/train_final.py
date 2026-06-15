from __future__ import annotations

import pandas as pd

from config import MODEL_DIR, OUTPUT_DIR, PROCESSED_DATA_DIR
from features import make_features
from model_utils import feature_importance_frame, get_feature_columns, save_feature_list, save_model, train_model


def main() -> None:
    daily = pd.read_csv(PROCESSED_DATA_DIR / "daily_balance.csv", parse_dates=["date"])
    history = daily[daily["date"] <= "2014-08-31"].copy()
    feat = make_features(history)
    feature_cols = get_feature_columns(feat)
    X = feat[feature_cols]
    for model_type in ["random_forest", "lightgbm", "extra_trees"]:
        purchase_model = train_model(model_type, X, feat["purchase"])
        redeem_model = train_model(model_type, X, feat["redeem"])
        save_model(purchase_model, MODEL_DIR / f"final_{model_type}_purchase_model.pkl")
        save_model(redeem_model, MODEL_DIR / f"final_{model_type}_redeem_model.pkl")
        feature_importance_frame(purchase_model, feature_cols).to_csv(
            OUTPUT_DIR / f"final_feature_importance_{model_type}_purchase.csv", index=False
        )
        feature_importance_frame(redeem_model, feature_cols).to_csv(
            OUTPUT_DIR / f"final_feature_importance_{model_type}_redeem.csv", index=False
        )

    save_feature_list(feature_cols, MODEL_DIR / "final_features.json")
    print("Final candidate models trained and saved.")


if __name__ == "__main__":
    main()
