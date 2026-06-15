from __future__ import annotations

import pandas as pd

from config import MODEL_DIR, PREDICT_END_DATE, PREDICT_START_DATE, PROCESSED_DATA_DIR, SUBMISSION_DIR
from data_utils import validate_submission
from model_utils import load_feature_list, load_model, recursive_predict
from rule_models import to_submission


def main() -> None:
    daily = pd.read_csv(PROCESSED_DATA_DIR / "daily_balance.csv", parse_dates=["date"])
    purchase_model = load_model(MODEL_DIR / "lightgbm_purchase.pkl")
    redeem_model = load_model(MODEL_DIR / "lightgbm_redeem.pkl")
    feature_cols = load_feature_list(MODEL_DIR / "lightgbm_features.json")
    pred = recursive_predict(daily, purchase_model, redeem_model, feature_cols, PREDICT_START_DATE, PREDICT_END_DATE)
    sub = to_submission(pred)
    out_path = SUBMISSION_DIR / "submission_v3_lightgbm.csv"
    sub.to_csv(out_path, index=False, header=False)
    validate_submission(out_path)
    print(f"LightGBM submission written to: {out_path}")


if __name__ == "__main__":
    main()
