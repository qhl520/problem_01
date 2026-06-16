"""DEPRECATED: Legacy single-model RandomForest prediction script.

Reads ``random_forest_purchase.pkl`` / ``random_forest_redeem.pkl`` and produces a
single-model submission. These model files are saved by the deprecated
``src/train_random_forest.py``, NOT by the canonical ``src/train_final.py``.

Use the canonical pipeline instead:
    ``src/predict_final.py`` — ensemble prediction using all model + rule components.

Kept for reference only; not used by ``run_all.py`` or ``run_original_plus_121.py``.
"""

from __future__ import annotations

import pandas as pd

from config import PREDICT_END_DATE, PREDICT_START_DATE, PROCESSED_DATA_DIR, SUBMISSION_DIR
from data_utils import validate_submission
from model_utils import load_feature_list, load_model, recursive_predict
from rule_models import to_submission


def main() -> None:
    daily = pd.read_csv(PROCESSED_DATA_DIR / "daily_balance.csv", parse_dates=["date"])
    purchase_model = load_model(SUBMISSION_DIR.parent / "models" / "random_forest_purchase.pkl")
    redeem_model = load_model(SUBMISSION_DIR.parent / "models" / "random_forest_redeem.pkl")
    feature_cols = load_feature_list(SUBMISSION_DIR.parent / "models" / "random_forest_features.json")
    pred = recursive_predict(daily, purchase_model, redeem_model, feature_cols, PREDICT_START_DATE, PREDICT_END_DATE)
    sub = to_submission(pred)
    out_path = SUBMISSION_DIR / "submission_v2_random_forest.csv"
    sub.to_csv(out_path, index=False, header=False)
    validate_submission(out_path)
    print(f"RandomForest submission written to: {out_path}")


if __name__ == "__main__":
    main()
