from __future__ import annotations

import pandas as pd

from config import PROCESSED_DATA_DIR
from features import make_features
from model_utils import get_feature_columns


def main() -> None:
    df = pd.read_csv(PROCESSED_DATA_DIR / "daily_balance.csv", parse_dates=["date"])
    feat = make_features(df)
    feature_cols = get_feature_columns(feat)
    leakage = {"date", "purchase", "redeem", "report_date"} & set(feature_cols)
    if leakage:
        raise ValueError(f"Feature columns contain leakage fields: {sorted(leakage)}")
    print(f"Feature columns: {len(feature_cols)}")
    print(feat.head())
    print("Top missing features:")
    print(feat[feature_cols].isna().sum().sort_values(ascending=False).head(15))
    print("Early NaN values are expected for lag and rolling windows.")


if __name__ == "__main__":
    main()
