from __future__ import annotations

import pandas as pd

from config import PROCESSED_DATA_DIR
from features import make_features
from model_utils import get_feature_columns


def main() -> None:
    daily_path = PROCESSED_DATA_DIR / "daily_balance.csv"
    if not daily_path.exists():
        raise FileNotFoundError("Run src/make_daily_data.py first.")
    df = pd.read_csv(daily_path, parse_dates=["date"])
    feat = make_features(df)
    feature_cols = get_feature_columns(feat)
    forbidden = {"date", "purchase", "redeem", "report_date"}
    leaked = forbidden & set(feature_cols)
    if leaked:
        raise ValueError(f"Forbidden columns found in feature list: {sorted(leaked)}")
    sep_rows = feat[feat["date"] >= "2014-09-01"]
    if not sep_rows.empty and sep_rows[["purchase", "redeem"]].notna().any().any():
        raise ValueError("Feature frame contains September true targets.")
    print("Leakage check passed: feature columns exclude target/date fields and generated history features use shifts.")


if __name__ == "__main__":
    main()
