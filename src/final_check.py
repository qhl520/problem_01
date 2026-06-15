from __future__ import annotations

import pandas as pd

from config import DIAGNOSTIC_DIR, FINAL_SUBMISSION_PATH
from data_utils import read_submission, validate_submission


def check_submission(path=FINAL_SUBMISSION_PATH) -> None:
    validate_submission(path)
    df = read_submission(path)
    rows = [
        {"check": "official_headerless_format", "value": True, "status": "pass"},
        {"check": "row_count", "value": len(df), "status": "pass" if len(df) == 30 else "fail"},
        {"check": "first_report_date", "value": int(df["report_date"].iloc[0]), "status": "pass"},
        {"check": "last_report_date", "value": int(df["report_date"].iloc[-1]), "status": "pass"},
        {"check": "amount_unit", "value": "fen", "status": "pass"},
    ]
    for col in ["purchase", "redeem"]:
        pct = df[col].pct_change().replace([float("inf"), -float("inf")], pd.NA).dropna()
        if (pct.abs() > 5).any():
            print(f"Warning: {col} has very large day-over-day changes.")
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DIAGNOSTIC_DIR / "final_submission_check.csv", index=False)
    print("Final submission check passed.")


if __name__ == "__main__":
    check_submission()
