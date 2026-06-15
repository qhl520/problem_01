from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DIAGNOSTIC_DIR, FINAL_SUBMISSION_PATH, RAW_DATA_DIR
from data_utils import parse_competition_date, read_csv_safely, read_submission, validate_submission
from evaluate import ERROR_CUTOFF, ERROR_ZERO_SCORE, PURCHASE_WEIGHT, REDEEM_WEIGHT


def _status(condition: bool) -> str:
    return "pass" if condition else "fail"


def _columns(path: Path) -> set[str]:
    df, _ = read_csv_safely(path, nrows=5)
    return {str(col) for col in df.columns}


def main() -> None:
    rows: list[dict] = []

    files = {
        "user_profile_table.csv": RAW_DATA_DIR / "user_profile_table.csv",
        "user_balance_table.csv": RAW_DATA_DIR / "user_balance_table.csv",
        "mfd_day_share_interest.csv": RAW_DATA_DIR / "mfd_day_share_interest.csv",
        "mfd_bank_shibor.csv": RAW_DATA_DIR / "mfd_bank_shibor.csv",
        "comp_predict_table.csv": RAW_DATA_DIR / "comp_predict_table.csv",
    }
    for name, path in files.items():
        rows.append({"requirement": f"raw_file_exists:{name}", "value": path.exists(), "status": _status(path.exists())})

    profile_required = {"user_id", "sex", "city", "constellation"}
    balance_required = {
        "user_id",
        "report_date",
        "tBalance",
        "yBalance",
        "total_purchase_amt",
        "total_redeem_amt",
        "consume_amt",
        "category1",
        "category2",
        "category3",
        "category4",
    }
    yield_required = {"mfd_date", "mfd_daily_yield", "mfd_7daily_yield"}
    shibor_required = {
        "mfd_date",
        "Interest_O_N",
        "Interest_1_W",
        "Interest_2_W",
        "Interest_1_M",
        "Interest_3_M",
        "Interest_6_M",
        "Interest_9_M",
        "Interest_1_Y",
    }

    table_checks = [
        ("user_profile_columns", files["user_profile_table.csv"], profile_required),
        ("user_balance_columns", files["user_balance_table.csv"], balance_required),
        ("yield_columns", files["mfd_day_share_interest.csv"], yield_required),
        ("shibor_columns", files["mfd_bank_shibor.csv"], shibor_required),
    ]
    for name, path, required in table_checks:
        if path.exists():
            cols = _columns(path)
            missing = sorted(required - cols)
            rows.append({"requirement": name, "value": "missing=" + ",".join(missing), "status": _status(not missing)})

    if files["user_balance_table.csv"].exists():
        dates_df, _ = read_csv_safely(files["user_balance_table.csv"], usecols=["report_date"])
        dates = parse_competition_date(dates_df["report_date"])
        rows.extend(
            [
                {
                    "requirement": "user_balance_start_date_is_20130701",
                    "value": dates.min().strftime("%Y%m%d"),
                    "status": _status(dates.min().strftime("%Y%m%d") == "20130701"),
                },
                {
                    "requirement": "user_balance_end_date_is_20140831",
                    "value": dates.max().strftime("%Y%m%d"),
                    "status": _status(dates.max().strftime("%Y%m%d") == "20140831"),
                },
            ]
        )

    try:
        with FINAL_SUBMISSION_PATH.open("r", encoding="utf-8") as fh:
            first_line = fh.readline().strip()
        has_header = first_line.lower() == "report_date,purchase,redeem"
        validate_submission(FINAL_SUBMISSION_PATH)
        sub = read_submission(FINAL_SUBMISSION_PATH)
        rows.extend(
            [
                {"requirement": "final_submission_exists", "value": True, "status": "pass"},
                {"requirement": "final_submission_headerless", "value": not has_header, "status": _status(not has_header)},
                {"requirement": "final_submission_row_count_30", "value": len(sub), "status": _status(len(sub) == 30)},
                {
                    "requirement": "final_submission_first_date_20140901",
                    "value": int(sub["report_date"].iloc[0]),
                    "status": _status(int(sub["report_date"].iloc[0]) == 20140901),
                },
                {
                    "requirement": "final_submission_last_date_20140930",
                    "value": int(sub["report_date"].iloc[-1]),
                    "status": _status(int(sub["report_date"].iloc[-1]) == 20140930),
                },
                {
                    "requirement": "final_submission_amounts_integer_fen",
                    "value": str(sub[["purchase", "redeem"]].dtypes.to_dict()),
                    "status": _status(all(pd.api.types.is_integer_dtype(sub[col]) for col in ["purchase", "redeem"])),
                },
                {
                    "requirement": "final_submission_amounts_non_negative",
                    "value": int((sub[["purchase", "redeem"]] < 0).sum().sum()),
                    "status": _status((sub[["purchase", "redeem"]] >= 0).all().all()),
                },
            ]
        )
    except Exception as exc:
        rows.append({"requirement": "final_submission_valid", "value": str(exc), "status": "fail"})

    rows.extend(
        [
            {"requirement": "score_zero_error_is_10", "value": ERROR_ZERO_SCORE, "status": _status(ERROR_ZERO_SCORE == 10.0)},
            {"requirement": "score_cutoff_relative_error_is_0_3", "value": ERROR_CUTOFF, "status": _status(ERROR_CUTOFF == 0.3)},
            {"requirement": "purchase_weight_is_45_percent", "value": PURCHASE_WEIGHT, "status": _status(PURCHASE_WEIGHT == 0.45)},
            {"requirement": "redeem_weight_is_55_percent", "value": REDEEM_WEIGHT, "status": _status(REDEEM_WEIGHT == 0.55)},
        ]
    )

    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out_path = DIAGNOSTIC_DIR / "official_requirement_check.csv"
    out.to_csv(out_path, index=False)
    print(out)
    failed = out[out["status"] == "fail"]
    if not failed.empty:
        raise SystemExit(f"Official requirement check failed: {failed['requirement'].tolist()}")
    print(f"Official requirement check written to: {out_path}")


if __name__ == "__main__":
    main()
