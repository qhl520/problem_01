from __future__ import annotations

import argparse
import shutil
import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from config import FINAL_SUBMISSION_PATH, OUTPUT_DIR, RAW_DATA_DIR
from data_utils import find_user_balance_file, parse_competition_date, read_csv_safely, read_submission, validate_submission


ORIGINAL_DIR = OUTPUT_DIR / "original"
ORIGINAL_INITIAL_PATH = ORIGINAL_DIR / "tc_initial.csv"
FINAL_DIR = OUTPUT_DIR / "high_score_final"
MAIN_121_PATH = FINAL_DIR / "tc_comp_predict_table.csv"
CONSERVATIVE_121_PATH = FINAL_DIR / "tc_comp_predict_table_conservative.csv"


PURCHASE_ADJUSTMENTS = {
    20140906: -0.015,
    20140907: -0.015,
    20140908: -0.030,
    20140909: 0.015,
    20140929: 0.015,
    20140930: 0.015,
}

REDEEM_ADJUSTMENTS = {
    20140906: -0.025,
    20140907: -0.025,
    20140908: -0.018,
    20140909: 0.018,
    20140929: 0.018,
    20140930: 0.018,
}


def _read_git_initial() -> pd.DataFrame | None:
    try:
        text = subprocess.check_output(
            ["git", "show", "HEAD:output/tc_comp_predict_table.csv"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return pd.read_csv(StringIO(text), header=None, names=["report_date", "purchase", "redeem"])


def _load_initial(path: Path | None) -> pd.DataFrame:
    if path is not None:
        df = read_submission(path)
    elif ORIGINAL_INITIAL_PATH.exists():
        df = read_submission(ORIGINAL_INITIAL_PATH)
    else:
        git_initial = _read_git_initial()
        df = git_initial if git_initial is not None else read_submission(FINAL_SUBMISSION_PATH)
    df["date"] = pd.to_datetime(df["report_date"].astype(str), format="%Y%m%d")
    df["weekday"] = df["date"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6])
    return df


def _load_daily_actual() -> pd.DataFrame:
    path = find_user_balance_file(RAW_DATA_DIR)
    if path is None:
        raise FileNotFoundError("No user_balance_table-like file found in data/raw/.")
    df, _ = read_csv_safely(path, usecols=["report_date", "total_purchase_amt", "total_redeem_amt"])
    df["date"] = parse_competition_date(df["report_date"])
    daily = (
        df.groupby("date", as_index=False)
        .agg(purchase=("total_purchase_amt", "sum"), redeem=("total_redeem_amt", "sum"))
        .sort_values("date")
    )
    daily["weekday"] = daily["date"].dt.weekday
    return daily


def _normalize(values: pd.Series, total: float) -> pd.Series:
    values = pd.Series(values, dtype=float).clip(lower=0)
    if values.sum() <= 0:
        return pd.Series(np.repeat(total / len(values), len(values)), index=values.index)
    return values * total / values.sum()


def _weighted_mean(values: pd.Series) -> float:
    values = pd.Series(values, dtype=float)
    if values.empty:
        return 0.0
    weights = np.arange(1, len(values) + 1, dtype=float)
    return float(np.average(values, weights=weights))


def _round_preserve_total(values: pd.Series, total: int) -> pd.Series:
    values = pd.Series(values, dtype=float).clip(lower=0)
    floors = np.floor(values).astype("int64")
    diff = int(total - floors.sum())
    frac = values - floors
    order = np.argsort(-frac.to_numpy()) if diff >= 0 else np.argsort(frac.to_numpy())
    result = floors.copy()
    if diff > 0:
        for idx in order[:diff]:
            result.iloc[idx] += 1
    elif diff < 0:
        needed = -diff
        for idx in order:
            if needed <= 0:
                break
            if result.iloc[idx] > 0:
                result.iloc[idx] -= 1
                needed -= 1
    return result.astype("int64")


def _shape_from_august_weekday(initial: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    aug = daily[(daily["date"] >= "2014-08-01") & (daily["date"] <= "2014-08-31")]
    med = aug.groupby("weekday")[["purchase", "redeem"]].median()
    raw = pd.DataFrame(
        {
            "report_date": initial["report_date"],
            "purchase": [med.loc[w, "purchase"] for w in initial["weekday"]],
            "redeem": [med.loc[w, "redeem"] for w in initial["weekday"]],
        }
    )
    return _normalize_shape(raw, initial)


def _shape_from_recent8_weekday(initial: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    recent = daily[(daily["date"] >= "2014-07-01") & (daily["date"] <= "2014-08-31")]
    rows = []
    for _, row in initial.iterrows():
        same = recent[recent["weekday"].eq(row["weekday"])].sort_values("date")
        rows.append(
            {
                "report_date": row["report_date"],
                "purchase": 0.5 * same["purchase"].median() + 0.5 * _weighted_mean(same["purchase"]),
                "redeem": 0.5 * same["redeem"].median() + 0.5 * _weighted_mean(same["redeem"]),
            }
        )
    return _normalize_shape(pd.DataFrame(rows), initial)


def _normalize_shape(shape: pd.DataFrame, initial: pd.DataFrame) -> pd.DataFrame:
    out = shape.copy()
    out["purchase"] = _normalize(out["purchase"], initial["purchase"].sum())
    out["redeem"] = _normalize(out["redeem"], initial["redeem"].sum())
    return out


def _blend_and_adjust(initial: pd.DataFrame, shape: pd.DataFrame) -> pd.DataFrame:
    out = initial[["report_date", "purchase", "redeem"]].merge(shape, on="report_date", suffixes=("_initial", "_shape"))
    out["purchase"] = 0.90 * out["purchase_initial"] + 0.10 * out["purchase_shape"]
    out["redeem"] = 0.90 * out["redeem_initial"] + 0.10 * out["redeem_shape"]
    out["purchase"] = _normalize(out["purchase"], initial["purchase"].sum())
    out["redeem"] = _normalize(out["redeem"], initial["redeem"].sum())
    out["purchase_factor"] = out["report_date"].map(PURCHASE_ADJUSTMENTS).fillna(0.0) + 1.0
    out["redeem_factor"] = out["report_date"].map(REDEEM_ADJUSTMENTS).fillna(0.0) + 1.0
    out["purchase"] = _normalize(out["purchase"] * out["purchase_factor"], initial["purchase"].sum())
    out["redeem"] = _normalize(out["redeem"] * out["redeem_factor"], initial["redeem"].sum())
    return pd.DataFrame(
        {
            "report_date": out["report_date"].astype("int64"),
            "purchase": _round_preserve_total(out["purchase"], int(initial["purchase"].sum())),
            "redeem": _round_preserve_total(out["redeem"], int(initial["redeem"].sum())),
        }
    )


def _write_submission(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, header=False)
    validate_submission(path)


def generate_121_submissions(initial_path: Path | None = None, copy_default: bool = True) -> None:
    initial = _load_initial(initial_path)
    daily = _load_daily_actual()
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    _write_submission(initial[["report_date", "purchase", "redeem"]], ORIGINAL_INITIAL_PATH)

    main = _blend_and_adjust(initial, _shape_from_august_weekday(initial, daily))
    conservative = _blend_and_adjust(initial, _shape_from_recent8_weekday(initial, daily))
    _write_submission(main, MAIN_121_PATH)
    _write_submission(conservative, CONSERVATIVE_121_PATH)
    if copy_default:
        shutil.copyfile(MAIN_121_PATH, FINAL_SUBMISSION_PATH)
        validate_submission(FINAL_SUBMISSION_PATH)

    print(f"Original initial: {ORIGINAL_INITIAL_PATH}")
    print(f"121 main submission: {MAIN_121_PATH}")
    print(f"121 conservative submission: {CONSERVATIVE_121_PATH}")
    print(f"Default submission updated: {FINAL_SUBMISSION_PATH if copy_default else 'no'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, default=None, help="Original model submission. Defaults to output/original/tc_initial.csv, then git HEAD, then current final.")
    parser.add_argument("--no-copy-default", action="store_true", help="Do not copy the 121 main submission to output/tc_comp_predict_table.csv.")
    args = parser.parse_args()
    generate_121_submissions(args.initial, copy_default=not args.no_copy_default)


if __name__ == "__main__":
    main()
