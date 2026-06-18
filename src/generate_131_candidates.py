from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import FINAL_SUBMISSION_PATH, OUTPUT_DIR
from config import PREDICT_END_DATE, PREDICT_START_DATE, RAW_DATA_DIR
from data_utils import find_user_balance_file, read_csv_safely, read_submission, validate_submission


BASELINE_DIR = OUTPUT_DIR / "baseline_131"
BASELINE_PATH = BASELINE_DIR / "tc_comp_predict_table.csv"
BASELINE_TOTALS_PATH = BASELINE_DIR / "target_totals.json"
BEST_131_PATH = OUTPUT_DIR / "best_131" / "tc_comp_predict_table.csv"
DEFAULT_CANDIDATE_DIR = OUTPUT_DIR / "candidates_131"


@dataclass(frozen=True)
class CandidateMeta:
    filename: str
    priority: int
    hypothesis: str
    target: str
    dates: str
    total_policy: str
    risk: str
    recommended_action: str


def ensure_baseline_131() -> None:
    """Create the protected 131 baseline once; never overwrite it."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    if BASELINE_PATH.exists():
        validate_submission(BASELINE_PATH)
        return

    if BEST_131_PATH.exists():
        source = BEST_131_PATH
    elif FINAL_SUBMISSION_PATH.exists():
        source = FINAL_SUBMISSION_PATH
    else:
        raise FileNotFoundError(
            "Cannot initialize baseline_131: missing output/best_131/tc_comp_predict_table.csv "
            "and output/tc_comp_predict_table.csv."
        )

    validate_submission(source)
    shutil.copyfile(source, BASELINE_PATH)
    baseline = read_submission(BASELINE_PATH)
    totals = {
        "purchase": int(baseline["purchase"].sum()),
        "redeem": int(baseline["redeem"].sum()),
    }
    BASELINE_TOTALS_PATH.write_text(json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8")
    (BASELINE_DIR / "baseline_131_summary.md").write_text(
        "# Baseline 131 Protected Copy\n\n"
        f"Initialized from `{source}`.\n\n"
        "- Candidate scripts must not overwrite this file.\n"
        "- Use this as the stable 131 score anchor for all probes.\n",
        encoding="utf-8",
    )


def load_baseline() -> pd.DataFrame:
    ensure_baseline_131()
    df = read_submission(BASELINE_PATH)
    df["date"] = pd.to_datetime(df["report_date"].astype(str), format="%Y%m%d")
    return df[["report_date", "date", "purchase", "redeem"]].copy()


def round_to_total(values: pd.Series | np.ndarray, target_total: int) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0, None)
    floors = np.floor(values).astype("int64")
    diff = int(target_total - floors.sum())
    if diff > 0:
        order = np.argsort(-(values - floors))
        floors[order[:diff]] += 1
    elif diff < 0:
        order = np.argsort(values - floors)
        for idx in order:
            if diff == 0:
                break
            take = min(int(floors[idx]), -diff)
            floors[idx] -= take
            diff += take
    return floors


def finalize_candidate(df: pd.DataFrame, totals: dict[str, int | None]) -> pd.DataFrame:
    out = df.copy()
    for col in ["purchase", "redeem"]:
        target_total = totals.get(col)
        if target_total is None:
            out[col] = np.rint(np.clip(out[col].astype(float), 0, None)).astype("int64")
        else:
            out[col] = round_to_total(out[col], int(target_total))
    out["report_date"] = out["report_date"].astype("int64")
    return out[["report_date", "purchase", "redeem"]]


def _compensate_to_original_total(
    original: pd.DataFrame,
    modified: pd.DataFrame,
    col: str,
    locked_mask: pd.Series,
) -> pd.DataFrame:
    out = modified.copy()
    original_total = float(original[col].sum())
    current_total = float(out[col].sum())
    delta = current_total - original_total
    if abs(delta) < 1e-6:
        return out

    pool_mask = ~locked_mask
    pool_sum = float(out.loc[pool_mask, col].sum())
    if pool_sum <= 0:
        return out
    out.loc[pool_mask, col] = out.loc[pool_mask, col] - delta * out.loc[pool_mask, col] / pool_sum
    out[col] = out[col].clip(lower=0)
    return out


def scale_total(base: pd.DataFrame, col: str, factor: float) -> pd.DataFrame:
    out = base.copy()
    out[col] = out[col].astype(float)
    out[col] = out[col].astype(float) * factor
    return out


def window_deviation(base: pd.DataFrame, col: str, dates: list[str], strength: float) -> pd.DataFrame:
    out = base.copy()
    out[col] = out[col].astype(float)
    mask = out["date"].dt.strftime("%Y-%m-%d").isin(dates)
    month_mean = float(out.loc[~mask, col].mean())
    out.loc[mask, col] = month_mean + (out.loc[mask, col].astype(float) - month_mean) * (1.0 + strength)
    return _compensate_to_original_total(base, out, col, mask)


def window_smooth(base: pd.DataFrame, col: str, dates: list[str], strength: float) -> pd.DataFrame:
    out = base.copy()
    out[col] = out[col].astype(float)
    mask = out["date"].dt.strftime("%Y-%m-%d").isin(dates)
    month_mean = float(out.loc[~mask, col].mean())
    out.loc[mask, col] = month_mean + (out.loc[mask, col].astype(float) - month_mean) * (1.0 - strength)
    return _compensate_to_original_total(base, out, col, mask)


def single_day_factor(base: pd.DataFrame, cols: list[str], date: str, factor: float) -> pd.DataFrame:
    out = base.copy()
    mask = out["date"].dt.strftime("%Y-%m-%d").eq(date)
    for col in cols:
        out[col] = out[col].astype(float)
        out.loc[mask, col] = out.loc[mask, col].astype(float) * factor
        out = _compensate_to_original_total(base, out, col, mask)
    return out


def date_window_factor(base: pd.DataFrame, col: str, dates: list[str], factor: float) -> pd.DataFrame:
    out = base.copy()
    out[col] = out[col].astype(float)
    mask = out["date"].dt.strftime("%Y-%m-%d").isin(dates)
    out.loc[mask, col] = out.loc[mask, col].astype(float) * factor
    return _compensate_to_original_total(base, out, col, mask)


def shift_between_windows(
    base: pd.DataFrame,
    col: str,
    source_dates: list[str],
    target_dates: list[str],
    source_rate: float,
) -> pd.DataFrame:
    out = base.copy()
    out[col] = out[col].astype(float)
    source_mask = out["date"].dt.strftime("%Y-%m-%d").isin(source_dates)
    target_mask = out["date"].dt.strftime("%Y-%m-%d").isin(target_dates)
    moved = out.loc[source_mask, col].astype(float) * source_rate
    amount = float(moved.sum())
    out.loc[source_mask, col] = out.loc[source_mask, col].astype(float) - moved
    target_sum = float(out.loc[target_mask, col].sum())
    if target_sum > 0:
        out.loc[target_mask, col] = out.loc[target_mask, col].astype(float) + amount * out.loc[target_mask, col] / target_sum
    return out


def blend_national_pre_2013_shape(
    base: pd.DataFrame,
    blend_weight: float,
    redeem_total_factor: float,
) -> pd.DataFrame:
    """Use 2013-09-25..30 as the historical National-Day pre-holiday shape."""
    raw_path = find_user_balance_file(RAW_DATA_DIR)
    if raw_path is None:
        raise FileNotFoundError("Missing raw user_balance data for historical bigfeast shape.")

    df, _ = read_csv_safely(raw_path, usecols=["report_date", "total_redeem_amt"])
    daily = (
        df.groupby("report_date", as_index=False)
        .agg(redeem=("total_redeem_amt", "sum"))
    )
    daily["date"] = pd.to_datetime(daily["report_date"].astype(str), format="%Y%m%d")
    hist = daily[(daily["date"] >= "2013-09-25") & (daily["date"] <= "2013-09-30")].copy()
    if len(hist) != 6 or hist["redeem"].sum() <= 0:
        raise ValueError("Cannot build 2013 National pre-holiday shape.")

    hist["day"] = hist["date"].dt.day
    hist_share = hist.set_index("day")["redeem"] / hist["redeem"].sum()

    out = base.copy()
    out["redeem"] = out["redeem"].astype(float)
    mask = out["date"].dt.strftime("%Y-%m-%d").between("2014-09-25", "2014-09-30")
    window = out.loc[mask, ["date", "redeem"]].copy()
    window["day"] = window["date"].dt.day
    window_total = float(window["redeem"].sum())
    target = window["day"].map(hist_share).astype(float).to_numpy() * window_total
    current = window["redeem"].astype(float).to_numpy()
    out.loc[mask, "redeem"] = (1.0 - blend_weight) * current + blend_weight * target
    out["redeem"] = out["redeem"] * redeem_total_factor
    return out


def linear_date_blend(base: pd.DataFrame, weight: float) -> pd.DataFrame:
    from simple_linear_date_model import predict_simple_linear_date_model

    totals = {"purchase": int(base["purchase"].sum()), "redeem": int(base["redeem"].sum())}
    weak = predict_simple_linear_date_model(normalize_totals=totals)
    out = base[["report_date", "date", "purchase", "redeem"]].merge(
        weak, on="report_date", suffixes=("", "_weak")
    )
    for col in ["purchase", "redeem"]:
        out[col] = (1.0 - weight) * out[col].astype(float) + weight * out[f"{col}_weak"].astype(float)
    return out[["report_date", "date", "purchase", "redeem"]]


def purchase_decomposition_blend(base: pd.DataFrame, weight: float) -> pd.DataFrame:
    """Blend purchase shape from direct_purchase + smoothed share_amt experiment."""
    raw_path = find_user_balance_file(RAW_DATA_DIR)
    if raw_path is None:
        return base.copy()

    usecols = ["report_date", "direct_purchase_amt", "share_amt"]
    df, _ = read_csv_safely(raw_path, usecols=usecols)
    daily = (
        df.groupby("report_date", as_index=False)
        .agg(
            direct_purchase_amt=("direct_purchase_amt", "sum"),
            share_amt=("share_amt", "sum"),
        )
    )
    daily["date"] = pd.to_datetime(daily["report_date"].astype(str), format="%Y%m%d")
    daily = daily[daily["date"] <= pd.Timestamp("2014-08-31")].sort_values("date")
    daily["weekday"] = daily["date"].dt.weekday

    future_dates = pd.date_range(PREDICT_START_DATE, PREDICT_END_DATE, freq="D")
    direct_values: list[float] = []
    for dt in future_dates:
        subset = daily[daily["weekday"] == dt.weekday()].tail(8)
        if subset.empty:
            subset = daily.tail(30)
        weights = np.linspace(0.75, 1.25, len(subset))
        direct_values.append(float(np.average(subset["direct_purchase_amt"], weights=weights)))

    share_anchor = float(daily["share_amt"].tail(45).ewm(span=12, adjust=False).mean().iloc[-1])
    decomp_purchase = np.asarray(direct_values, dtype=float) + share_anchor
    target_total = int(base["purchase"].sum())
    decomp_purchase = decomp_purchase / decomp_purchase.sum() * target_total

    out = base.copy()
    out["purchase"] = (1.0 - weight) * out["purchase"].astype(float) + weight * decomp_purchase
    out["purchase"] = round_to_total(out["purchase"], target_total)
    return out


def aggressive_composite_blend(base: pd.DataFrame) -> pd.DataFrame:
    out = purchase_decomposition_blend(base, weight=0.22)
    out = linear_date_blend(out, weight=0.08)
    out = single_day_factor(out, ["purchase", "redeem"], "2014-09-09", 1.035)
    return out


def save_candidate(
    candidate_dir: Path,
    base: pd.DataFrame,
    raw: pd.DataFrame,
    meta: CandidateMeta,
    totals: dict[str, int | None],
) -> dict[str, object]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    final = finalize_candidate(raw, totals)
    path = candidate_dir / meta.filename
    final.to_csv(path, index=False, header=False)
    validate_submission(path)

    merged = final.merge(base[["report_date", "purchase", "redeem"]], on="report_date", suffixes=("", "_base"))
    purchase_delta = int(final["purchase"].sum() - base["purchase"].sum())
    redeem_delta = int(final["redeem"].sum() - base["redeem"].sum())
    return {
        "priority": meta.priority,
        "file": str(path),
        "hypothesis": meta.hypothesis,
        "target": meta.target,
        "dates": meta.dates,
        "total_policy": meta.total_policy,
        "purchase_sum": int(final["purchase"].sum()),
        "redeem_sum": int(final["redeem"].sum()),
        "purchase_delta_vs_131": purchase_delta,
        "redeem_delta_vs_131": redeem_delta,
        "max_abs_daily_purchase_delta": int((merged["purchase"] - merged["purchase_base"]).abs().max()),
        "max_abs_daily_redeem_delta": int((merged["redeem"] - merged["redeem_base"]).abs().max()),
        "risk": meta.risk,
        "recommended_action": meta.recommended_action,
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_candidates(output_dir: Path = DEFAULT_CANDIDATE_DIR) -> pd.DataFrame:
    base = load_baseline()
    base_totals = {"purchase": int(base["purchase"].sum()), "redeem": int(base["redeem"].sum())}
    rows: list[dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in list(output_dir.glob("*.csv")) + list(output_dir.glob("*.md")):
        old_path.unlink()

    specs: list[tuple[CandidateMeta, pd.DataFrame, dict[str, int | None]]] = [
        (
            CandidateMeta(
                "04_bigfeast_2013_shape_w18_plus_0_6pct.csv",
                1,
                "Use 2013 National pre-holiday redeem distribution as bigfeast shape, with a moderate 18% blend and the proven +0.6% total lift.",
                "redeem",
                "20140925-20140930",
                "redeem +0.6%; 20140925-30 shape blended 18% toward 2013 National pre-holiday distribution; purchase unchanged",
                "high",
                "First submit; safest continuation of the 133-point early-shift direction.",
            ),
            blend_national_pre_2013_shape(base, blend_weight=0.18, redeem_total_factor=1.006),
            {"purchase": base_totals["purchase"], "redeem": None},
        ),
        (
            CandidateMeta(
                "05_bigfeast_2013_shape_w26_plus_0_6pct.csv",
                2,
                "More aggressive bigfeast table-style adjustment: stronger 2013-shape pull while keeping the same +0.6% redeem total lift.",
                "redeem",
                "20140925-20140930",
                "redeem +0.6%; 20140925-30 shape blended 26% toward 2013 National pre-holiday distribution; purchase unchanged",
                "high",
                "Second submit; tests whether the 133-point signal wants a stronger f_bigfeast date table.",
            ),
            blend_national_pre_2013_shape(base, blend_weight=0.26, redeem_total_factor=1.006),
            {"purchase": base_totals["purchase"], "redeem": None},
        ),
    ]

    for meta, candidate, totals in specs:
        rows.append(save_candidate(output_dir, base, candidate, meta, totals))

    summary = pd.DataFrame(rows).sort_values("priority")
    summary_path = output_dir / "candidates_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    (output_dir / "candidates_summary.md").write_text(dataframe_to_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reversible probes from protected 131 baseline.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    args = parser.parse_args()

    ensure_baseline_131()
    summary = build_candidates(args.output_dir)
    print(f"Protected baseline: {BASELINE_PATH}")
    print(f"Generated {len(summary)} candidates under: {args.output_dir}")
    print(f"Summary: {args.output_dir / 'candidates_summary.csv'}")
    print("No candidate was copied to output/tc_comp_predict_table.csv.")


if __name__ == "__main__":
    main()
