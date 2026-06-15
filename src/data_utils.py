from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import PREDICT_END_DATE, PREDICT_START_DATE


CSV_ENCODINGS = ("utf-8", "utf-8-sig", "gbk", "gb18030")


@dataclass
class CsvInfo:
    path: Path
    file_type: str
    shape: tuple[int, int]
    columns: list[str]
    missing: dict[str, int]
    date_columns: list[str]
    date_ranges: dict[str, tuple[str | None, str | None]]
    head_markdown: str
    encoding: str


def dataframe_preview_markdown(df: pd.DataFrame, max_rows: int = 5) -> str:
    """Render a small markdown-like preview without optional tabulate dependency."""
    preview = df.head(max_rows).copy()
    if preview.empty:
        return "_Empty file_"
    headers = [str(col) for col in preview.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in preview.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in preview.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def read_csv_safely(path: Path, nrows: int | None = None, **kwargs) -> tuple[pd.DataFrame, str]:
    """Read a CSV using common encodings for Tianchi datasets."""
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, nrows=nrows, **kwargs), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path, nrows=nrows, **kwargs), "default"


def _looks_like_headerless_submission(df: pd.DataFrame) -> bool:
    if df.shape[1] != 3:
        return False
    try:
        first_col = [int(float(x)) for x in df.columns]
    except (TypeError, ValueError):
        return False
    return 20140901 <= first_col[0] <= 20140930


def scan_csv_files(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("*.csv"))


def normalize_columns(columns: Iterable[str]) -> set[str]:
    return {str(col).strip() for col in columns}


def identify_file_type(columns: Iterable[str]) -> str:
    cols = normalize_columns(columns)
    if {"total_purchase_amt", "total_redeem_amt", "report_date"}.issubset(cols):
        return "user_balance"
    if {"report_date", "purchase", "redeem"}.issubset(cols):
        return "submission_template"
    if "mfd_daily_yield" in cols:
        return "fund_yield"
    if "Interest_O_N" in cols or any(col.startswith("Interest_") for col in cols):
        return "shibor"
    if {"user_id", "sex"}.issubset(cols) or {"city", "constellation"}.issubset(cols):
        return "user_profile"
    return "unknown"


def likely_date_columns(df: pd.DataFrame) -> list[str]:
    names = []
    for col in df.columns:
        lowered = str(col).lower()
        if "date" in lowered or lowered in {"dt", "day"}:
            names.append(str(col))
    return names


def parse_competition_date(series: pd.Series) -> pd.Series:
    values = series.copy()
    if pd.api.types.is_numeric_dtype(values):
        values = values.astype("Int64").astype(str)
    else:
        values = values.astype(str).str.strip()
    return pd.to_datetime(values, format="%Y%m%d", errors="coerce")


def inspect_csv(path: Path) -> CsvInfo:
    df, encoding = read_csv_safely(path)
    if _looks_like_headerless_submission(df):
        df, encoding = read_csv_safely(path, header=None, names=["report_date", "purchase", "redeem"])
    date_cols = likely_date_columns(df)
    date_ranges: dict[str, tuple[str | None, str | None]] = {}
    for col in date_cols:
        parsed = parse_competition_date(df[col])
        valid = parsed.dropna()
        if valid.empty:
            date_ranges[col] = (None, None)
        else:
            date_ranges[col] = (
                valid.min().strftime("%Y-%m-%d"),
                valid.max().strftime("%Y-%m-%d"),
            )

    return CsvInfo(
        path=path,
        file_type=identify_file_type(df.columns),
        shape=df.shape,
        columns=[str(col) for col in df.columns],
        missing=df.isna().sum().astype(int).to_dict(),
        date_columns=date_cols,
        date_ranges=date_ranges,
        head_markdown=dataframe_preview_markdown(df),
        encoding=encoding,
    )


def find_user_balance_file(raw_dir: Path) -> Path | None:
    for path in scan_csv_files(raw_dir):
        df, _ = read_csv_safely(path, nrows=5)
        if identify_file_type(df.columns) == "user_balance":
            return path
    return None


def read_submission(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Submission file does not exist: {path}")

    df, _ = read_csv_safely(path, header=None, names=["report_date", "purchase", "redeem"])
    return df


def validate_submission(path: Path | str) -> None:
    path = Path(path)
    df = read_submission(path)
    expected_cols = ["report_date", "purchase", "redeem"]
    first_row = df.iloc[0].astype(str).str.strip().str.lower().tolist()
    if first_row == expected_cols:
        raise ValueError("Official submission must not contain a header row.")
    if len(df) != 30:
        raise ValueError(f"Submission must contain 30 rows, got {len(df)}")
    if df.isna().any().any():
        raise ValueError("Submission contains missing values")
    if df["report_date"].duplicated().any():
        raise ValueError("Submission contains duplicate report_date values")

    dates = pd.to_datetime(df["report_date"].astype(str), format="%Y%m%d", errors="coerce")
    expected_dates = pd.date_range(PREDICT_START_DATE, PREDICT_END_DATE, freq="D")
    if dates.isna().any():
        raise ValueError("Submission contains invalid report_date values")
    if not dates.is_monotonic_increasing:
        raise ValueError("Submission dates must be sorted ascending")
    if not dates.equals(pd.Series(expected_dates)):
        expected = [int(d.strftime("%Y%m%d")) for d in expected_dates]
        got = df["report_date"].tolist()
        raise ValueError(f"Submission dates are not complete. Expected {expected}, got {got}")

    for col in ["purchase", "redeem"]:
        if (df[col] < 0).any():
            raise ValueError(f"{col} contains negative values")
        if not pd.api.types.is_integer_dtype(df[col]):
            as_int = df[col].round().astype("int64")
            if not (df[col] == as_int).all():
                raise ValueError(f"{col} must contain integer amounts")
