from __future__ import annotations

from pathlib import Path

from config import OUTPUT_DIR, RAW_DATA_DIR, ensure_directories
from data_utils import inspect_csv, scan_csv_files


def render_report() -> str:
    ensure_directories()
    csv_files = scan_csv_files(RAW_DATA_DIR)
    lines: list[str] = [
        "# Data Schema Report",
        "",
        f"Raw data directory: `{RAW_DATA_DIR}`",
        f"CSV files found: {len(csv_files)}",
        "",
    ]

    if not csv_files:
        lines.extend(
            [
                "## Status",
                "",
                "No CSV files were found in `data/raw/`.",
                "",
                "Please log in to Tianchi, download the competition data, unzip it, and place the CSV files in `data/raw/`.",
                "Common files include `user_balance_table.csv`, `user_profile_table.csv`, `mfd_day_share_interest.csv`, `mfd_bank_shibor.csv`, and `tc_comp_predict_table.csv`.",
                "",
            ]
        )
        return "\n".join(lines)

    for path in csv_files:
        info = inspect_csv(path)
        print("=" * 80)
        print(f"File: {path.name}")
        print(f"Encoding: {info.encoding}")
        print(f"Type: {info.file_type}")
        print(f"Shape: {info.shape}")
        print(f"Columns: {info.columns}")
        print("Missing values:")
        for col, value in info.missing.items():
            print(f"  {col}: {value}")
        print("Date ranges:")
        for col, (start, end) in info.date_ranges.items():
            print(f"  {col}: {start} to {end}")
        print("Head:")
        print(info.head_markdown)

        lines.extend(
            [
                f"## {path.name}",
                "",
                f"- Encoding: `{info.encoding}`",
                f"- Inferred type: `{info.file_type}`",
                f"- Shape: `{info.shape[0]}` rows x `{info.shape[1]}` columns",
                f"- Columns: `{', '.join(info.columns)}`",
                "",
                "### Date Columns",
                "",
            ]
        )
        if info.date_ranges:
            for col, (start, end) in info.date_ranges.items():
                lines.append(f"- `{col}`: {start} to {end}")
        else:
            lines.append("- No likely date columns detected.")

        lines.extend(["", "### Missing Values", ""])
        for col, value in info.missing.items():
            lines.append(f"- `{col}`: {value}")
        lines.extend(["", "### Head", "", info.head_markdown, ""])

    return "\n".join(lines)


def main() -> None:
    report = render_report()
    output_path = OUTPUT_DIR / "data_schema_report.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"Data schema report written to: {output_path}")


if __name__ == "__main__":
    main()
