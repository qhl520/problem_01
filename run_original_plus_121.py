from __future__ import annotations

import shutil
import subprocess
import sys
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FINAL_SUBMISSION = PROJECT_ROOT / "output" / "tc_comp_predict_table.csv"
INITIAL_SUBMISSION = PROJECT_ROOT / "output" / "original" / "tc_initial.csv"
INTERMEDIATE_DIRS = [
    PROJECT_ROOT / "output" / "models",
    PROJECT_ROOT / "output" / "eda",
    PROJECT_ROOT / "output" / "diagnostics",
    PROJECT_ROOT / "output" / "logs",
    PROJECT_ROOT / "output" / "submissions",
]


def _run(command: list[str]) -> None:
    print(f"\n>>> {' '.join(command)}")
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def _clean_intermediate_outputs() -> None:
    for path in INTERMEDIATE_DIRS:
        if path.exists():
            shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep regenerable model, EDA, diagnostics, log, and candidate output directories.",
    )
    args = parser.parse_args()

    _run([sys.executable, "run_all.py"])
    INITIAL_SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FINAL_SUBMISSION, INITIAL_SUBMISSION)
    _run([sys.executable, "src/generate_121_submissions.py", "--initial", str(INITIAL_SUBMISSION)])
    if not args.keep_intermediate:
        _clean_intermediate_outputs()
    print("\nOriginal + 121 pipeline completed.")
    print(f"Original model submission: {INITIAL_SUBMISSION}")
    print(f"121 main submission: {PROJECT_ROOT / 'output' / 'high_score_final' / 'tc_comp_predict_table.csv'}")
    print(f"121 conservative submission: {PROJECT_ROOT / 'output' / 'high_score_final' / 'tc_comp_predict_table_conservative.csv'}")


if __name__ == "__main__":
    main()
