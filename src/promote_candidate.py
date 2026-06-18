from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from config import FINAL_SUBMISSION_PATH, OUTPUT_DIR
from data_utils import validate_submission
from generate_131_candidates import BASELINE_PATH, ensure_baseline_131


def promote_candidate(candidate: Path) -> Path:
    ensure_baseline_131()
    candidate = candidate.resolve()
    validate_submission(candidate)
    validate_submission(BASELINE_PATH)

    backup_dir = OUTPUT_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if FINAL_SUBMISSION_PATH.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"tc_comp_predict_table_before_promote_{stamp}.csv"
        shutil.copyfile(FINAL_SUBMISSION_PATH, backup_path)
    else:
        backup_path = backup_dir / "no_previous_final.txt"
        backup_path.write_text("No previous final submission existed before promotion.\n", encoding="utf-8")

    shutil.copyfile(candidate, FINAL_SUBMISSION_PATH)
    validate_submission(FINAL_SUBMISSION_PATH)
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Manually promote a validated candidate to final submission.")
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    backup_path = promote_candidate(args.candidate)
    print(f"Promoted candidate: {args.candidate}")
    print(f"Final submission: {FINAL_SUBMISSION_PATH}")
    print(f"Previous final backup: {backup_path}")
    print(f"Protected baseline remains unchanged: {BASELINE_PATH}")


if __name__ == "__main__":
    main()
