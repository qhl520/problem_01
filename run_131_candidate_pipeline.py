from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _run(command: list[str]) -> None:
    print(f"\n>>> {' '.join(command)}")
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def main() -> None:
    _run([sys.executable, "src/generate_131_candidates.py"])
    _run([sys.executable, "src/final_check.py"])
    print("\nCandidate pipeline completed.")
    print(f"Protected baseline: {PROJECT_ROOT / 'output' / 'baseline_131' / 'tc_comp_predict_table.csv'}")
    print(f"Candidate summary: {PROJECT_ROOT / 'output' / 'candidates_131' / 'candidates_summary.csv'}")
    print("Final submission was not overwritten by candidate generation.")


if __name__ == "__main__":
    main()
