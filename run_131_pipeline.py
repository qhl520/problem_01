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
    _run([sys.executable, "src/optimize_ensemble_v2.py"])
    _run([sys.executable, "src/final_check.py"])
    print("\n131 pipeline completed.")
    print(f"Final submission: {PROJECT_ROOT / 'output' / 'tc_comp_predict_table.csv'}")
    print(f"Frozen 131 copy: {PROJECT_ROOT / 'output' / 'best_131' / 'tc_comp_predict_table.csv'}")


if __name__ == "__main__":
    main()
