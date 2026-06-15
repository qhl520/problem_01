from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


COMMANDS = [
    [sys.executable, "src/check_data.py"],
    [sys.executable, "src/check_business_rules.py"],
    [sys.executable, "src/make_daily_data.py"],
    [sys.executable, "src/eda.py"],
    [sys.executable, "src/validate_rule_baseline.py"],
    [sys.executable, "src/test_features.py"],
    [sys.executable, "src/leakage_check.py"],
    [sys.executable, "src/outlier_analysis.py"],
    [sys.executable, "src/cross_validate.py", "--model", "lightgbm"],
    [sys.executable, "src/cross_validate.py", "--model", "random_forest"],
    [sys.executable, "src/cross_validate.py", "--model", "extra_trees"],
    [sys.executable, "src/search_ensemble_weights.py", "--step", "0.05"],
    [sys.executable, "src/train_final.py"],
    [sys.executable, "src/predict_final.py"],
    [sys.executable, "src/final_check.py"],
    [sys.executable, "src/check_official_requirements.py"],
]


def main() -> None:
    for command in COMMANDS:
        print(f"\n>>> {' '.join(command)}")
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise SystemExit(f"Command failed with exit code {result.returncode}: {' '.join(command)}")
    print("\nPipeline completed.")
    print(f"Final submission: {PROJECT_ROOT / 'output' / 'tc_comp_predict_table.csv'}")


if __name__ == "__main__":
    main()
