from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
EDA_OUTPUT_DIR = OUTPUT_DIR / "eda"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
LOG_DIR = OUTPUT_DIR / "logs"
DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostics"
REPORT_DIR = PROJECT_ROOT / "report"

TRAIN_END_DATE = "2014-08-31"
PREDICT_START_DATE = "2014-09-01"
PREDICT_END_DATE = "2014-09-30"

FINAL_SUBMISSION_PATH = OUTPUT_DIR / "tc_comp_predict_table.csv"


def ensure_directories() -> None:
    """Create only stable project directories.

    Runtime-specific directories are created by the scripts that actually
    write to them, which keeps a clean checkout from filling with empty dirs.
    """
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUT_DIR,
        REPORT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


ensure_directories()
