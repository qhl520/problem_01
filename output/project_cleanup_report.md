# Project Cleanup Report

## Deleted Or Cleaned

| item | reason |
| --- | --- |
| output/high_score/ | obsolete broad candidate pool removed |
| output/models/ | model pkl files are regenerable and not part of final delivery |
| output/eda/ | EDA images are regenerable analysis artifacts |
| output/diagnostics/ | local check artifacts are regenerable and summarized in reports |
| output/logs/ | local run logs are not needed for final delivery |
| output/submissions/ | historical submission candidates removed |
| __pycache__/, *.pyc, *.pyo | Python cache removed |
| old output/*.csv/json/png artifacts | historical experiment outputs removed from working tree |

## Retained Core Files

| file | purpose |
| --- | --- |
| output/tc_comp_predict_table.csv | default final submission; equal to 121 main submission |
| output/original/tc_initial.csv | original initial baseline retained for comparison and reproduction |
| output/high_score_final/tc_comp_predict_table.csv | 121 main submission |
| output/high_score_final/tc_comp_predict_table_conservative.csv | 121 conservative backup submission |
| run_all.py | original full modeling pipeline |
| run_original_plus_121.py | original + 121 reproduction pipeline; cleans regenerable intermediate outputs by default |
| src/generate_121_submissions.py | minimal 121 generation logic |

## Do Not Commit To GitHub

- data/raw/
- output/models/
- output/eda/
- output/diagnostics/
- output/logs/
- output/submissions/
- .git/
- __pycache__/, *.pyc, *.pyo
