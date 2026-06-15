# Source Integrity Check

## Core Scripts

| file | exists |
| --- | --- |
| run_all.py | True |
| run_original_plus_121.py | True |
| src/final_check.py | True |
| src/check_official_requirements.py | True |
| src/generate_121_submissions.py | True |
| src/config.py | True |
| src/data_utils.py | True |
| src/predict_final.py | True |
| src/train_final.py | True |

## Reproduction Status

| check | status |
| --- | --- |
| original pipeline exists | pass |
| original + 121 pipeline exists | pass |
| 121 generator exists | pass |
| default final equals 121 main | pass |
| Python cache removed | pass |

## Optional Historical High-Score Scripts

The previous broad search and candidate-selection scripts are no longer required. Their winning logic has been consolidated into `src/generate_121_submissions.py`, which preserves only the original baseline plus the two retained 121-point outputs.
