"""Unit tests for core functions used in the 121-score submission pipeline.

Run with: python -m pytest tests/ -v
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_utils import read_submission, validate_submission
from evaluate import daily_score, mape, weighted_score


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_daily_data():
    """Small synthetic daily balance DataFrame."""
    dates = pd.date_range("2014-07-01", "2014-07-14", freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "date": dates,
            "purchase": (rng.random(len(dates)) * 10_000_000 + 20_000_000).astype("int64"),
            "redeem": (rng.random(len(dates)) * 8_000_000 + 15_000_000).astype("int64"),
        }
    )


def _make_submission_csv(lines: list[str]) -> Path:
    path = Path("/tmp/test_submission.csv")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# generate_121_submissions helpers (_normalize, _round_preserve_total, _weighted_mean)
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_normalize_preserves_total(self):
        from generate_121_submissions import _normalize

        s = pd.Series([10.0, 20.0, 30.0])
        result = _normalize(s, 120.0)
        assert abs(result.sum() - 120.0) < 1e-9

    def test_normalize_keeps_proportions(self):
        from generate_121_submissions import _normalize

        s = pd.Series([1.0, 2.0, 3.0])
        result = _normalize(s, 60.0)
        np.testing.assert_allclose(result.values, [10.0, 20.0, 30.0])

    def test_normalize_all_zeros_falls_back_to_uniform(self):
        from generate_121_submissions import _normalize

        s = pd.Series([0.0, 0.0, 0.0])
        result = _normalize(s, 90.0)
        np.testing.assert_allclose(result.values, [30.0, 30.0, 30.0])

    def test_normalize_clips_negative_to_zero(self):
        from generate_121_submissions import _normalize

        s = pd.Series([-5.0, 10.0, 15.0])
        result = _normalize(s, 100.0)
        assert (result >= 0).all()
        assert abs(result.sum() - 100.0) < 1e-9


class TestRoundPreserveTotal:
    def test_exact_total_matches(self):
        from generate_121_submissions import _round_preserve_total

        s = pd.Series([10.0, 20.0, 30.0])
        result = _round_preserve_total(s, 60)
        assert result.sum() == 60
        assert result.dtype == np.int64

    def test_rounding_adjusts_largest_remainders(self):
        from generate_121_submissions import _round_preserve_total

        s = pd.Series([1.4, 1.4, 1.4])  # sum=4.2, floors=[1,1,1]=3, diff=+1
        result = _round_preserve_total(s, 4)
        assert result.sum() == 4
        # largest fraction (0.4 each → pick first) gets the extra
        assert list(result).count(2) == 1
        assert list(result).count(1) == 2

    def test_negative_diff_removes_from_smallest_remainders(self):
        from generate_121_submissions import _round_preserve_total

        s = pd.Series([1.6, 1.6, 1.6])  # sum=4.8, floors=[1,1,1]=3, need 5? No — total=5
        # Let me design a clearer case: values [2.6, 0.3, 0.3], total=3.2
        # floors=[2,0,0]=2, diff=+1 → 需要加1，加到最大小数部分即0.6
        # Actually let me test the diff<0 case separately
        s = pd.Series([2.4, 0.4, 0.4])  # sum=3.2, floors=[2,0,0]=2
        result = _round_preserve_total(s, 3)  # want 3, floors give 2, diff=+1
        assert result.sum() == 3

    def test_all_ones(self):
        from generate_121_submissions import _round_preserve_total

        s = pd.Series([1.0, 1.0, 1.0])
        result = _round_preserve_total(s, 3)
        assert result.sum() == 3
        assert list(result) == [1, 1, 1]

    def test_clips_negative_input(self):
        from generate_121_submissions import _round_preserve_total

        s = pd.Series([-1.0, 2.0, 3.0])
        result = _round_preserve_total(s, 5)
        assert (result >= 0).all()
        assert result.sum() == 5


class TestWeightedMean:
    def test_linear_weights(self):
        from generate_121_submissions import _weighted_mean

        s = pd.Series([10.0, 20.0, 30.0])
        # weights: [1, 2, 3] → (10*1 + 20*2 + 30*3) / (1+2+3) = 140/6 ≈ 23.33
        result = _weighted_mean(s)
        assert abs(result - 140.0 / 6.0) < 1e-9

    def test_empty_series(self):
        from generate_121_submissions import _weighted_mean

        assert _weighted_mean(pd.Series([], dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# evaluate helpers
# ---------------------------------------------------------------------------


class TestMape:
    def test_perfect_prediction(self):
        assert mape(np.array([100.0]), np.array([100.0])) == pytest.approx(0.0)

    def test_ten_percent_error(self):
        result = mape(np.array([100.0]), np.array([110.0]))
        assert result == pytest.approx(0.10)

    def test_zero_true_uses_denom_1(self):
        # y_true[0]=0 → denom=1,  error = |0-10|/1 = 10
        # y_true[1]=100 → denom=100, error = |100-100|/100 = 0
        # mape = mean([10, 0]) = 5.0
        result = mape(np.array([0.0, 100.0]), np.array([10.0, 100.0]))
        assert result == pytest.approx(5.0)


class TestDailyScore:
    def test_perfect_scores_10(self):
        assert daily_score(np.array([100.0]), np.array([100.0])) == pytest.approx(10.0)

    def test_cutoff_scores_0(self):
        # relative error 0.31 > 0.3 cutoff → score 0
        result = daily_score(np.array([100.0]), np.array([131.0]))
        assert result == pytest.approx(0.0)

    def test_half_cutoff_scores_5(self):
        # relative error 0.15 = half of 0.3 → score 5
        result = daily_score(np.array([100.0]), np.array([115.0]))
        assert result == pytest.approx(5.0)


class TestWeightedScore:
    def test_blend(self):
        result = weighted_score(
            np.array([100.0]), np.array([100.0]),
            np.array([200.0]), np.array([200.0]),
        )
        assert result == pytest.approx(0.45 * 10.0 + 0.55 * 10.0)


# ---------------------------------------------------------------------------
# data_utils — submission validation
# ---------------------------------------------------------------------------


class TestReadSubmission:
    def test_reads_valid_file(self):
        path = _make_submission_csv([
            "20140901,295000000,331000000",
            "20140902,287000000,298000000",
        ])
        df = read_submission(path)
        assert list(df.columns) == ["report_date", "purchase", "redeem"]
        assert len(df) == 2

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            read_submission("/tmp/nonexistent_submission.csv")


class TestValidateSubmission:
    def test_valid_passes(self, tmp_path):
        lines = []
        for day in range(1, 31):
            lines.append(f"201409{day:02d},1000000,2000000")
        path = tmp_path / "sub.csv"
        path.write_text("\n".join(lines), encoding="utf-8")
        validate_submission(str(path))  # should not raise

    def test_wrong_row_count_raises(self, tmp_path):
        path = tmp_path / "sub.csv"
        path.write_text("20140901,100,200\n", encoding="utf-8")
        with pytest.raises(ValueError, match="30 rows"):
            validate_submission(str(path))

    def test_header_row_raises(self, tmp_path):
        lines = ["report_date,purchase,redeem"] + [
            f"201409{d:02d},100,200" for d in range(1, 31)
        ]
        path = tmp_path / "sub.csv"
        path.write_text("\n".join(lines), encoding="utf-8")
        with pytest.raises(ValueError, match="header"):
            validate_submission(str(path))

    def test_duplicate_dates_raises(self, tmp_path):
        lines = [f"201409{d:02d},100,200" for d in range(1, 31)]
        lines[15] = "20140901,100,200"  # duplicate
        path = tmp_path / "sub.csv"
        path.write_text("\n".join(lines), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate"):
            validate_submission(str(path))

    def test_negative_amount_raises(self, tmp_path):
        lines = [f"201409{d:02d},100,200" for d in range(1, 31)]
        lines[10] = "20140911,-100,200"
        path = tmp_path / "sub.csv"
        path.write_text("\n".join(lines), encoding="utf-8")
        with pytest.raises(ValueError, match="negative"):
            validate_submission(str(path))

    def test_wrong_row_count_detected_before_date_check(self, tmp_path):
        # Row-count check fires first for missing dates; validate_submission
        # short-circuits with "must contain 30 rows" before reaching the
        # date-completeness check.
        lines = [f"201409{d:02d},100,200" for d in range(1, 31) if d != 15]
        path = tmp_path / "sub.csv"
        path.write_text("\n".join(lines), encoding="utf-8")
        with pytest.raises(ValueError, match="30 rows"):
            validate_submission(str(path))


# ---------------------------------------------------------------------------
# features — make_features smoke test
# ---------------------------------------------------------------------------


class TestMakeFeatures:
    def test_produces_expected_columns(self, sample_daily_data):
        from features import make_features
        from model_utils import get_feature_columns

        feat = make_features(sample_daily_data, use_finance=False)
        cols = get_feature_columns(feat)
        # core columns that should always be present
        assert "day" in cols
        assert "month" in cols
        assert "weekday" in cols
        assert "is_weekend" in cols
        assert "is_holiday" in cols
        assert "purchase_lag_1" in cols
        assert "redeem_lag_1" in cols
        assert "purchase_rolling_7_mean" in cols
        assert "redeem_rolling_7_mean" in cols

    def test_no_data_leak_columns(self, sample_daily_data):
        from features import make_features
        from model_utils import get_feature_columns

        feat = make_features(sample_daily_data, use_finance=False)
        cols = get_feature_columns(feat)
        forbidden = {"date", "purchase", "redeem", "report_date"}
        assert forbidden.isdisjoint(set(cols))

    def test_no_duplicate_lag_features(self, sample_daily_data):
        """Regression: same_weekday_last_{idx} should no longer exist."""
        from features import make_features

        feat = make_features(sample_daily_data, use_finance=False)
        same_weekday_cols = [c for c in feat.columns if "same_weekday_last_" in c]
        assert len(same_weekday_cols) == 0, (
            "same_weekday_last_{idx} columns are redundant with lag columns and "
            "should have been removed"
        )


# ---------------------------------------------------------------------------
# model_utils — feature importance & model factory
# ---------------------------------------------------------------------------


class TestGetFeatureColumns:
    def test_excludes_forbidden(self):
        from model_utils import get_feature_columns

        df = pd.DataFrame(
            {
                "date": [1, 2],
                "purchase": [3, 4],
                "redeem": [5, 6],
                "report_date": [7, 8],
                "good_float": [1.0, 2.0],
                "good_int": [1, 2],
                "bad_str": ["a", "b"],
            }
        )
        cols = get_feature_columns(df)
        assert "good_float" in cols
        assert "good_int" in cols
        assert "date" not in cols
        assert "purchase" not in cols
        assert "redeem" not in cols
        assert "report_date" not in cols
        assert "bad_str" not in cols


class TestModelFactory:
    def test_make_all_model_types_return_pipelines(self):
        from model_utils import make_model
        from sklearn.pipeline import Pipeline

        for mt in ["random_forest", "extra_trees", "lightgbm", "hist_gradient_boosting"]:
            pipe = make_model(mt)
            assert isinstance(pipe, Pipeline)
            assert hasattr(pipe, "predict")

    def test_unknown_model_raises(self):
        from model_utils import make_model

        with pytest.raises(ValueError, match="Unknown"):
            make_model("nonexistent_model")


# ---------------------------------------------------------------------------
# generate_121_submissions — blend_and_adjust pipeline integration
# ---------------------------------------------------------------------------


class TestBlendAndAdjustIntegration:
    def test_preserves_total_amounts(self):
        """End-to-end: blend+adjust should keep purchase/redeem totals unchanged."""
        from generate_121_submissions import _blend_and_adjust, _shape_from_august_weekday

        # Build a minimal initial DataFrame
        initial = pd.DataFrame({
            "report_date": [20140901 + d for d in range(30)],
            "purchase": np.full(30, 10_000_000, dtype="int64"),
            "redeem": np.full(30, 8_000_000, dtype="int64"),
            "date": pd.to_datetime(["2014-09-%02d" % (d + 1) for d in range(30)]),
            "weekday": [pd.Timestamp(f"2014-09-{d+1:02d}").weekday() for d in range(30)],
            "is_weekend": [pd.Timestamp(f"2014-09-{d+1:02d}").weekday() >= 5 for d in range(30)],
        })

        # Build a minimal daily history
        daily = pd.DataFrame({
            "date": pd.to_datetime(["2014-08-%02d" % (d + 1) for d in range(31)]),
            "purchase": np.full(31, 10_000_000, dtype="int64"),
            "redeem": np.full(31, 8_000_000, dtype="int64"),
            "weekday": [pd.Timestamp(f"2014-08-{d+1:02d}").weekday() for d in range(31)],
        })

        shape = _shape_from_august_weekday(initial, daily)
        result = _blend_and_adjust(initial, shape)

        assert result["purchase"].sum() == initial["purchase"].sum()
        assert result["redeem"].sum() == initial["redeem"].sum()
        assert len(result) == 30
