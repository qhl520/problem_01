from __future__ import annotations

import numpy as np

PURCHASE_WEIGHT = 0.45
REDEEM_WEIGHT = 0.55
ERROR_ZERO_SCORE = 10.0
ERROR_CUTOFF = 0.3


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1.0, np.abs(y_true))
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def daily_score(y_true, y_pred) -> float:
    """Approximate the official monotonic daily score on a 0-10 scale.

    The competition states that exact zero relative error scores 10 and
    relative error above 0.3 scores 0, but does not publish the full mapping.
    This local implementation uses a linear interpolation between those points
    only for offline model comparison.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1.0, np.abs(y_true))
    relative_error = np.abs(y_true - y_pred) / denom
    score = np.where(
        relative_error > ERROR_CUTOFF,
        0.0,
        ERROR_ZERO_SCORE * (1.0 - relative_error / ERROR_CUTOFF),
    )
    return float(np.mean(score))


def weighted_score(y_purchase_true, y_purchase_pred, y_redeem_true, y_redeem_pred) -> float:
    purchase_score = daily_score(y_purchase_true, y_purchase_pred)
    redeem_score = daily_score(y_redeem_true, y_redeem_pred)
    return float(PURCHASE_WEIGHT * purchase_score + REDEEM_WEIGHT * redeem_score)
