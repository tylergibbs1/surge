"""Availability-aware rolling conformal calibration for probabilistic forecasts.

The calibrator only uses errors from forecast origins whose complete target
window has matured before the forecast it adjusts. It can calibrate each series
independently or pool scale-normalized scores across related series while
preserving a separate adjustment per lead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from surge.verification import interval_score_80, weighted_interval_score

NORMALIZATION_EPSILON = 1e-6


@dataclass(frozen=True)
class CalibratedIntervals:
    lower: np.ndarray
    upper: np.ndarray
    adjustment: np.ndarray
    eligible: np.ndarray


def finite_sample_quantile(
    scores: np.ndarray,
    *,
    coverage: float,
    calibration_units: int | None = None,
) -> float:
    """Return a conservative empirical quantile.

    ``calibration_units`` is normally the number of scores. For cross-series
    pooling it is the number of complete origin blocks, which applies the
    finite-sample correction at the temporal unit instead of pretending that
    simultaneous, correlated RTO errors are independent observations.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("at least one finite conformity score is required")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be in (0, 1)")
    units = len(values) if calibration_units is None else calibration_units
    if units < 1 or units > len(values):
        raise ValueError("calibration_units must be between 1 and the score count")
    unit_rank = min(math.ceil((units + 1) * coverage), units)
    rank = min(
        (len(values) * unit_rank + units - 1) // units,
        len(values),
    )
    return float(np.partition(values, rank - 1)[rank - 1])


def cqr_scores(
    lower: np.ndarray,
    upper: np.ndarray,
    truth: np.ndarray,
    *,
    normalized: bool,
    epsilon: float = NORMALIZATION_EPSILON,
) -> np.ndarray:
    """Conformalized-quantile-regression scores; negative means safely inside."""
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    actual = np.asarray(truth, dtype=np.float64)
    if low.shape != high.shape or low.shape != actual.shape:
        raise ValueError("lower, upper, and truth shapes must match")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("forecast bounds must be finite")
    if np.any(high < low):
        raise ValueError("upper must not be below lower")
    scores = np.maximum(low - actual, actual - high)
    if normalized:
        scores = scores / np.maximum(high - low, epsilon)
    return scores


def rolling_conformalize(
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
    truth: np.ndarray,
    *,
    origin_times_utc: np.ndarray,
    outcome_delay_hours: int,
    window: int,
    min_history: int,
    coverage: float = 0.8,
    pooled_series: bool = False,
    normalized: bool = True,
) -> CalibratedIntervals:
    """Calibrate ``(origin, series, horizon)`` arrays sequentially.

    A prior origin becomes usable only once its last hourly target plus
    ``outcome_delay_hours`` is no later than the current origin. Windows count
    the most recent *mature origins*, not raw score cells. Pooled calibration
    uses complete cross-series origin blocks so every origin has equal weight.
    """
    low = np.asarray(lower, dtype=np.float64)
    mid = np.asarray(median, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    actual = np.asarray(truth, dtype=np.float64)
    if not (low.shape == mid.shape == high.shape == actual.shape):
        raise ValueError("forecast and truth shapes must match")
    if low.ndim != 3:
        raise ValueError("inputs must have shape (origin, series, horizon)")
    if any(size < 1 for size in low.shape):
        raise ValueError("origin, series, and horizon dimensions must be nonempty")
    if window < 1 or min_history < 1 or min_history > window:
        raise ValueError("require 1 <= min_history <= window")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be in (0, 1)")
    if outcome_delay_hours < 0:
        raise ValueError("outcome_delay_hours must be nonnegative")
    if pooled_series and not normalized:
        raise ValueError("cross-series pooling requires normalized scores")
    if not np.isfinite(low).all() or not np.isfinite(mid).all() or not np.isfinite(high).all():
        raise ValueError("forecast quantiles must be finite")
    if np.any((low > mid) | (mid > high)):
        raise ValueError("input quantiles must satisfy lower <= median <= upper")

    origin_times = np.asarray(origin_times_utc)
    if origin_times.shape != (low.shape[0],):
        raise ValueError("origin_times_utc must have one timestamp per origin")
    if not np.issubdtype(origin_times.dtype, np.datetime64):
        raise ValueError("origin_times_utc must contain numpy datetime64 values")
    origin_times = origin_times.astype("datetime64[us]")
    if np.isnat(origin_times).any():
        raise ValueError("origin_times_utc must not contain NaT")
    if len(origin_times) > 1 and np.any(np.diff(origin_times) <= np.timedelta64(0, "us")):
        raise ValueError("origin_times_utc must be strictly increasing")

    historical_scores = cqr_scores(low, high, actual, normalized=normalized)
    widths = (
        np.maximum(high - low, NORMALIZATION_EPSILON)
        if normalized
        else np.ones_like(low)
    )
    calibrated_low = np.full_like(low, np.nan)
    calibrated_high = np.full_like(high, np.nan)
    adjustments = np.full_like(low, np.nan)
    eligible = np.zeros_like(low, dtype=bool)
    _, series_count, horizon_count = low.shape

    maturity_offset = np.timedelta64(
        (horizon_count - 1) + outcome_delay_hours,
        "h",
    )
    available_at = origin_times + maturity_offset

    for origin in range(1, low.shape[0]):
        mature_origins = np.flatnonzero(available_at[:origin] <= origin_times[origin])
        mature_origins = mature_origins[-window:]
        for series in range(series_count):
            for horizon in range(horizon_count):
                if pooled_series:
                    score_block = historical_scores[mature_origins, :, horizon]
                    complete_origins = np.all(np.isfinite(score_block), axis=1)
                    score_block = score_block[complete_origins]
                    history_origins = len(score_block)
                    scores = score_block.reshape(-1)
                else:
                    scores = historical_scores[mature_origins, series, horizon]
                    history_origins = int(np.isfinite(scores).sum())
                scores = scores[np.isfinite(scores)]
                if history_origins < min_history:
                    continue
                adjustment = max(
                    finite_sample_quantile(
                        scores,
                        coverage=coverage,
                        calibration_units=history_origins,
                    ),
                    0.0,
                )
                delta = adjustment * widths[origin, series, horizon]
                # Negative CQR quantiles do not shrink a served interval. This
                # conservative policy keeps p50 enclosed and cannot reduce
                # empirical coverage relative to the uncalibrated model.
                calibrated_low[origin, series, horizon] = low[origin, series, horizon] - delta
                calibrated_high[origin, series, horizon] = high[origin, series, horizon] + delta
                adjustments[origin, series, horizon] = adjustment
                eligible[origin, series, horizon] = True

    return CalibratedIntervals(
        lower=calibrated_low,
        upper=calibrated_high,
        adjustment=adjustments,
        eligible=eligible,
    )


def interval_metrics(
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
    truth: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Score p10/p50/p90 with the ledger's exact 80% interval and WIS helpers."""
    low = np.asarray(lower, dtype=np.float64)
    mid = np.asarray(median, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    actual = np.asarray(truth, dtype=np.float64)
    if not (low.shape == mid.shape == high.shape == actual.shape):
        raise ValueError("forecast and truth shapes must match")
    if mask is not None and np.asarray(mask).shape != low.shape:
        raise ValueError("mask shape must match forecast shape")
    forecast_finite = np.isfinite(low) & np.isfinite(mid) & np.isfinite(high)
    if np.any(forecast_finite & ((low > mid) | (mid > high))):
        raise ValueError("forecast quantiles must satisfy lower <= median <= upper")
    selected = forecast_finite & np.isfinite(actual)
    if mask is not None:
        selected &= np.asarray(mask, dtype=bool)
    if not np.any(selected):
        raise ValueError("no finite points selected")
    low = low[selected]
    mid = mid[selected]
    high = high[selected]
    actual = actual[selected]
    interval_score = np.fromiter(
        (interval_score_80(a, lo, hi) for a, lo, hi in zip(actual, low, high, strict=True)),
        dtype=np.float64,
        count=len(actual),
    )
    wis = np.fromiter(
        (
            weighted_interval_score(a, lo, med, hi)
            for a, lo, med, hi in zip(actual, low, mid, high, strict=True)
        ),
        dtype=np.float64,
        count=len(actual),
    )
    return {
        "n_points": len(actual),
        "coverage": float(np.mean((actual >= low) & (actual <= high))),
        "mean_width": float(np.mean(high - low)),
        "interval_score": float(np.mean(interval_score)),
        "wis": float(np.mean(wis)),
    }
