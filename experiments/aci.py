"""Adaptive conformal inference over the shared CQR score plumbing.

``rolling_conformalize`` applies one fixed quantile level to every RTO. The
v0.2 validation run showed that a single shared level cannot hold all seven
RTOs inside a two-point coverage band: ISNE and SWPP systematically over-cover
while PJM and ERCO under-cover.

Adaptive conformal inference learns the miscoverage level online instead:

    alpha <- alpha + gamma * (alpha_target - 1{y not in C})

Two deliberate departures from the literature:

- Bias-corrected ACI additionally recenters intervals on an EWMA bias estimate.
  Surge's calibration policy forbids moving an interval off p50, so only the
  width-adaptive half is implemented here.
- Feedback is delayed. A forecast origin may only update ``alpha`` once its
  final target hour plus the outcome delay has matured, matching
  ``eia-latest-at-plus72h-v1``. Updates are therefore applied in batches at the
  origin where they first become observable, not one per step.

``alpha_scope`` selects how much feedback each level receives. ``"per-lead"``
keeps one level per (series, lead) as in the multi-step ACI literature;
``"per-series"`` shares one level across a series' leads, so it observes every
lead's outcome and adapts far faster on a one-year validation cohort.
"""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np

from experiments.conformal import (
    NORMALIZATION_EPSILON,
    CalibratedIntervals,
    cqr_scores,
    finite_sample_quantile,
)

AlphaScope = Literal["per-lead", "per-series"]
ALPHA_SCOPES: tuple[AlphaScope, ...] = get_args(AlphaScope)

# A learned level must stay a usable two-sided interval level: never so small
# that the calibrated set degenerates, never past the median.
ALPHA_FLOOR = 0.001
ALPHA_CEILING = 0.5
EFFECTIVE_COVERAGE_FLOOR = 0.5
EFFECTIVE_COVERAGE_CEILING = 0.999


def aci_conformalize(
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
    gamma: float = 0.01,
    alpha_scope: AlphaScope = "per-series",
    normalized: bool = True,
) -> CalibratedIntervals:
    """Calibrate ``(origin, series, horizon)`` arrays with an adaptive level.

    Maturity, windowing, score normalization and the no-shrink guarantee match
    ``rolling_conformalize`` exactly. The only difference is that the quantile
    level adapts per series instead of staying at ``coverage``.
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
    if not 0 < gamma < 1:
        raise ValueError("gamma must be in (0, 1)")
    if alpha_scope not in ALPHA_SCOPES:
        raise ValueError(f"alpha_scope must be one of {ALPHA_SCOPES}")
    if outcome_delay_hours < 0:
        raise ValueError("outcome_delay_hours must be nonnegative")
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
        np.maximum(high - low, NORMALIZATION_EPSILON) if normalized else np.ones_like(low)
    )
    calibrated_low = np.full_like(low, np.nan)
    calibrated_high = np.full_like(high, np.nan)
    adjustments = np.full_like(low, np.nan)
    eligible = np.zeros_like(low, dtype=bool)
    origin_count, series_count, horizon_count = low.shape

    alpha_target = 1.0 - coverage
    alpha = np.full((series_count, horizon_count), alpha_target, dtype=np.float64)

    maturity_offset = np.timedelta64((horizon_count - 1) + outcome_delay_hours, "h")
    available_at = origin_times + maturity_offset

    # Miscoverage of the interval actually issued at each origin. NaN until that
    # origin is calibrated; only matured entries are ever fed back into alpha.
    miscovered = np.full(low.shape, np.nan)
    fed_back = np.zeros(origin_count, dtype=bool)

    for origin in range(1, origin_count):
        now = origin_times[origin]
        _feed_back_matured(
            alpha,
            miscovered=miscovered,
            fed_back=fed_back,
            available_at=available_at,
            now=now,
            origin=origin,
            gamma=gamma,
            alpha_target=alpha_target,
            alpha_scope=alpha_scope,
        )

        mature_origins = np.flatnonzero(available_at[:origin] <= now)[-window:]
        for series in range(series_count):
            for horizon in range(horizon_count):
                scores = historical_scores[mature_origins, series, horizon]
                history_origins = int(np.isfinite(scores).sum())
                scores = scores[np.isfinite(scores)]
                if history_origins < min_history:
                    continue
                effective_coverage = float(
                    np.clip(
                        1.0 - alpha[series, horizon],
                        EFFECTIVE_COVERAGE_FLOOR,
                        EFFECTIVE_COVERAGE_CEILING,
                    )
                )
                adjustment = max(
                    finite_sample_quantile(scores, coverage=effective_coverage), 0.0
                )
                scaled = adjustment * widths[origin, series, horizon]
                calibrated_low[origin, series, horizon] = low[origin, series, horizon] - scaled
                calibrated_high[origin, series, horizon] = high[origin, series, horizon] + scaled
                adjustments[origin, series, horizon] = scaled
                eligible[origin, series, horizon] = True
                observed = actual[origin, series, horizon]
                if np.isfinite(observed):
                    miscovered[origin, series, horizon] = float(
                        observed < calibrated_low[origin, series, horizon]
                        or observed > calibrated_high[origin, series, horizon]
                    )

    return CalibratedIntervals(
        lower=calibrated_low,
        upper=calibrated_high,
        adjustment=adjustments,
        eligible=eligible,
    )


def _feed_back_matured(
    alpha: np.ndarray,
    *,
    miscovered: np.ndarray,
    fed_back: np.ndarray,
    available_at: np.ndarray,
    now: np.datetime64,
    origin: int,
    gamma: float,
    alpha_target: float,
    alpha_scope: AlphaScope,
) -> None:
    """Apply every origin whose outcome window matured since the last step."""
    newly_mature = np.flatnonzero((available_at[:origin] <= now) & (~fed_back[:origin]))
    for past in newly_mature:
        observed = miscovered[past]
        known = np.isfinite(observed)
        if known.any():
            if alpha_scope == "per-lead":
                alpha[known] += gamma * (alpha_target - observed[known])
            else:
                # One level per series. Every lead's outcome is feedback for it,
                # so average them to keep the step size comparable to per-lead.
                for series in range(alpha.shape[0]):
                    lead_mask = known[series]
                    if lead_mask.any():
                        miss_rate = float(observed[series][lead_mask].mean())
                        alpha[series, :] += gamma * (alpha_target - miss_rate)
        fed_back[past] = True
    np.clip(alpha, ALPHA_FLOOR, ALPHA_CEILING, out=alpha)
