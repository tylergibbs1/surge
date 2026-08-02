"""Conformal interval calibration for served forecasts.

The shipped model's own published evaluation reports 0.725 mean coverage on a
nominal 80% band. A band labelled 80% that is right 72.5% of the time is not a
small defect: it is the number the whole product rests on.

This applies the policy validated in ``docs/adaptive-conformal-experiment.md``
to live issuances. It holds every RTO within 1.6 points of nominal across 2021,
2022, 2023 and the untouched 2024 reporting half.

Three properties make it safe to serve:

- **Causal.** A past forecast may only inform calibration once its own outcome
  has matured under ``eia-latest-at-plus72h-v1``. Nothing here can see an
  outcome a live forecaster would not have had.
- **Fails open.** With too little matured history the forecast is served
  uncalibrated and says so, rather than serving a fabricated adjustment.
- **Declared.** Every issuance records the policy version, the adjustment
  applied per lead, and how many mature origins informed it, so a published
  interval can be audited back to the evidence that widened it.

The score is signed, as canonical CQR requires: negative means the interval was
comfortably right and should tighten. Clamping that branch away leaves a
one-way ratchet that pushes an already-calibrated series past nominal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

CALIBRATION_POLICY_VERSION = "surge-conformal-aci-v1"

TARGET_COVERAGE = 0.8
GAMMA = 0.05
WINDOW_MATURE_ORIGINS = 168
MIN_HISTORY_MATURE_ORIGINS = 28
#: Floor on calibrated width as a fraction of the model's own interval. Zero is
#: canonical CQR. Raise it to refuse to ship an interval narrower than the
#: model's, at the cost of over-covering already-calibrated series.
MIN_WIDTH_FRACTION = 0.0
ALPHA_FLOOR = 0.001
ALPHA_CEILING = 0.5
NORMALIZATION_EPSILON = 1e-6


@dataclass(frozen=True)
class MaturedOrigin:
    """One past issuance whose outcomes have all settled.

    ``lower``/``upper`` are the model's own interval, which is what the
    conformity score is normalized against. ``issued_lower``/``issued_upper``
    are the interval actually published after calibration, and are what the
    online level learns from -- the level is tracking how the *served* interval
    performed, not how the raw model performed. Confusing the two makes the
    level chase an error calibration already corrected, and it widens without
    bound.
    """

    origin_utc: datetime
    lower: np.ndarray
    upper: np.ndarray
    actual: np.ndarray
    issued_lower: np.ndarray | None = None
    issued_upper: np.ndarray | None = None

    @property
    def served_lower(self) -> np.ndarray:
        return self.lower if self.issued_lower is None else self.issued_lower

    @property
    def served_upper(self) -> np.ndarray:
        return self.upper if self.issued_upper is None else self.issued_upper


@dataclass(frozen=True)
class CalibrationState:
    """What calibration did, and on what evidence."""

    applied: bool
    policy_version: str = CALIBRATION_POLICY_VERSION
    mature_origins: int = 0
    alpha: float | None = None
    effective_coverage: float | None = None
    adjustment_mw: list[float] = field(default_factory=list)
    reason: str | None = None

    def as_provenance(self) -> dict[str, object]:
        return {
            "calibration_policy_version": self.policy_version,
            "calibration_applied": self.applied,
            "calibration_mature_origins": self.mature_origins,
            "calibration_alpha": self.alpha,
            "calibration_effective_coverage": self.effective_coverage,
            "calibration_reason": self.reason,
        }


def finite_sample_quantile(scores: np.ndarray, *, coverage: float) -> float:
    """Conservative empirical quantile with the finite-sample correction."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("at least one finite conformity score is required")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be in (0, 1)")
    rank = min(math.ceil((len(values) + 1) * coverage), len(values))
    return float(np.partition(values, rank - 1)[rank - 1])


def cqr_scores(lower: np.ndarray, upper: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Signed, width-normalized conformity scores. Negative means safely inside."""
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    truth = np.asarray(actual, dtype=np.float64)
    raw = np.maximum(low - truth, truth - high)
    width = np.maximum(high - low, NORMALIZATION_EPSILON)
    return raw / width


def mature_before(
    history: list[MaturedOrigin],
    *,
    now_utc: datetime,
    horizon: int,
    outcome_delay_hours: int,
) -> list[MaturedOrigin]:
    """Keep only origins whose final target plus the settlement delay has passed."""
    offset = timedelta(hours=(horizon - 1) + outcome_delay_hours)
    return [origin for origin in history if origin.origin_utc + offset <= now_utc]


def _replay_alpha(history: list[MaturedOrigin]) -> float:
    """Re-derive the online level from the outcomes, oldest first.

    One level per series, shared across its leads: on a horizon this short a
    per-lead level sees too few matured outcomes to converge.
    """
    alpha = 1.0 - TARGET_COVERAGE
    target = alpha
    for origin in history:
        finite = np.isfinite(origin.actual)
        if not finite.any():
            continue
        missed = (origin.actual < origin.served_lower) | (
            origin.actual > origin.served_upper
        )
        alpha += GAMMA * (target - float(missed[finite].mean()))
        alpha = float(np.clip(alpha, ALPHA_FLOOR, ALPHA_CEILING))
    return alpha


def calibrate(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    history: list[MaturedOrigin],
) -> tuple[np.ndarray, np.ndarray, CalibrationState]:
    """Adjust one issuance's interval from its own matured history.

    ``history`` must already be filtered to matured origins, oldest first.
    """
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if low.shape != high.shape or low.ndim != 1:
        raise ValueError("lower and upper must be one-dimensional and the same length")
    if np.any(high < low):
        raise ValueError("upper must not be below lower")

    if len(history) < MIN_HISTORY_MATURE_ORIGINS:
        return (
            low,
            high,
            CalibrationState(
                applied=False,
                mature_origins=len(history),
                reason=(
                    f"needs {MIN_HISTORY_MATURE_ORIGINS} matured origins, "
                    f"has {len(history)}"
                ),
            ),
        )

    window = history[-WINDOW_MATURE_ORIGINS:]
    alpha = _replay_alpha(window)
    effective = float(np.clip(1.0 - alpha, 0.5, 0.999))

    horizon = low.shape[0]
    adjusted_low = low.copy()
    adjusted_high = high.copy()
    adjustments: list[float] = []
    width = np.maximum(high - low, NORMALIZATION_EPSILON)
    for lead in range(horizon):
        scores = np.concatenate(
            [
                cqr_scores(
                    origin.lower[lead : lead + 1],
                    origin.upper[lead : lead + 1],
                    origin.actual[lead : lead + 1],
                )
                for origin in window
                if lead < origin.actual.shape[0]
            ]
        )
        scores = scores[np.isfinite(scores)]
        if not len(scores):
            adjustments.append(0.0)
            continue
        scaled = finite_sample_quantile(scores, coverage=effective) * width[lead]
        adjusted_low[lead] = low[lead] - scaled
        adjusted_high[lead] = high[lead] + scaled
        if MIN_WIDTH_FRACTION > 0.0:
            floor_width = MIN_WIDTH_FRACTION * (high[lead] - low[lead])
            if adjusted_high[lead] - adjusted_low[lead] < floor_width:
                centre = 0.5 * (adjusted_low[lead] + adjusted_high[lead])
                adjusted_low[lead] = centre - 0.5 * floor_width
                adjusted_high[lead] = centre + 0.5 * floor_width
        elif adjusted_high[lead] < adjusted_low[lead]:
            centre = 0.5 * (adjusted_low[lead] + adjusted_high[lead])
            adjusted_low[lead] = centre
            adjusted_high[lead] = centre
        adjustments.append(round(float(scaled), 4))

    return (
        adjusted_low,
        adjusted_high,
        CalibrationState(
            applied=True,
            mature_origins=len(window),
            alpha=round(alpha, 6),
            effective_coverage=round(effective, 6),
            adjustment_mw=adjustments,
        ),
    )


def matured_history(
    ba: str,
    *,
    now_utc: datetime,
    horizon: int,
    outcome_delay_hours: int,
    limit: int = WINDOW_MATURE_ORIGINS,
) -> list[MaturedOrigin]:
    """Read settled issuances for one BA from the ledger, oldest first.

    Joins the interval that was published with the outcome that settled it. An
    issuance is only eligible once its own final target hour plus the
    settlement delay has passed, so this can never see something a live
    forecaster would not have had.
    """
    import polars as pl

    from surge import store

    try:
        # forecast_points carries no BA of its own; the BA and the issuance
        # timing live on forecast_issuances, so the join is required rather
        # than incidental.
        issuances = (
            store.scan("forecast_issuances")
            .filter(pl.col("ba") == ba)
            .select("issuance_id", "issued_at_utc", "first_valid_at_utc")
            .collect()
        )
        if issuances.is_empty():
            return []
        wanted = issuances["issuance_id"].implode()
        points = (
            store.scan("forecast_points")
            .filter(pl.col("issuance_id").is_in(wanted))
            .select("issuance_id", "valid_at_utc", "p10_mw", "p90_mw")
            .collect()
        )
        verified = (
            store.scan("forecast_verifications")
            .filter(pl.col("ba") == ba)
            .select("issuance_id", "valid_at_utc", "actual_mw")
            .collect()
        )
    except Exception:
        # No ledger yet, or an unreadable one: serve uncalibrated rather than
        # guess. The caller records that calibration did not apply.
        return []
    if points.is_empty() or verified.is_empty():
        return []

    joined = points.join(verified, on=["issuance_id", "valid_at_utc"], how="inner").join(
        issuances.select("issuance_id", "issued_at_utc"), on="issuance_id", how="inner"
    )
    if joined.is_empty():
        return []

    out: list[MaturedOrigin] = []
    for issuance_id, group in joined.sort("valid_at_utc").group_by(
        "issuance_id", maintain_order=True
    ):
        group = group.sort("valid_at_utc")
        if group.height != horizon:
            # A partially settled or short issuance is not a usable origin.
            continue
        origin = group["issued_at_utc"][0]
        del issuance_id
        out.append(
            MaturedOrigin(
                origin_utc=origin,
                lower=group["p10_mw"].to_numpy().astype(float),
                upper=group["p90_mw"].to_numpy().astype(float),
                actual=group["actual_mw"].to_numpy().astype(float),
            )
        )
    out.sort(key=lambda item: item.origin_utc)
    settled = mature_before(
        out, now_utc=now_utc, horizon=horizon, outcome_delay_hours=outcome_delay_hours
    )
    return settled[-limit:]
