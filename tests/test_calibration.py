"""Serving-side calibration: causal, fails open, and agrees with the research lane."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from surge.calibration import (
    TARGET_COVERAGE,
    CalibrationState,
    MaturedOrigin,
    calibrate,
    cqr_scores,
    finite_sample_quantile,
    mature_before,
)

HORIZON = 4


def _history(count: int, *, band: float = 5.0, noise: float = 20.0) -> list[MaturedOrigin]:
    """Origins whose band is far too narrow, so calibration must widen."""
    generator = np.random.default_rng(11)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    out = []
    for index in range(count):
        centre = np.full(HORIZON, 1000.0)
        out.append(
            MaturedOrigin(
                origin_utc=start + timedelta(hours=24 * index),
                lower=centre - band,
                upper=centre + band,
                actual=centre + generator.normal(0.0, noise, size=HORIZON),
            )
        )
    return out


def test_too_little_history_serves_the_model_interval_unchanged() -> None:
    low = np.full(HORIZON, 995.0)
    high = np.full(HORIZON, 1005.0)
    out_low, out_high, state = calibrate(low, high, history=_history(3))
    assert state.applied is False
    assert "needs" in (state.reason or "")
    assert np.array_equal(out_low, low)
    assert np.array_equal(out_high, high)


def test_a_persistently_missed_interval_is_widened() -> None:
    low = np.full(HORIZON, 995.0)
    high = np.full(HORIZON, 1005.0)
    out_low, out_high, state = calibrate(low, high, history=_history(60))
    assert state.applied is True
    assert state.mature_origins == 60
    assert np.all(out_high - out_low > high - low)


def test_an_over_wide_interval_is_allowed_to_tighten() -> None:
    """Canonical CQR is two-sided; without this the level can only ratchet up."""
    history = _history(60, band=400.0, noise=5.0)
    low = np.full(HORIZON, 600.0)
    high = np.full(HORIZON, 1400.0)
    out_low, out_high, _ = calibrate(low, high, history=history)
    assert np.all(out_high - out_low < high - low)


def test_only_settled_origins_are_eligible() -> None:
    """An outcome a live forecaster could not have had must not calibrate it."""
    history = _history(5)
    last = history[-1].origin_utc
    # The final origin settles at its last target hour plus 72 h.
    just_before = last + timedelta(hours=(HORIZON - 1) + 72) - timedelta(minutes=1)
    just_after = last + timedelta(hours=(HORIZON - 1) + 72)
    assert len(mature_before(history, now_utc=just_before, horizon=HORIZON,
                             outcome_delay_hours=72)) == 4
    assert len(mature_before(history, now_utc=just_after, horizon=HORIZON,
                             outcome_delay_hours=72)) == 5


def test_calibration_state_is_declared_in_provenance() -> None:
    _, _, state = calibrate(
        np.full(HORIZON, 995.0), np.full(HORIZON, 1005.0), history=_history(40)
    )
    provenance = state.as_provenance()
    assert provenance["calibration_applied"] is True
    assert provenance["calibration_mature_origins"] == 40
    assert provenance["calibration_policy_version"] == "surge-conformal-aci-v1"
    assert provenance["calibration_effective_coverage"] is not None


def test_uncalibrated_issuances_still_declare_why() -> None:
    _, _, state = calibrate(
        np.full(HORIZON, 995.0), np.full(HORIZON, 1005.0), history=_history(2)
    )
    provenance = state.as_provenance()
    assert provenance["calibration_applied"] is False
    assert provenance["calibration_reason"]


def test_primitives_match_the_research_implementation() -> None:
    """Serving and research must not drift apart silently."""
    from experiments.conformal import cqr_scores as research_scores
    from experiments.conformal import finite_sample_quantile as research_quantile

    generator = np.random.default_rng(5)
    lower = generator.normal(90, 5, size=50)
    upper = lower + generator.uniform(5, 20, size=50)
    actual = generator.normal(100, 12, size=50)

    mine = cqr_scores(lower, upper, actual)
    theirs = research_scores(
        lower.reshape(-1, 1, 1),
        upper.reshape(-1, 1, 1),
        actual.reshape(-1, 1, 1),
        normalized=True,
    ).reshape(-1)
    assert np.allclose(mine, theirs)
    assert finite_sample_quantile(mine, coverage=TARGET_COVERAGE) == pytest.approx(
        research_quantile(theirs, coverage=TARGET_COVERAGE)
    )


def test_shape_errors_are_rejected() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        calibrate(np.zeros((2, 2)), np.ones((2, 2)), history=_history(40))
    with pytest.raises(ValueError, match="upper must not be below lower"):
        calibrate(np.full(HORIZON, 10.0), np.full(HORIZON, 1.0), history=_history(40))


def test_calibration_state_defaults_are_inert() -> None:
    state = CalibrationState(applied=False)
    assert state.adjustment_mw == []
    assert state.mature_origins == 0


def test_history_loader_joins_through_issuances_for_the_ba() -> None:
    """forecast_points carries no BA of its own; the join is required.

    Filtering points on a column they do not have raised, and the loader's
    broad except turned that into "no history" -- so calibration would have
    silently never engaged in production.
    """
    import inspect

    from surge import calibration

    source = inspect.getsource(calibration.matured_history)
    assert 'store.scan("forecast_issuances")' in source
    assert 'pl.col("ba") == ba' in source
    # The BA filter must be applied to issuances, never to forecast_points.
    points_call = source.split('store.scan("forecast_points")')[1].split(".collect()")[0]
    assert '"ba"' not in points_call
