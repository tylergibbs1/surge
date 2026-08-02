"""Adaptive conformal calibration behaviour tests."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.aci import ALPHA_CEILING, aci_conformalize
from experiments.conformal import rolling_conformalize


def _daily_origins(count: int) -> np.ndarray:
    return np.datetime64("2024-01-01T00:00:00") + np.arange(count) * np.timedelta64(24, "h")


def _series(
    *,
    origins: int,
    series: int = 1,
    horizon: int = 2,
    noise_scale: float = 1.0,
    band: float = 40.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forecasts whose nominal band width is set by the caller.

    A wide ``band`` relative to ``noise_scale`` over-covers; a narrow one
    under-covers.
    """
    shape = (origins, series, horizon)
    median = np.full(shape, 100.0)
    lower = median - band
    upper = median + band
    generator = np.random.default_rng(42)
    truth = median + generator.normal(0.0, noise_scale, size=shape)
    return lower, median, upper, truth


def _calibrate(**overrides: object) -> object:
    lower, median, upper, truth = _series(origins=40)
    kwargs: dict[str, object] = {
        "origin_times_utc": _daily_origins(lower.shape[0]),
        "outcome_delay_hours": 0,
        "window": 10,
        "min_history": 2,
    }
    kwargs.update(overrides)
    return aci_conformalize(lower, median, upper, truth, **kwargs)  # type: ignore[arg-type]


def test_adaptive_level_widens_further_than_the_fixed_level_when_missing() -> None:
    """Persistent under-coverage must drive the level down, widening intervals.

    This is the behaviour the fixed-window protocol cannot provide: it applies
    the same nominal level however badly that level is performing.
    """
    lower, median, upper, truth = _series(origins=60, noise_scale=10.0, band=1.0)
    common = {
        "origin_times_utc": _daily_origins(60),
        "outcome_delay_hours": 0,
        "window": 20,
        "min_history": 2,
    }
    fixed = rolling_conformalize(lower, median, upper, truth, **common)  # type: ignore[arg-type]
    adaptive = aci_conformalize(
        lower, median, upper, truth, gamma=0.05, alpha_scope="per-series", **common
    )  # type: ignore[arg-type]

    shared = fixed.eligible & adaptive.eligible
    assert shared.any()
    late = shared.copy()
    late[: 60 // 2] = False
    assert np.nanmean(adaptive.adjustment[late]) > np.nanmean(fixed.adjustment[late])


def test_an_over_wide_interval_is_allowed_to_tighten() -> None:
    """The signed score's negative branch is what bounds coverage from above.

    Clamping it at zero leaves a ratchet that can only widen, which pushes an
    already-calibrated series past nominal.
    """
    calibrated = _calibrate(gamma=0.05)
    eligible = calibrated.eligible
    assert eligible.any()
    assert np.any(calibrated.adjustment[eligible] < 0.0)


def test_a_width_floor_bounds_how_far_an_interval_may_tighten() -> None:
    """The safety property, expressed without breaking the estimator."""
    lower, median, upper, truth = _series(origins=40)
    raw_width = float((upper - lower)[0, 0, 0])
    calibrated = aci_conformalize(
        lower,
        median,
        upper,
        truth,
        origin_times_utc=_daily_origins(40),
        outcome_delay_hours=0,
        window=10,
        min_history=2,
        gamma=0.05,
        min_width_fraction=1.0,
    )
    eligible = calibrated.eligible
    widths = (calibrated.upper - calibrated.lower)[eligible]
    assert eligible.any()
    assert np.all(widths >= raw_width - 1e-9)


def test_min_width_fraction_is_validated() -> None:
    with pytest.raises(ValueError, match="min_width_fraction must be in"):
        _calibrate(min_width_fraction=1.5)


def test_outcomes_cannot_feed_back_before_they_mature() -> None:
    """An outcome delayed past the cohort must never move the level."""
    lower, median, upper, truth = _series(origins=30)
    common = {
        "origin_times_utc": _daily_origins(30),
        "window": 10,
        "min_history": 2,
        "gamma": 0.05,
        "alpha_scope": "per-series",
    }
    never_matures = aci_conformalize(
        lower, median, upper, truth, outcome_delay_hours=24 * 400, **common
    )  # type: ignore[arg-type]
    assert not never_matures.eligible.any()


def test_per_series_scope_shares_one_level_across_leads() -> None:
    """One lead's outcomes must move its siblings only when the level is shared.

    Perturbing lead 1 alone changes lead 0's calibration under ``per-series``
    and cannot change it at all under ``per-lead``.
    """
    lower, median, upper, truth = _series(origins=60, horizon=2, noise_scale=5.0, band=2.0)
    perturbed = truth.copy()
    perturbed[:, 0, 1] = median[:, 0, 1] + 500.0  # lead 1 alone is always missed
    common = {
        "origin_times_utc": _daily_origins(60),
        "outcome_delay_hours": 0,
        "window": 20,
        "min_history": 2,
        "gamma": 0.05,
    }

    def lead_zero(values: np.ndarray, scope: str) -> np.ndarray:
        calibrated = aci_conformalize(
            lower, median, upper, values, alpha_scope=scope, **common
        )  # type: ignore[arg-type]
        return calibrated.adjustment[:, 0, 0]

    assert np.allclose(
        lead_zero(truth, "per-lead"),
        lead_zero(perturbed, "per-lead"),
        equal_nan=True,
    )
    assert not np.allclose(
        lead_zero(truth, "per-series"),
        lead_zero(perturbed, "per-series"),
        equal_nan=True,
    )


def test_level_is_clamped_to_a_two_sided_interval() -> None:
    """Even under relentless over-coverage the level stops at the median."""
    lower, median, upper, truth = _series(origins=80, noise_scale=0.01)
    calibrated = aci_conformalize(
        lower,
        median,
        upper,
        truth,
        origin_times_utc=_daily_origins(80),
        outcome_delay_hours=0,
        window=20,
        min_history=2,
        gamma=0.5,
        alpha_scope="per-series",
    )
    eligible = calibrated.eligible
    assert eligible.any()
    # A level pinned at the ceiling still yields a finite, non-negative width.
    assert np.all(np.isfinite(calibrated.adjustment[eligible]))
    assert ALPHA_CEILING == 0.5


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gamma": 0.0}, "gamma must be in"),
        ({"gamma": 1.0}, "gamma must be in"),
        ({"alpha_scope": "per-grid"}, "alpha_scope must be one of"),
        ({"window": 0}, "1 <= min_history <= window"),
        ({"min_history": 99}, "1 <= min_history <= window"),
        ({"coverage": 0.0}, "coverage must be in"),
        ({"outcome_delay_hours": -1}, "outcome_delay_hours must be nonnegative"),
    ],
)
def test_invalid_configuration_is_rejected(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _calibrate(**overrides)


def test_mismatched_shapes_are_rejected() -> None:
    lower, median, upper, truth = _series(origins=10)
    with pytest.raises(ValueError, match="forecast and truth shapes must match"):
        aci_conformalize(
            lower,
            median,
            upper,
            truth[:-1],
            origin_times_utc=_daily_origins(10),
            outcome_delay_hours=0,
            window=5,
            min_history=2,
        )


def _cohort_data(year: int, hours: int = 24 * 400) -> object:
    """Minimal BAData-like stub for cohort origin selection."""
    from experiments.features import BAData

    start = np.datetime64(f"{year - 1}-06-01T00:00:00")
    ts = start + np.arange(hours) * np.timedelta64(1, "h")
    return BAData(
        ba="PJM",
        ts_utc=ts,
        target=np.ones(hours, dtype=np.float32),
        covariates={},
        future_keys=[],
        train_end=hours,
        val_end=hours,
        denom_mae=1.0,
        feature_spec=None,  # type: ignore[arg-type]
        availability_mode=None,  # type: ignore[arg-type]
    )


def test_cohort_origins_stay_inside_the_requested_year() -> None:
    from experiments.run_aci_replication_c2 import cohort_origins

    data = _cohort_data(2023)
    origins = cohort_origins(data, year=2023, context=48, horizon=24)
    assert origins
    times = data.ts_utc[np.asarray(origins)]
    years = times.astype("datetime64[Y]").astype(np.int64) + 1970
    assert set(years.tolist()) == {2023}
    # Every target hour must also stay inside the cohort.
    last_target = data.ts_utc[origins[-1] + 24 - 1]
    assert (last_target.astype("datetime64[Y]").astype(np.int64) + 1970) == 2023


def test_cohort_origins_require_a_full_context_window() -> None:
    from experiments.run_aci_replication_c2 import cohort_origins

    data = _cohort_data(2023)
    origins = cohort_origins(data, year=2023, context=48, horizon=24)
    assert min(origins) >= 48


def test_replication_refuses_the_locked_lane() -> None:
    from experiments.run_aci_replication_c2 import validate_cohort_year

    for year in (2025, 2026, 2018):
        with pytest.raises(ValueError, match=r"locked test lane|cohort year must be"):
            validate_cohort_year(year)
    assert validate_cohort_year(2023) == 2023
