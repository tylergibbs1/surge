"""The rail that stops a weather feature becoming an oracle input."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from surge.weather_vintage import (
    OBSERVATIONAL_SOURCES,
    ForecastVintage,
    VintageError,
    assert_no_leakage,
    clear_sky_index,
    usable_at_origin,
)

ORIGIN = datetime(2026, 6, 1, 0, tzinfo=UTC)


def _vintage(*, issued_hours_before: float, source: str = "open-meteo-previous-runs"):
    return ForecastVintage(
        source=source,
        variable="shortwave_radiation",
        valid_at_utc=ORIGIN + timedelta(hours=12),
        issued_at_utc=ORIGIN - timedelta(hours=issued_hours_before),
        value=600.0,
        lead_convention="previous_day1 (~24 h lead, errs older)",
    )


def test_a_value_issued_before_the_origin_is_usable() -> None:
    kept = usable_at_origin([_vintage(issued_hours_before=6)], origin_utc=ORIGIN)
    assert len(kept) == 1


def test_a_value_issued_after_the_origin_is_dropped() -> None:
    kept = usable_at_origin([_vintage(issued_hours_before=-1)], origin_utc=ORIGIN)
    assert kept == []


def test_leakage_fails_loudly_instead_of_being_filtered() -> None:
    """A backtest that silently drops leaked rows still reports a wrong number."""
    with pytest.raises(VintageError, match="issued after the forecast origin"):
        assert_no_leakage([_vintage(issued_hours_before=-3)], origin_utc=ORIGIN)


def test_the_stitched_archive_cannot_be_represented_at_all() -> None:
    """Open-Meteo's 'Historical Forecast API' sounds vintage-correct and is not."""
    with pytest.raises(VintageError, match="observational or reanalysis"):
        _vintage(issued_hours_before=6, source="open-meteo-historical-forecast")


@pytest.mark.parametrize("source", sorted(OBSERVATIONAL_SOURCES))
def test_every_named_observational_source_is_refused(source: str) -> None:
    with pytest.raises(VintageError, match="observational or reanalysis"):
        _vintage(issued_hours_before=6, source=source)


def test_a_forecast_cannot_be_issued_after_the_hour_it_describes() -> None:
    with pytest.raises(VintageError, match="cannot be issued after"):
        ForecastVintage(
            source="nbm",
            variable="ghi",
            valid_at_utc=ORIGIN,
            issued_at_utc=ORIGIN + timedelta(hours=1),
            value=1.0,
            lead_convention="test",
        )


def test_naive_timestamps_are_refused() -> None:
    with pytest.raises(VintageError, match="timezone-aware"):
        ForecastVintage(
            source="nbm",
            variable="ghi",
            valid_at_utc=datetime(2026, 6, 1, 12),
            issued_at_utc=datetime(2026, 6, 1),
            value=1.0,
            lead_convention="test",
        )


def test_clear_sky_index_isolates_cloud_from_season_and_hour() -> None:
    clear = np.array([0.0, 10.0, 400.0, 800.0])
    forecast = np.array([0.0, 5.0, 200.0, 800.0])
    index = clear_sky_index(forecast, clear)
    # Night and near-night return 1.0 rather than a noisy quotient.
    assert index[0] == 1.0
    assert index[1] == 1.0
    # Half the clear-sky maximum is a half index.
    assert index[2] == pytest.approx(0.5)
    # A cloudless hour is 1.0.
    assert index[3] == pytest.approx(1.0)
