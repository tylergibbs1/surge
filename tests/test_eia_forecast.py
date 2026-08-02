"""The operator baseline must be hour-aligned, per BA, or the headline lies."""

from __future__ import annotations

import polars as pl
import pytest

from surge.scrapers.eia import DF_HOUR_OFFSET, forecast


class _Response:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def json(self) -> dict:
        return {"response": {"data": self._rows}}


@pytest.fixture
def _offline(monkeypatch) -> list[dict]:
    """Serve one fixed DF page and keep every side effect off the filesystem."""
    rows = [
        {"period": "2024-06-01T00", "respondent": "PJM", "value": 100.0},
        {"period": "2024-06-01T01", "respondent": "PJM", "value": 110.0},
    ]
    monkeypatch.setenv("EIA_API_KEY", "test-key")
    monkeypatch.setenv("SURGE_VINTAGE_ARCHIVE", "0")
    monkeypatch.setattr("surge.scrapers.eia.client", lambda: _NullClient())
    monkeypatch.setattr("surge.scrapers.eia.get", lambda c, url, params: _Response(rows))
    monkeypatch.setattr("surge.scrapers.eia.store.write_through", lambda *a, **k: None)
    monkeypatch.setattr("surge.scrapers.eia.store.append", lambda *a, **k: None)
    return rows


class _NullClient:
    def __enter__(self) -> _NullClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _for(ba: str, monkeypatch) -> pl.DataFrame:
    monkeypatch.setattr(
        "surge.scrapers.eia.get",
        lambda c, url, params: _Response(
            [
                {"period": "2024-06-01T00", "respondent": ba, "value": 100.0},
                {"period": "2024-06-01T01", "respondent": ba, "value": 110.0},
            ]
        ),
    )
    return forecast(ba, "2024-06-01", "2024-06-02")


def test_pjm_and_ciso_are_shifted_one_hour(_offline, monkeypatch) -> None:
    """Measured on real data: the shift cuts their apparent error by 20-31%."""
    for ba in ("PJM", "CISO"):
        frame = _for(ba, monkeypatch)
        assert frame["hour_offset_applied"][0] == 1
        assert (
            frame["ts_utc"][0] - frame["published_ts_utc"][0]
        ).total_seconds() == 3600.0


def test_other_rtos_are_left_alone(_offline, monkeypatch) -> None:
    """The same shift makes ERCO 24-71% worse, so it must not be global."""
    for ba in ("ERCO", "MISO", "NYIS", "ISNE", "SWPP"):
        frame = _for(ba, monkeypatch)
        assert frame["hour_offset_applied"][0] == 0
        assert frame["ts_utc"][0] == frame["published_ts_utc"][0]


def test_the_published_period_is_retained_for_audit(_offline, monkeypatch) -> None:
    """A silent correction is indistinguishable from a data error."""
    frame = _for("PJM", monkeypatch)
    assert "published_ts_utc" in frame.columns
    assert "hour_offset_applied" in frame.columns


def test_the_offset_registry_covers_only_the_measured_bas() -> None:
    assert set(DF_HOUR_OFFSET) == {"PJM", "CISO"}
    assert all(value == 1 for value in DF_HOUR_OFFSET.values())


def test_the_offset_can_be_measured_rather_than_assumed() -> None:
    """A static table cannot express a convention that changes between years."""
    import numpy as np

    from surge.scrapers.eia import measure_df_hour_offset

    hours = pl.datetime_range(
        pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 20), interval="1h", eager=True
    )
    shape = 1000 + 200 * np.sin(2 * np.pi * np.arange(len(hours)) / 24)
    demand = pl.DataFrame({"ts": hours, "D": shape})
    # A forecast labelled one hour early: DF(t) actually describes D(t+1).
    forecast_frame = pl.DataFrame({"ts": hours, "DF": np.roll(shape, -1)})

    best, scores = measure_df_hour_offset(demand, forecast_frame)
    assert best == 1
    assert scores[1] < scores[0]


def test_a_perfectly_aligned_series_measures_zero_offset() -> None:
    import numpy as np

    from surge.scrapers.eia import measure_df_hour_offset

    hours = pl.datetime_range(
        pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 20), interval="1h", eager=True
    )
    shape = 1000 + 200 * np.sin(2 * np.pi * np.arange(len(hours)) / 24)
    demand = pl.DataFrame({"ts": hours, "D": shape})
    forecast_frame = pl.DataFrame({"ts": hours, "DF": shape * 1.01})
    best, _ = measure_df_hour_offset(demand, forecast_frame)
    assert best == 0


def test_the_validated_window_is_declared() -> None:
    """The table is a measurement with a scope, not a constant."""
    from surge.scrapers.eia import DF_OFFSET_VALIDATED_WINDOW

    assert DF_OFFSET_VALIDATED_WINDOW == ("2024-01-01", "2026-01-01")


def test_the_measured_offset_agrees_in_sign_with_the_applied_one() -> None:
    """A measurement that reads backwards from its correction is a trap."""
    import numpy as np

    from surge.scrapers.eia import DF_HOUR_OFFSET, measure_df_hour_offset

    hours = pl.datetime_range(
        pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 20), interval="1h", eager=True
    )
    shape = 1000 + 200 * np.sin(2 * np.pi * np.arange(len(hours)) / 24)
    demand = pl.DataFrame({"ts": hours, "D": shape})
    # Reproduce what the ingest corrects: PJM's published DF(t) describes D(t+1),
    # which the ingest fixes by adding DF_HOUR_OFFSET["PJM"] == +1 hour.
    forecast_frame = pl.DataFrame({"ts": hours, "DF": np.roll(shape, -1)})
    best, _ = measure_df_hour_offset(demand, forecast_frame)
    assert best == DF_HOUR_OFFSET["PJM"] == 1
