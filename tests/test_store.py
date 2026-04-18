"""Parquet datastore tests."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from surge import store


def _sample() -> pl.DataFrame:
    return pl.DataFrame({
        "ts_utc": [
            datetime(2024, 1, 15, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 15, 1, tzinfo=timezone.utc),
            datetime(2024, 2, 1, 0, tzinfo=timezone.utc),
        ],
        "ba": ["PJM", "PJM", "PJM"],
        "load_mw": [80000.0, 78000.0, 81000.0],
    })


def test_append_partitions_by_year_and_month(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    store.append("load_hourly", _sample())

    assert (tmp_path / "load_hourly" / "year=2024" / "month=01").exists()
    assert (tmp_path / "load_hourly" / "year=2024" / "month=02").exists()


def test_scan_reads_everything_back(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    store.append("load_hourly", _sample())
    df = store.scan("load_hourly").collect()
    assert df.height == 3
    # partition columns round-trip when hive_partitioning=True
    assert {"year", "month"}.issubset(df.columns)


def test_manifest_and_ingested_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    store.record("load_hourly", source="eia-930", key="PJM-2024-01", n_rows=744)
    assert "PJM-2024-01" in store.ingested_keys("load_hourly", "eia-930")
    assert "PJM-2024-02" not in store.ingested_keys("load_hourly", "eia-930")


def test_write_through_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    store.write_through("load_hourly", _sample(), source="eia-930", key="PJM-2024-01")
    # Second call should be a no-op (same key).
    store.write_through("load_hourly", _sample(), source="eia-930", key="PJM-2024-01")
    df = store.scan("load_hourly").collect()
    assert df.height == 3
