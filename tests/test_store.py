"""Parquet datastore tests."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from surge import store


def _sample() -> pl.DataFrame:
    return pl.DataFrame({
        "ts_utc": [
            datetime(2024, 1, 15, 0, tzinfo=UTC),
            datetime(2024, 1, 15, 1, tzinfo=UTC),
            datetime(2024, 2, 1, 0, tzinfo=UTC),
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


def test_scan_as_of_filters_before_latest_revision_wins(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    valid_at = datetime(2026, 1, 1, tzinfo=UTC)
    rows = pl.DataFrame({
        "ts_utc": [valid_at, valid_at],
        "ba": ["PJM", "PJM"],
        "load_mw": [100.0, 150.0],
        "source": ["eia-930", "eia-930"],
        "as_of": [
            datetime(2026, 1, 2, tzinfo=UTC),
            datetime(2026, 1, 5, tzinfo=UTC),
        ],
    })
    store.append("load_hourly", rows)

    observed = store.scan_as_of(
        "load_hourly",
        cutoff=datetime(2026, 1, 3, tzinfo=UTC),
        dedupe_on=["ts_utc", "ba"],
    ).collect()

    assert observed.height == 1
    assert observed["load_mw"][0] == 100.0


def test_write_immutable_is_idempotent_but_rejects_conflicts(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    original = pl.DataFrame({"value": [1]})
    changed = pl.DataFrame({"value": [2]})

    first = store.write_immutable("forecast_points", "stable-id", original)
    retry = store.write_immutable("forecast_points", "stable-id", original)

    assert first == retry
    with pytest.raises(store.ImmutableConflictError, match="different content"):
        store.write_immutable("forecast_points", "stable-id", changed)


def test_write_immutable_recovers_from_stale_lock_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    frame = pl.DataFrame({"value": [1]})
    dest = store.immutable_path("forecast_points", "crashed-writer")
    dest.parent.mkdir(parents=True)
    lock = dest.with_suffix(".lock")
    lock.write_text("pid=999999 created_at=2020-01-01T00:00:00+00:00\n")

    committed = store.write_immutable("forecast_points", "crashed-writer", frame)

    assert committed == dest
    assert pl.read_parquet(committed).equals(frame, null_equal=True)
    # Lock files are stable inode anchors; their mere existence is not a lock.
    assert lock.exists()


def test_write_immutable_never_clobbers_a_noncompliant_racing_writer(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    intended = pl.DataFrame({"value": [1]})
    competing = pl.DataFrame({"value": [2]})
    real_link = store.os.link

    def inject_competing_destination(source, destination) -> None:
        competing.write_parquet(destination)
        real_link(source, destination)

    monkeypatch.setattr(store.os, "link", inject_competing_destination)

    with pytest.raises(store.ImmutableConflictError, match="conflicting race"):
        store.write_immutable("forecast_points", "noncompliant-race", intended)

    committed = store.read_immutable("forecast_points", "noncompliant-race")
    assert committed.equals(competing, null_equal=True)
