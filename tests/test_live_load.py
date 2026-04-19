"""Tests for the /current-load aggregate (src/surge/api/live_load.py).

Covers the three edge cases flagged in code review:
  1. Empty store — should raise RuntimeError, not succeed with garbage
  2. All hours below the BA-coverage threshold — same: raise
  3. Sanity ceiling (`_SANE_BA_MW_CEILING`) — a corrupt oversized row
     for one BA must not leak into the aggregate
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from surge import store
from surge.api import live_load


def _hours_back(n: int) -> list[datetime]:
    now = datetime.now(tz=UTC).replace(minute=0, second=0, microsecond=0)
    return [now - timedelta(hours=i) for i in range(n, 0, -1)]


def _write_full_coverage(tmp_path, n_hours: int, per_ba_mw: float = 1000.0) -> None:
    """Seed every BA x every hour at `per_ba_mw`. With 50 synthetic BAs,
    the aggregate at every hour is above the coverage threshold."""
    ts = _hours_back(n_hours)
    bas = [f"BA{i:02d}" for i in range(50)]
    rows = {"ts_utc": [], "ba": [], "load_mw": []}
    for t in ts:
        for b in bas:
            rows["ts_utc"].append(t)
            rows["ba"].append(b)
            rows["load_mw"].append(per_ba_mw)
    store.append("load_hourly", pl.DataFrame(rows))


def _clear_cache() -> None:
    """live_load caches per (hours, hour_bucket). Drop it between cases."""
    with live_load._CACHE_LOCK:
        live_load._CACHE.clear()


def test_empty_store_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    _clear_cache()
    with pytest.raises(RuntimeError, match="no recent load data"):
        asyncio.run(live_load.aggregate_load(hours=24))


def test_all_hours_below_coverage_raises(tmp_path, monkeypatch) -> None:
    """Only 5 BAs reporting — far below the 35-BA threshold. Every hour
    is dropped, so there's nothing to return. Must raise, not return zero
    rows or a garbage sum from the lagging subset."""
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    _clear_cache()
    ts = _hours_back(10)
    bas = [f"SMALL{i}" for i in range(5)]
    rows = {"ts_utc": [], "ba": [], "load_mw": []}
    for t in ts:
        for b in bas:
            rows["ts_utc"].append(t)
            rows["ba"].append(b)
            rows["load_mw"].append(500.0)
    store.append("load_hourly", pl.DataFrame(rows))

    with pytest.raises(RuntimeError, match="no recent load data"):
        asyncio.run(live_load.aggregate_load(hours=24))


def test_ceiling_clamp_excludes_corrupt_row(tmp_path, monkeypatch) -> None:
    """If one BA reports an absurd 500 GW value, the clamp at
    `_SANE_BA_MW_CEILING` should keep it out of the sum. Compare the
    aggregate with and without the bad row: must be identical."""
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    _clear_cache()
    _write_full_coverage(tmp_path, n_hours=10, per_ba_mw=1000.0)
    out_clean = asyncio.run(live_load.aggregate_load(hours=24))
    expected_total = 50 * 1000.0  # 50 BAs x 1 GW each

    _clear_cache()
    # Inject a corrupt 500_000 MW row for a 51st BA into one hour.
    bad_hour = _hours_back(10)[5]
    store.append(
        "load_hourly",
        pl.DataFrame({
            "ts_utc": [bad_hour],
            "ba": ["CORRUPT"],
            "load_mw": [500_000.0],
        }),
    )
    out_with_corrupt = asyncio.run(live_load.aggregate_load(hours=24))

    # Find the corrupted hour in both runs; totals must match.
    def total_for(payload, target: datetime) -> float:
        for p in payload["points"]:
            if p["ts_utc"] == target:
                return p["total_mw"]
        raise AssertionError(f"hour {target} missing from payload")

    assert total_for(out_clean, bad_hour) == expected_total
    assert total_for(out_with_corrupt, bad_hour) == expected_total


def test_overlapping_ingest_does_not_double_aggregate(tmp_path, monkeypatch) -> None:
    """Regression for the (ts_utc, ba) dedupe bug.

    `store.append` writes a new parquet per call — so running `--days 7`
    and then `--days 120` literally writes the same row twice. Before the
    store-layer dedupe, the aggregate summed both copies and the "live US
    demand" hero jumped to ~2× reality whenever backfill windows overlapped.
    """
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    _clear_cache()
    # First ingest: 50 BAs × 3 hours at 1 GW each.
    _write_full_coverage(tmp_path, n_hours=3, per_ba_mw=1000.0)
    single = asyncio.run(live_load.aggregate_load(hours=24))
    single_total = single["points"][-1]["total_mw"]

    _clear_cache()
    # Second ingest: identical rows. Parquet on disk now has exact
    # duplicates of every (ts_utc, ba) pair.
    _write_full_coverage(tmp_path, n_hours=3, per_ba_mw=1000.0)
    doubled = asyncio.run(live_load.aggregate_load(hours=24))
    doubled_total = doubled["points"][-1]["total_mw"]

    # Store-layer dedupe collapses the duplicates on read: the sum must
    # match, not double.
    assert doubled_total == single_total
    assert doubled_total == 50 * 1000.0


def test_ttl_cache_reuses_payload(tmp_path, monkeypatch) -> None:
    """Two calls with the same args in the same hour must return the
    literally-same object (cache hit), not rescan the store."""
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    _clear_cache()
    _write_full_coverage(tmp_path, n_hours=3, per_ba_mw=2000.0)
    first = asyncio.run(live_load.aggregate_load(hours=24))
    second = asyncio.run(live_load.aggregate_load(hours=24))
    assert first is second
