"""Rolling US aggregate demand from the local load_hourly store.

Powers the `/current-load` endpoint which drives the "US demand right now"
hero on the playground. Not a forecast — just a time series of summed
actuals, used so the landing feels tied to what the grid is doing right
this moment.
"""
from __future__ import annotations

import asyncio
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

from surge import store

# Any single-BA row above this is almost certainly a data error (PJM peaks
# near 165 GW; nothing else comes close). Clamp to avoid one bad row from
# CPLW or a federal admin dwarfing the whole aggregate.
_SANE_BA_MW_CEILING = 200_000

# Minimum number of BAs that must have reported for an hour before we
# trust the sum. Small BAs lag, so the most recent hour often has only a
# handful reporting and its total looks like a cliff. Drop those.
_MIN_BA_COVERAGE = 35

# Process-local TTL cache. Keyed on (hours, hour_bucket) — bucketed on the
# hour so two callers in the same hour share the same polars scan. The
# Vercel edge cache (s-maxage=50) already absorbs most traffic; this is
# the backstop for direct callers and cold edges.
_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple[int, int], tuple[float, dict[str, Any]]] = {}
_CACHE_TTL_S = 60.0


def _current_hour_bucket() -> int:
    """Epoch-hour of the most recent clock hour (UTC)."""
    return int(time.time() // 3600)


def _scan(hours: int) -> dict[str, Any]:
    """Synchronous polars scan. Called via asyncio.to_thread from the
    endpoint so the event loop isn't blocked on disk I/O. Partition
    pruning relies on year/month predicates matching the hive layout.
    """
    now = datetime.now(tz=UTC).replace(minute=0, second=0, microsecond=0)
    # Wider lookback than requested so we can drop trailing low-coverage
    # rows and still have `hours` good ones to return.
    lookback_start = now - timedelta(hours=hours + 12)

    # Partition predicate: the store is hive-partitioned on (year, month).
    # Handing polars a (year, month) pair rather than just ts_utc lets it
    # skip every partition outside our window — critical given the store
    # has 7 years x 12 months = 84 partitions.
    years_months = set()
    cursor = lookback_start
    while cursor <= now:
        years_months.add((cursor.year, cursor.month))
        cursor += timedelta(days=1)

    scan = store.scan("load_hourly")
    partition_filter = pl.lit(False)
    for y, m in years_months:
        partition_filter = partition_filter | (
            (pl.col("year") == y) & (pl.col("month") == m)
        )

    try:
        df = (
            scan.filter(partition_filter)
            .filter(pl.col("ts_utc") >= lookback_start)
            .filter(pl.col("load_mw").is_not_null())
            .filter(pl.col("load_mw") > 0)
            .filter(pl.col("load_mw") < _SANE_BA_MW_CEILING)
            # Dedupe on (ts_utc, ba): `store.append` isn't idempotent —
            # overlapping ingest windows (e.g. running `--days 7` then
            # `--days 120`) write the same row twice and the aggregate
            # doubles. Prefer the most-recently-written copy so in-place
            # EIA revisions are honoured.
            .sort(["ts_utc", "ba", "as_of"], descending=[False, False, True])
            .unique(subset=["ts_utc", "ba"], keep="first")
            .group_by("ts_utc")
            .agg(
                pl.col("load_mw").sum().alias("total_mw"),
                pl.col("ba").n_unique().alias("ba_count"),
            )
            .filter(pl.col("ba_count") >= _MIN_BA_COVERAGE)
            .sort("ts_utc")
            .tail(hours)
            .collect()
        )
    except pl.exceptions.ColumnNotFoundError:
        # An empty store scans zero files, so the `year` / `month`
        # partition columns don't exist and the predicate fails schema
        # resolution. Treat it the same as "no data" from a caller's POV.
        raise RuntimeError("no recent load data available") from None

    if df.is_empty():
        raise RuntimeError("no recent load data available")

    points = [
        {
            "ts_utc": row["ts_utc"],
            "total_mw": float(row["total_mw"]),
            "ba_count": int(row["ba_count"]),
        }
        for row in df.iter_rows(named=True)
    ]
    latest = points[-1]
    return {
        "as_of_utc": datetime.now(tz=UTC),
        "latest_ts_utc": latest["ts_utc"],
        "latest_total_mw": latest["total_mw"],
        "hours": len(points),
        "points": points,
    }


async def aggregate_load(hours: int = 24) -> dict[str, Any]:
    """Async wrapper: hit the TTL cache, fall through to a threaded scan.

    The cache key is (hours, current hour bucket) so the first call in a
    given clock hour pays the polars cost and every subsequent call
    within `_CACHE_TTL_S` gets a dict copy for free.
    """
    key = (hours, _current_hour_bucket())
    now_s = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None and now_s - cached[0] < _CACHE_TTL_S:
            return cached[1]

    payload = await asyncio.to_thread(_scan, hours)
    with _CACHE_LOCK:
        _CACHE[key] = (time.monotonic(), payload)
        # Keep the cache from growing unbounded on pods that stay warm
        # across many hours of distinct hour buckets.
        if len(_CACHE) > 16:
            oldest = min(_CACHE.items(), key=lambda kv: kv[1][0])[0]
            _CACHE.pop(oldest, None)
    return payload
