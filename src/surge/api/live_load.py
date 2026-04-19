"""Rolling US aggregate demand from the local load_hourly store.

Powers the `/current-load` endpoint which drives the "US demand right now"
hero on the playground. Not a forecast — just a time series of summed
actuals, used so the landing feels tied to what the grid is doing right
this moment.
"""
from __future__ import annotations

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


def aggregate_load(hours: int = 24) -> dict[str, Any]:
    """Return an hour-by-hour sum of US demand over the last `hours` hours.

    Strategy: group load_hourly rows by ts_utc and sum load_mw across every
    BA that reported for that hour. Low-coverage hours (under
    `_MIN_BA_COVERAGE` reporting BAs) are dropped — otherwise the
    most-recent hour shows a cliff as it waits for lagging BAs to publish.
    """
    now = datetime.now(tz=UTC).replace(minute=0, second=0, microsecond=0)
    # Wider lookback than requested so we can drop trailing low-coverage
    # rows and still have `hours` good ones to return.
    lookback_start = now - timedelta(hours=hours + 12)

    df = (
        store.scan("load_hourly")
        .filter(pl.col("ts_utc") >= lookback_start)
        .filter(pl.col("load_mw").is_not_null())
        .filter(pl.col("load_mw") > 0)
        .filter(pl.col("load_mw") < _SANE_BA_MW_CEILING)
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
