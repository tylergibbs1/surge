"""EIA-930 hourly BA operational data via the EIA Open Data API v2.

Docs: https://www.eia.gov/opendata/documentation.php
Route: electricity/rto/region-data (demand, generation, net generation, interchange).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import polars as pl

from surge import store
from surge.scrapers.base import client, get

API = "https://api.eia.gov/v2/electricity/rto/region-data/data/"


def _api_key() -> str:
    key = os.environ.get("EIA_API_KEY")
    if not key:
        raise RuntimeError(
            "EIA_API_KEY is not set. Get a free key at https://www.eia.gov/opendata/register.php"
        )
    return key


def load(ba: str, start: str, end: str, *, force: bool = False) -> pl.DataFrame:
    """Hourly demand (load) for one balancing authority.

    Args:
        ba: EIA BA code, e.g. "PJM", "CISO", "ERCO", "MISO", "NYIS", "ISNE", "SWPP".
        start, end: ISO dates (inclusive start, exclusive end).
        force: Bypass `store.write_through`'s (ba, start, end) manifest
            idempotency and always append. Used by the hourly ingest cron
            so rerunning the same window actually refetches EIA's recent
            in-place revisions. Duplicates collapse on read via
            `store.scan(dedupe_on=...)`.
    """
    params = {
        "api_key": _api_key(),
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": ba,
        "facets[type][]": "D",  # Demand
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }
    rows: list[dict] = []
    with client() as c:
        while True:
            r = get(c, API, params=params)
            payload = r.json()["response"]
            batch = payload.get("data", [])
            rows.extend(batch)
            if len(batch) < params["length"]:
                break
            params["offset"] += params["length"]

    if not rows:
        return pl.DataFrame(schema={"ts_utc": pl.Datetime(time_zone="UTC"),
                                    "ba": pl.Utf8, "load_mw": pl.Float64})

    as_of = datetime.now(tz=UTC)
    df = (
        pl.DataFrame(rows)
        .select(
            (pl.col("period") + ":00")
              .str.to_datetime("%Y-%m-%dT%H:%M", time_zone="UTC").alias("ts_utc"),
            pl.col("respondent").alias("ba"),
            pl.col("value").cast(pl.Float64).alias("load_mw"),
        )
        .with_columns(
            pl.lit("eia-930").alias("source"),
            pl.lit(as_of).alias("as_of"),
        )
    )
    if force:
        store.append("load_hourly", df)
    else:
        store.write_through(
            "load_hourly", df, source="eia-930", key=f"{ba}:{start}:{end}"
        )
    return df
