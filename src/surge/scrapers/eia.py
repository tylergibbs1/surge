"""EIA-930 hourly BA operational data via the EIA Open Data API v2.

Docs: https://www.eia.gov/opendata/documentation.php
Route: electricity/rto/region-data (demand, generation, net generation, interchange).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import polars as pl

from surge import store, vintage
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

    # Archive what EIA said before anything reshapes it. EIA-930 is preliminary
    # on publication and revised afterwards, so a score is a claim about a
    # vintage; a vintage not captured now cannot be recovered later.
    if os.environ.get("SURGE_VINTAGE_ARCHIVE", "1") != "0":
        try:
            vintage.capture(
                dataset="eia930-demand",
                ba=ba,
                start=start,
                end=end,
                rows=rows,
                request={**params, "url": API},
            )
        except Exception as exc:
            print(f"vintage archive failed for {ba} {start}..{end}: {type(exc).__name__}")

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


# EIA-930 publishes each BA's own day-ahead demand forecast as type=DF. Two of
# the seven RTOs label it against a different hour convention than their own
# realized demand, so DF and D for the same period string describe different
# hours. Measured against realized demand, MAPE for DF(t) vs D(t) versus
# DF(t) vs D(t+1):
#
#   PJM   Jan-24 3.27 -> 2.48   Jan-25 3.09 -> 2.45   Aug-25 3.85 -> 2.66
#   CISO  Jan-24 5.91 -> 4.13   Jan-25 7.84 -> 5.92   Aug-25 7.45 -> 6.64
#   ERCO  Jan-24 3.46 -> 4.29   Jan-25 3.12 -> 3.89   Aug-25 2.16 -> 3.69
#
# PJM and CISO improve by 20-31% under the shift; ERCO degrades by 24-71%, so
# the correction is per-BA and must never be applied globally. Uncorrected, an
# operator baseline overstates PJM and CISO forecast error by roughly a
# quarter -- flattering Surge in any published comparison, which is exactly the
# direction that must not go unnoticed.
DF_HOUR_OFFSET: dict[str, int] = {"PJM": 1, "CISO": 1}


def forecast(ba: str, start: str, end: str, *, force: bool = False) -> pl.DataFrame:
    """The BA's own published day-ahead demand forecast, hour-aligned to demand.

    ``ts_utc`` is the valid hour the forecast describes after correction. The
    published period and the correction applied are both retained so the
    alignment is auditable rather than silently baked in.
    """
    params = {
        "api_key": _api_key(),
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": ba,
        "facets[type][]": "DF",  # Day-ahead demand forecast
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

    if os.environ.get("SURGE_VINTAGE_ARCHIVE", "1") != "0":
        try:
            vintage.capture(
                dataset="eia930-demand-forecast",
                ba=ba,
                start=start,
                end=end,
                rows=rows,
                request={**params, "url": API},
            )
        except Exception as exc:
            print(f"vintage archive failed for {ba} DF {start}..{end}: {type(exc).__name__}")

    if not rows:
        return pl.DataFrame(
            schema={
                "ts_utc": pl.Datetime(time_zone="UTC"),
                "ba": pl.Utf8,
                "load_forecast_mw": pl.Float64,
            }
        )

    offset_hours = DF_HOUR_OFFSET.get(ba.upper(), 0)
    as_of = datetime.now(tz=UTC)
    df = (
        pl.DataFrame(rows)
        .select(
            (pl.col("period") + ":00")
            .str.to_datetime("%Y-%m-%dT%H:%M", time_zone="UTC")
            .alias("published_ts_utc"),
            pl.col("respondent").alias("ba"),
            pl.col("value").cast(pl.Float64).alias("load_forecast_mw"),
        )
        .with_columns(
            (pl.col("published_ts_utc") + pl.duration(hours=offset_hours)).alias("ts_utc"),
            pl.lit(offset_hours).alias("hour_offset_applied"),
            pl.lit("eia-930-df").alias("source"),
            pl.lit(as_of).alias("as_of"),
        )
        .select(
            "ts_utc",
            "ba",
            "load_forecast_mw",
            "published_ts_utc",
            "hour_offset_applied",
            "source",
            "as_of",
        )
    )
    if force:
        store.append("load_forecast_hourly", df)
    else:
        store.write_through(
            "load_forecast_hourly", df, source="eia-930-df", key=f"{ba}:{start}:{end}"
        )
    return df
