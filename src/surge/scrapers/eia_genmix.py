"""EIA-930 hourly generation by fuel type, written to `gen_by_fuel` dataset.

Uses the same v2 API as load but with the fuel-type-data route.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime

import polars as pl

from surge import store
from surge.scrapers.base import client, get

API = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"


def _api_key() -> str:
    key = os.environ.get("EIA_API_KEY")
    if not key:
        raise RuntimeError("EIA_API_KEY missing")
    return key


def fetch(ba: str, fueltype: str, start: str, end: str) -> pl.DataFrame:
    """Hourly generation for (ba, fueltype). fueltype in {COL,NG,NUC,OIL,SUN,WND,WAT,OTH}."""
    params = {
        "api_key": _api_key(),
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": ba,
        "facets[fueltype][]": fueltype,
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
            batch = r.json()["response"].get("data", [])
            rows.extend(batch)
            if len(batch) < params["length"]:
                break
            params["offset"] += params["length"]

    if not rows:
        return pl.DataFrame(schema={
            "ts_utc": pl.Datetime(time_zone="UTC"),
            "ba": pl.Utf8, "fuel": pl.Utf8, "gen_mw": pl.Float64,
        })

    df = (pl.DataFrame(rows)
          .select(
              (pl.col("period") + ":00").str.to_datetime("%Y-%m-%dT%H:%M", time_zone="UTC").alias("ts_utc"),
              pl.col("respondent").alias("ba"),
              pl.col("fueltype").alias("fuel"),
              pl.col("value").cast(pl.Float64).alias("gen_mw"),
          )
          .with_columns(
              pl.lit("eia-930-fuel").alias("source"),
              pl.lit(datetime.now(tz=UTC)).alias("as_of"),
          ))

    store.write_through(
        "gen_by_fuel", df,
        source="eia-930-fuel", key=f"{ba}:{fueltype}:{start}:{end}",
    )
    return df
