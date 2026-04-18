"""Hourly ASOS station observations via Iowa Mesonet.

No auth required. One station per BA, chosen as a representative major
load centre. Output schema:
    ts_utc, station, ba, temp_c, source="asos-iowa", as_of
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import polars as pl

from surge import store
from surge.scrapers.base import client, get

BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# BA -> representative ASOS station (major load-centre airport).
BA_STATIONS: dict[str, str] = {
    "PJM":  "DCA",   # Washington National
    "CISO": "SFO",   # San Francisco International
    "ERCO": "AUS",   # Austin-Bergstrom
    "MISO": "MSP",   # Minneapolis-St Paul
    "NYIS": "JFK",   # NYC John F. Kennedy
    "ISNE": "BOS",   # Boston Logan
    "SWPP": "OKC",   # Oklahoma City
}


def _fahrenheit_to_c(f: pl.Expr) -> pl.Expr:
    return (f - 32.0) * (5.0 / 9.0)


def fetch_station(
    station: str,
    ba: str,
    start: str,          # "2018-01-01"
    end: str,            # "2026-01-01"  (exclusive upper bound year)
    *,
    persist: bool = True,
) -> pl.DataFrame:
    y1, m1, d1 = start.split("-")
    y2, m2, d2 = end.split("-")
    params = {
        "station": station,
        "data": "tmpf",
        "year1": y1, "month1": m1.lstrip("0") or "1", "day1": d1.lstrip("0") or "1",
        "year2": y2, "month2": m2.lstrip("0") or "1", "day2": d2.lstrip("0") or "1",
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": "3",   # routine hourly
    }
    with client() as c:
        r = get(c, BASE, params=params, timeout=120.0)
    # CSV: station,valid,tmpf
    df = pl.read_csv(
        io.StringIO(r.text),
        null_values=["M", "T", ""],
        schema_overrides={"tmpf": pl.Utf8},
    )
    df = df.with_columns(
        pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC").alias("ts_utc"),
        pl.col("tmpf").cast(pl.Float64, strict=False).alias("tmpf"),
    )
    # Hour-end bucket: floor to the hour and take the mean observation.
    df = (df.with_columns(pl.col("ts_utc").dt.truncate("1h").alias("ts_utc"))
            .group_by("ts_utc").agg(pl.col("tmpf").mean())
            .sort("ts_utc")
            .with_columns(_fahrenheit_to_c(pl.col("tmpf")).alias("temp_c"))
            .drop("tmpf")
            .with_columns(
                pl.lit(station).alias("station"),
                pl.lit(ba).alias("ba"),
                pl.lit("asos-iowa").alias("source"),
                pl.lit(datetime.now(tz=timezone.utc)).alias("as_of"),
            ))

    if persist:
        store.write_through(
            "weather_hourly", df,
            source="asos-iowa", key=f"{ba}:{station}:{start}:{end}",
        )
    return df
