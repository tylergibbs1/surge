"""Hourly ASOS station observations via Iowa Mesonet.

No auth required. One station per BA, chosen as a representative major
load centre. Output schema:
    ts_utc, station, ba, temp_c, source="asos-iowa", as_of
"""
from __future__ import annotations

import io
import time
from datetime import UTC, date, datetime

import polars as pl

from surge import bas as _bas
from surge import store
from surge.scrapers.base import client, get

BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# BA -> representative ASOS station. Single source of truth is surge.bas;
# exposed here as a dict for backwards compatibility with existing callers.
BA_STATIONS: dict[str, str] = _bas.stations()


def _fahrenheit_to_c(f: pl.Expr) -> pl.Expr:
    return (f - 32.0) * (5.0 / 9.0)


def _fetch_window(station: str, start: str, end: str) -> pl.DataFrame:
    """One request to Iowa Mesonet covering [start, end). Returns an
    hour-bucketed DataFrame with (ts_utc, temp_c); empty if no obs."""
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
    df = pl.read_csv(
        io.StringIO(r.text),
        null_values=["M", "T", ""],
        schema_overrides={"tmpf": pl.Utf8},
    )
    if df.is_empty():
        return pl.DataFrame(schema={
            "ts_utc": pl.Datetime(time_zone="UTC"),
            "temp_c": pl.Float64,
        })
    df = df.with_columns(
        pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC").alias("ts_utc"),
        pl.col("tmpf").cast(pl.Float64, strict=False).alias("tmpf"),
    )
    return (df.with_columns(pl.col("ts_utc").dt.truncate("1h").alias("ts_utc"))
              .group_by("ts_utc").agg(pl.col("tmpf").mean())
              .sort("ts_utc")
              .with_columns(_fahrenheit_to_c(pl.col("tmpf")).alias("temp_c"))
              .drop("tmpf"))


def _year_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split [start, end) into calendar-year windows. Mesonet 429s or times
    out on multi-year single requests (a 7yr pull reliably fails), so long
    spans fan out into year-sized chunks."""
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    out: list[tuple[str, str]] = []
    cur = s
    while cur < e:
        nxt = min(date(cur.year + 1, 1, 1), e)
        out.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt
    return out


def fetch_station(
    station: str,
    ba: str,
    start: str,          # "2018-01-01"
    end: str,            # "2026-01-01"  (exclusive upper bound)
    *,
    persist: bool = True,
) -> pl.DataFrame:
    chunks = _year_chunks(start, end)
    frames: list[pl.DataFrame] = []
    for i, (c_start, c_end) in enumerate(chunks):
        if i > 0:
            # Iowa Mesonet rate-limits sessions aggressively on multi-year
            # backfills — 7 back-to-back year pulls reliably trip 429. A
            # longer inter-chunk pause keeps the session below their cap.
            time.sleep(8.0)
        part = _fetch_window(station, c_start, c_end)
        if not part.is_empty():
            frames.append(part)

    if frames:
        df = pl.concat(frames).unique(subset=["ts_utc"]).sort("ts_utc")
    else:
        df = pl.DataFrame(schema={
            "ts_utc": pl.Datetime(time_zone="UTC"),
            "temp_c": pl.Float64,
        })

    df = df.with_columns(
        pl.lit(station).alias("station"),
        pl.lit(ba).alias("ba"),
        pl.lit("asos-iowa").alias("source"),
        pl.lit(datetime.now(tz=UTC)).alias("as_of"),
    )

    if persist:
        store.write_through(
            "weather_hourly", df,
            source="asos-iowa", key=f"{ba}:{station}:{start}:{end}",
        )
    return df
