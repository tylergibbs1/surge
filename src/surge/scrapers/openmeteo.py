"""Archived day-ahead weather *forecasts* via Open-Meteo's Previous Runs API.

    ts_utc, ba, temp_c_fcst, temp_c_anal, rad_wm2_fcst, wind100_fcst,
    lead_hours, source, as_of

Why a separate dataset from `weather_hourly`
--------------------------------------------
`weather_hourly` holds observed ASOS values. Feeding those over a forecast
horizon is the leak that inflated every earlier accuracy number in this repo.
This module fetches what was actually *forecast* for hour t, issued roughly 24 h
before t, which is genuinely knowable at day-ahead forecast time.

The `temperature_2m_previous_dayN` family is aligned to fixed lead-time offsets:
`previous_day1` is the prediction made ~24 h before the valid time. That is the
right variable for a day-ahead task; `temperature_2m` on the same endpoint is
the latest (effectively analysed) value and must NOT be used as a covariate.

Archive coverage: GFS 2 m temperature reaches back to ~2021-03. Most other
models only start ~2024-01, which is why `gfs_seamless` is the default.

Measured day-ahead skill over 2025 at the seven RTO centroids: MAE 1.06-1.55 C,
RMSE 1.37-2.03 C, correlation 0.983-0.990, bias -0.40..+0.20 C.
"""
from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta

import polars as pl

from surge import store
from surge import bas as _bas
from surge.scrapers.base import client, get

BASE = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Open-Meteo serves long ranges in one response, but keep chunks modest so a
# transient failure costs little and we stay polite on the free tier.
CHUNK_DAYS = 120
ARCHIVE_START = date(2021, 3, 1)


def _fetch_window(lat: float, lon: float, start: date, end: date,
                  *, lead_days: int, model: str) -> pl.DataFrame:
    # Renewable proxies alongside temperature. Realized wind/solar generation is
    # never knowable ahead, but *forecast* irradiance and 100 m wind speed are,
    # and they drive the same physics — so they are the causal substitute for the
    # EIA actual-generation channels this repo used to leak.
    var = f"temperature_2m_previous_day{lead_days}"
    rad = f"shortwave_radiation_previous_day{lead_days}"
    wnd = f"wind_speed_100m_previous_day{lead_days}"
    # Fetch the analysis alongside the forecast, from the SAME grid point. The
    # observed ASOS station can sit hundreds of km from a BA's centroid (PJM's
    # centroid is in the WV mountains; its station is DCA), which would leave a
    # multi-degree discontinuity where the past covariate meets the future one.
    # Taking both channels from one source keeps them coherent.
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": f"temperature_2m,{var},{rad},{wnd}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "models": model,
        "timezone": "UTC",
    }
    with client() as c:
        r = get(c, BASE, params=params, timeout=120.0)
    hourly = (r.json() or {}).get("hourly") or {}
    times = hourly.get("time") or []
    n = len(times)
    cols = {
        "temp_c_fcst": hourly.get(var) or [None] * n,
        "temp_c_anal": hourly.get("temperature_2m") or [None] * n,
        "rad_wm2_fcst": hourly.get(rad) or [None] * n,
        "wind100_fcst": hourly.get(wnd) or [None] * n,
    }
    schema = {"ts_utc": pl.Datetime(time_zone="UTC"),
              **{k: pl.Float64 for k in cols}}
    if not times:
        return pl.DataFrame(schema=schema)
    return (
        pl.DataFrame({"ts_utc": times, **cols})
          .with_columns(
              pl.col("ts_utc").str.to_datetime().dt.replace_time_zone("UTC"),
              *[pl.col(k).cast(pl.Float64) for k in cols],
          )
          .drop_nulls("temp_c_fcst")
    )


def fetch_ba(ba: str, start: date, end: date, *, lead_days: int = 1,
             model: str = "gfs_seamless", persist: bool = True) -> pl.DataFrame:
    """Day-ahead forecast temperature for `ba`'s centroid over [start, end]."""
    meta = _bas.get(ba)
    lon, lat = meta.centroid          # registry stores (lon, lat)
    start = max(start, ARCHIVE_START)
    if start > end:
        raise ValueError(f"{ba}: start {start} is after end {end}")

    frames: list[pl.DataFrame] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=CHUNK_DAYS - 1))
        part = _fetch_window(lat, lon, cursor, chunk_end,
                             lead_days=lead_days, model=model)
        if not part.is_empty():
            frames.append(part)
        cursor = chunk_end + timedelta(days=1)
        if cursor <= end:
            time.sleep(1.0)           # courtesy pause between chunks

    if frames:
        df = pl.concat(frames).unique(subset=["ts_utc"]).sort("ts_utc")
    else:
        df = pl.DataFrame(schema={
            "ts_utc": pl.Datetime(time_zone="UTC"),
            "temp_c_fcst": pl.Float64, "temp_c_anal": pl.Float64,
            "rad_wm2_fcst": pl.Float64, "wind100_fcst": pl.Float64})

    df = df.with_columns(
        pl.lit(ba).alias("ba"),
        pl.lit(lead_days * 24).cast(pl.Int32).alias("lead_hours"),
        pl.lit(f"open-meteo-{model}").alias("source"),
        pl.lit(datetime.now(tz=UTC)).alias("as_of"),
    )

    if persist and not df.is_empty():
        store.write_through(
            "weather_fcst_hourly", df,
            source=f"open-meteo-{model}",
            key=f"{ba}:d{lead_days}:{start}:{end}",
        )
    return df
