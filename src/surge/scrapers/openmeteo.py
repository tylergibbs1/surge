"""Archived day-ahead weather *forecasts* via Open-Meteo's Previous Runs API.

    ts_utc, ba, temp_c_fcst, temp_c_anal, rad_wm2_fcst, wind100_fcst,
    temp_spread_c, lead_hours, source, as_of

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
    # Ask three independent NWP models for the same day-ahead temperature. Their
    # disagreement is a causal measure of how uncertain tomorrow's weather is,
    # which a single deterministic run cannot express — the load model can use it
    # to widen its own intervals on genuinely uncertain days instead of applying
    # one constant widening everywhere.
    spread_models = [model, "icon_seamless", "ecmwf_ifs025"]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": f"temperature_2m,{var},{rad},{wnd}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "models": ",".join(spread_models),
        "timezone": "UTC",
    }
    with client() as c:
        r = get(c, BASE, params=params, timeout=120.0)
    hourly = (r.json() or {}).get("hourly") or {}
    times = hourly.get("time") or []
    n = len(times)

    def pick(base: str, mdl: str | None = None):
        """Open-Meteo suffixes variable names per model when several are asked for."""
        for key in ((f"{base}_{mdl}",) if mdl else ()) + (base,):
            if hourly.get(key):
                return hourly[key]
        return [None] * n

    # Spread across the three models, per hour, ignoring absent members.
    members = [pick(var, m) for m in spread_models]
    spread = []
    for i in range(n):
        vals = [m[i] for m in members if i < len(m) and m[i] is not None]
        spread.append(max(vals) - min(vals) if len(vals) >= 2 else None)

    cols = {
        "temp_c_fcst": pick(var, model),
        "temp_c_anal": pick("temperature_2m", model),
        "rad_wm2_fcst": pick(rad, model),
        "wind100_fcst": pick(wnd, model),
        "temp_spread_c": spread,
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


LIVE_BASE = "https://api.open-meteo.com/v1/forecast"

# Major load centres per RTO, (lat, lon).
#
# A single BA centroid is a poor stand-in for a multi-state footprint: ISO-NE's
# centroid lands in New Hampshire while its demand sits in Boston and
# Connecticut, and PJM's covers thirteen states from one point in the West
# Virginia highlands. FETS handles the same problem by sampling 16 NUTS1
# centroids for Germany rather than one national point.
#
# These are the largest metropolitan load centres inside each footprint, which
# approximates a demand-weighted average of the weather the load actually sees.
LOAD_CENTERS: dict[str, list[tuple[float, float]]] = {
    "PJM":  [(39.95, -75.16), (38.90, -77.04), (41.88, -87.63),
             (40.44, -79.996), (39.29, -76.61), (40.74, -74.17)],
    "CISO": [(34.05, -118.24), (37.77, -122.42), (32.72, -117.16),
             (38.58, -121.49), (36.75, -119.77)],
    "ERCO": [(29.76, -95.37), (32.78, -96.80), (30.27, -97.74),
             (29.42, -98.49), (31.76, -106.49)],
    "MISO": [(44.98, -93.27), (42.33, -83.05), (38.63, -90.20),
             (39.77, -86.16), (43.04, -87.91)],
    "NYIS": [(40.71, -74.01), (42.89, -78.88), (42.65, -73.76),
             (43.16, -77.61)],
    "ISNE": [(42.36, -71.06), (41.76, -72.69), (41.82, -71.41),
             (43.66, -70.26), (42.10, -72.59)],
    "SWPP": [(39.10, -94.58), (35.47, -97.52), (41.26, -95.93),
             (36.15, -95.99), (37.69, -97.34)],
}


def centers_for(ba: str) -> list[tuple[float, float]]:
    """Weather sampling points for a BA: load centres if known, else centroid."""
    if ba in LOAD_CENTERS:
        return LOAD_CENTERS[ba]
    lon, lat = _bas.get(ba).centroid
    return [(lat, lon)]


def live_forecast(ba: str, *, past_days: int = 7, forecast_days: int = 2,
                  model: str = "gfs_seamless") -> pl.DataFrame:
    """Current weather forecast for `ba`'s centroid: recent past + next hours.

    For serving, not for benchmarking. Returns one coherent series covering both
    sides of "now" from a single model run, so the history the model sees and the
    forecast it is given come from the same source — mixing an observation
    station's history with a model-grid forecast leaves a step change exactly at
    the forecast boundary.

    Columns: ts_utc, temp_c, rad_wm2, wind100.
    """
    empty = pl.DataFrame(schema={"ts_utc": pl.Datetime(time_zone="UTC"),
                                 "temp_c": pl.Float64, "rad_wm2": pl.Float64,
                                 "wind100": pl.Float64})
    # Same load-centre averaging as the archive path. Serving and evaluation must
    # build this channel identically, or the model meets a covariate at inference
    # that differs from the one it was scored on.
    frames = []
    for lat, lon in centers_for(ba):
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,shortwave_radiation,wind_speed_100m",
            "past_days": int(past_days),
            "forecast_days": int(forecast_days),
            "models": model,
            "timezone": "UTC",
        }
        with client() as c:
            r = get(c, LIVE_BASE, params=params, timeout=30.0)
        h = (r.json() or {}).get("hourly") or {}
        times = h.get("time") or []
        if not times:
            continue
        n = len(times)
        frames.append(
            pl.DataFrame({
                "ts_utc": times,
                "temp_c": h.get("temperature_2m") or [None] * n,
                "rad_wm2": h.get("shortwave_radiation") or [None] * n,
                "wind100": h.get("wind_speed_100m") or [None] * n,
            })
            .with_columns(
                pl.col("ts_utc").str.to_datetime().dt.replace_time_zone("UTC"),
                pl.col("temp_c").cast(pl.Float64),
                pl.col("rad_wm2").cast(pl.Float64),
                pl.col("wind100").cast(pl.Float64),
            )
        )
    if not frames:
        return empty
    return (pl.concat(frames).group_by("ts_utc").agg(pl.all().mean()).sort("ts_utc"))


def fetch_ba(ba: str, start: date, end: date, *, lead_days: int = 1,
             model: str = "gfs_seamless", persist: bool = True) -> pl.DataFrame:
    """Day-ahead forecast temperature for `ba`'s centroid over [start, end]."""
    start = max(start, ARCHIVE_START)
    if start > end:
        raise ValueError(f"{ba}: start {start} is after end {end}")

    # Average over the BA's load centres rather than a single centroid, so the
    # channel reflects the weather the demand actually experiences.
    points = centers_for(ba)
    frames: list[pl.DataFrame] = []
    for lat, lon in points:
        cursor = start
        while cursor <= end:
            chunk_end = min(end, cursor + timedelta(days=CHUNK_DAYS - 1))
            part = _fetch_window(lat, lon, cursor, chunk_end,
                                 lead_days=lead_days, model=model)
            if not part.is_empty():
                frames.append(part)
            cursor = chunk_end + timedelta(days=1)
            if cursor <= end:
                time.sleep(0.5)       # courtesy pause between chunks

    if frames:
        # Mean across load centres per hour. Averaging (rather than keeping each
        # point as its own channel) keeps the covariate count unchanged, so this
        # is comparable to the single-point runs.
        df = (pl.concat(frames)
                .group_by("ts_utc")
                .agg(pl.all().mean())
                .sort("ts_utc"))
    else:
        df = pl.DataFrame(schema={
            "ts_utc": pl.Datetime(time_zone="UTC"),
            "temp_c_fcst": pl.Float64, "temp_c_anal": pl.Float64,
            "rad_wm2_fcst": pl.Float64, "wind100_fcst": pl.Float64,
            "temp_spread_c": pl.Float64})

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
