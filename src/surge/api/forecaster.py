"""Pure forecasting logic. The FastAPI app creates & injects the pipeline.

No globals, no singletons, no locks — the app's lifespan manager owns the
model and passes it into this module via `forecast_ba(pipe=..., ba=...)`.
"""
from __future__ import annotations

import logging
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import holidays
import numpy as np
import polars as pl

from surge import store
from surge.clean import clean_load

# Prefer HF Hub (so users don't need to download 478 MB before running). If
# a local path override is set, use that (e.g. during offline dev or CI).
# Default is the 53-BA surge-fm-v3 generalist; set SURGE_MODEL_PATH=
# Tylerbry1/surge-fm-v2 to serve the 7-RTO specialist instead.
_DEFAULT_HF = "Tylerbry1/surge-fm-v3"
_LOCAL_FALLBACK = Path(__file__).resolve().parents[3] / "models" / "chronos2_full_v3"
MODEL_PATH = os.environ.get(
    "SURGE_MODEL_PATH",
    str(_LOCAL_FALLBACK) if _LOCAL_FALLBACK.exists() else _DEFAULT_HF,
)
# Pin a specific commit SHA when loading from the HF hub. Defends against
# upstream-repo takeover (the loader would otherwise pull `main` which is
# mutable). Override via env when publishing a new checkpoint.
MODEL_REVISION = os.environ.get(
    "SURGE_MODEL_REVISION", "b84726ca520b9d443236d025a000cc95616a334c"
)
MODEL_NAME = "surge-fm-v3"
CONTEXT_LENGTH = 2048

US_HOLIDAYS = holidays.UnitedStates()

_log = logging.getLogger(__name__)


def _ffill(x: np.ndarray) -> np.ndarray:
    out = x.astype(np.float64).copy()
    last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    m = np.isnan(out)
    if m.any():
        # Backfill leading NaNs; fall back to 0 if the whole array is NaN
        # (e.g. a BA we've never ingested weather for — served as a flat
        # covariate rather than crashing the forecast).
        real = out[~m]
        out[m] = real[0] if real.size else 0.0
    return out


def _calendar(ts_utc: np.ndarray) -> dict[str, np.ndarray]:
    ts = ts_utc.astype("datetime64[h]")
    hour = (ts - ts.astype("datetime64[D]")).astype(int) % 24
    day = ts.astype("datetime64[D]").astype("datetime64[s]").astype("O")
    dow = np.array([d.weekday() for d in day], dtype=np.float32)
    weekend = (dow >= 5).astype(np.float32)
    holiday = np.array([1.0 if date(d.year, d.month, d.day) in US_HOLIDAYS else 0.0
                        for d in day], dtype=np.float32)
    two_pi = 2 * np.pi
    return {
        "hour_sin": np.sin(two_pi * hour / 24).astype(np.float32),
        "hour_cos": np.cos(two_pi * hour / 24).astype(np.float32),
        "dow_sin":  np.sin(two_pi * dow / 7).astype(np.float32),
        "dow_cos":  np.cos(two_pi * dow / 7).astype(np.float32),
        "is_weekend": weekend,
        "is_holiday": holiday,
    }


def _load_ba(ba: str) -> dict[str, Any]:
    # Dedupe at the store layer so overlapping ingest windows or in-place
    # EIA revisions don't feed double-counted rows into the model context.
    load = (store.scan("load_hourly", dedupe_on=["ts_utc", "ba"])
              .filter(pl.col("ba") == ba)
              .select("ts_utc", "load_mw")
              .sort("ts_utc")
              .collect())
    # Same robust rejection as the evaluation path, so the context the model sees
    # in production matches the one it was scored on.
    load = clean_load(load, "load_mw")
    weather = (store.scan("weather_hourly", dedupe_on=["ts_utc", "ba"])
                 .filter(pl.col("ba") == ba)
                 .select("ts_utc", "temp_c")
                 .sort("ts_utc")
                 .collect())
    j = load.join(weather, on="ts_utc", how="left")
    return {
        "ts": j["ts_utc"].to_numpy(),
        "target": _ffill(j["load_mw"].to_numpy()),
        "temp_c": _ffill(j["temp_c"].to_numpy()).astype(np.float32),
    }


def data_end_utc() -> datetime | None:
    try:
        df = (store.scan("load_hourly")
                .select(pl.col("ts_utc").max().alias("m"))
                .collect())
        if df.is_empty() or df["m"][0] is None:
            return None
        return df["m"][0]
    except Exception:
        return None


def _weather_channels(ba: str, past_ts: np.ndarray,
                      future_ts: np.ndarray) -> dict | None:
    """Live Open-Meteo weather aligned to both the context and the horizon.

    Returns {"past": {...}, "future": {...}} of covariate arrays, or None if the
    forecast cannot be obtained or does not fully cover the horizon. Callers must
    then omit future weather rather than substitute a constant, which scores
    worse than omitting it entirely.

    Both sides come from one model run, so history and forecast are on the same
    footing — and the channel set matches what the model was evaluated with,
    which means `rad`/`wind` need history too, not just future values. 92 past
    days is the API maximum and covers the 2048-hour context (85 days).
    """
    try:
        from surge.scrapers.openmeteo import live_forecast
        wx = live_forecast(ba, past_days=92, forecast_days=3)
    except Exception as e:                      # network, API change, unknown BA
        _log.warning("%s: live_forecast failed (%s); omitting future weather",
                     ba, type(e).__name__)
        return None
    if wx.is_empty():
        return None

    src = wx["ts_utc"].to_numpy().astype("datetime64[h]")

    def align(want: np.ndarray) -> np.ndarray | None:
        want = np.asarray(want).astype("datetime64[h]")
        idx = np.searchsorted(src, want)
        if len(src) == 0 or idx.max(initial=0) >= len(src):
            return None
        return idx if np.all(src[idx] == want) else None

    fut_idx = align(future_ts)
    if fut_idx is None:
        _log.warning("%s: forecast does not cover the horizon; omitting", ba)
        return None
    past_idx = align(past_ts)          # may be None if context predates the window

    cols = {"temp_c": "temp_c", "rad_fcst": "rad_wm2", "wind_fcst": "wind100"}
    future, past = {}, {}
    for name, col in cols.items():
        v = wx[col].to_numpy().astype(np.float64)
        if np.all(np.isnan(v)):
            continue
        v = _ffill(v)
        future[name] = v[fut_idx].astype(np.float32)
        if past_idx is not None:
            past[name] = v[past_idx].astype(np.float32)

    if not future:
        return None
    # Without aligned history the extra channels would appear only over the
    # horizon, a schema the model never saw. Serve temperature alone in that case.
    if past_idx is None:
        future = {k: v for k, v in future.items() if k == "temp_c"}
        past = {}
    return {"past": past, "future": future}


def forecast_ba(pipe: Any, ba: str, horizon: int = 24) -> dict[str, Any]:
    """Produce a 1-BA probabilistic forecast using the loaded pipeline."""
    if horizon < 1 or horizon > 168:
        raise ValueError("horizon must be in 1..168")

    bd = _load_ba(ba)
    if len(bd["target"]) < CONTEXT_LENGTH + 1:
        raise ValueError(f"not enough history for {ba} (need {CONTEXT_LENGTH}h)")

    end_idx = len(bd["target"])
    start_idx = end_idx - CONTEXT_LENGTH
    target = bd["target"][start_idx:end_idx].astype(np.float32)
    temp_past = bd["temp_c"][start_idx:end_idx]
    cal_past = _calendar(bd["ts"][start_idx:end_idx])

    last_ts = bd["ts"][end_idx - 1]
    future_ts = (last_ts + np.arange(1, horizon + 1, dtype="timedelta64[h]")).astype("datetime64[h]")
    cal_future = _calendar(future_ts.astype("datetime64[us]"))

    past_covariates = {"temp_c": temp_past.astype(np.float32), **cal_past}
    future_covariates: dict[str, np.ndarray] = {**cal_future}

    # Real day-ahead weather forecast over the horizon.
    #
    # This used to send a flat persisted temperature, which is measurably WORSE
    # than sending no future temperature at all: on the 7 RTOs, test MASE 0.740
    # with it against 0.594 without. A constant 24 h temperature implies no
    # diurnal cycle, and the checkpoint was fine-tuned to trust the covariate.
    # A real forecast reaches 0.536, so this is the single largest accuracy lever
    # in the serving path.
    #
    # If the forecast is unavailable, fall back to OMITTING future weather —
    # never to a flat value, which would be worse than nothing.
    temp_future = None
    wx = _weather_channels(ba, bd["ts"][start_idx:end_idx], future_ts)
    if wx is not None:
        past_covariates.update(wx["past"])
        future_covariates.update(wx["future"])
        temp_future = wx["future"].get("temp_c")
    else:
        _log.warning(
            "%s: no weather forecast available; serving without future weather "
            "(expect ~0.59 vs ~0.54 MASE). Not substituting a flat temperature.", ba)

    task = [{
        "target": target,
        "past_covariates": past_covariates,
        "future_covariates": future_covariates,
    }]
    quants_list, _ = pipe.predict_quantiles(
        task, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9],
        batch_size=1,
    )
    q = quants_list[0].squeeze(0).float().cpu().numpy()  # (H, 3)

    points = []
    for i in range(horizon):
        ts_i = future_ts[i].astype("datetime64[s]").astype(datetime).replace(tzinfo=UTC)
        points.append({
            "ts_utc": ts_i,
            "median_mw": float(q[i, 1]),
            "p10_mw": float(q[i, 0]),
            "p90_mw": float(q[i, 2]),
            # The forecast temperature actually fed to the model, or null when no
            # forecast was available. Never a flat placeholder: reporting a
            # constant here would misrepresent the input as well as degrade it.
            "temp_c": (float(temp_future[i]) if temp_future is not None else None),
        })

    return {
        "points": points,
        "context_start_utc": bd["ts"][start_idx].astype("datetime64[s]").astype(datetime).replace(tzinfo=UTC),
        "context_end_utc":   bd["ts"][end_idx - 1].astype("datetime64[s]").astype(datetime).replace(tzinfo=UTC),
    }
