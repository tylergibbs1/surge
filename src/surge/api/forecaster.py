"""Pure forecasting logic. The FastAPI app creates & injects the pipeline.

No globals, no singletons, no locks — the app's lifespan manager owns the
model and passes it into this module via `forecast_ba(pipe=..., ba=...)`.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import holidays
import numpy as np
import polars as pl

from surge import store


MODEL_PATH = os.environ.get(
    "SURGE_MODEL_PATH",
    str(Path(__file__).resolve().parents[3] / "models" / "chronos2_full_v2"),
)
MODEL_NAME = "chronos-2-ft-v2"
CONTEXT_LENGTH = 2048

US_HOLIDAYS = holidays.UnitedStates()


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
        out[m] = out[~m][0]
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
    load = (store.scan("load_hourly")
              .filter(pl.col("ba") == ba)
              .select("ts_utc", "load_mw")
              .sort("ts_utc")
              .collect())
    load = load.with_columns(
        pl.when(pl.col("load_mw") > 200_000).then(None).otherwise(pl.col("load_mw")).alias("load_mw")
    )
    weather = (store.scan("weather_hourly")
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
    temp_future = np.full(horizon, bd["temp_c"][end_idx - 1], dtype=np.float32)

    past_covariates = {"temp_c": temp_past.astype(np.float32), **cal_past}
    future_covariates = {"temp_c": temp_future, **cal_future}

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
        ts_i = future_ts[i].astype("datetime64[s]").astype(datetime).replace(tzinfo=timezone.utc)
        points.append({
            "ts_utc": ts_i,
            "median_mw": float(q[i, 1]),
            "p10_mw": float(q[i, 0]),
            "p90_mw": float(q[i, 2]),
        })

    return {
        "points": points,
        "context_start_utc": bd["ts"][start_idx].astype("datetime64[s]").astype(datetime).replace(tzinfo=timezone.utc),
        "context_end_utc":   bd["ts"][end_idx - 1].astype("datetime64[s]").astype(datetime).replace(tzinfo=timezone.utc),
    }
