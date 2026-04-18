"""Build joined multi-BA datasets: load + temperature + calendar features.

Returned layout per BA:
    {
      "target":           np.ndarray[T] load in MW,
      "past_covariates":  {name: np.ndarray[T]},
      "future_keys":      set[str]  # covariates known into the future
    }

`future_keys` contains the covariate names that are deterministic or
assumed-known-at-forecast-time:
    temp_c  (assumes perfect weather forecast — upper-bound)
    hour_sin, hour_cos, dow_sin, dow_cos, is_weekend, is_holiday
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import holidays
import numpy as np
import polars as pl

from surge import store


US_HOLIDAYS = holidays.UnitedStates()


def _ffill_np(x: np.ndarray) -> np.ndarray:
    out = x.astype(np.float64).copy()
    last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    mask = np.isnan(out)
    if mask.any():
        out[mask] = out[~mask][0]
    return out


def _calendar(ts_utc: np.ndarray) -> dict[str, np.ndarray]:
    # `ts_utc` is a datetime64[us, UTC] numpy array from polars.
    ts = ts_utc.astype("datetime64[h]")
    hour = (ts - ts.astype("datetime64[D]")).astype(int) % 24
    day  = ts.astype("datetime64[D]").astype("datetime64[s]").astype("O")
    dow  = np.array([d.weekday() for d in day], dtype=np.float32)
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


@dataclass
class BAData:
    ba: str
    ts_utc: np.ndarray                   # (T,) datetime64
    target: np.ndarray                   # (T,) load MW
    covariates: dict[str, np.ndarray]    # each (T,)
    future_keys: list[str]
    train_end: int
    val_end: int
    denom_mae: float

    def slice(self, start: int, end: int) -> dict:
        return {
            "target": self.target[start:end],
            "past_covariates": {k: v[start:end] for k, v in self.covariates.items()},
        }

    def future_dict(self, start: int, end: int) -> dict[str, np.ndarray]:
        return {k: self.covariates[k][start:end] for k in self.future_keys}


def _join_ba(ba: str, *, with_gen: bool = True) -> BAData:
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

    if with_gen:
        gen = (store.scan("gen_by_fuel")
                 .filter(pl.col("ba") == ba)
                 .filter(pl.col("fuel").is_in(["WND", "SUN"]))
                 .group_by(["ts_utc", "fuel"])
                 .agg(pl.col("gen_mw").mean())
                 .collect()
                 .pivot(values="gen_mw", index="ts_utc", on="fuel")
                 .sort("ts_utc"))
        rename = {"WND": "wind_mw", "SUN": "solar_mw"}
        gen = gen.rename({k: v for k, v in rename.items() if k in gen.columns})
        j = j.join(gen, on="ts_utc", how="left")

    ts = j["ts_utc"].to_numpy()
    target = _ffill_np(j["load_mw"].to_numpy().astype(np.float64))
    temp   = _ffill_np(j["temp_c"].to_numpy().astype(np.float64)).astype(np.float32)
    cal = _calendar(ts)

    covariates: dict[str, np.ndarray] = {"temp_c": temp, **cal}
    future_keys = ["temp_c", *cal.keys()]

    if with_gen:
        for col in ("wind_mw", "solar_mw"):
            if col in j.columns:
                v = _ffill_np(j[col].to_numpy().astype(np.float64))
                covariates[col] = v.astype(np.float32)
                # Wind/solar are *assumed-known-at-forecast-time* (i.e., using a perfect
                # renewable forecast). Production would substitute a forecast series.
                future_keys.append(col)

    future_keys = sorted(set(future_keys))

    years = j["ts_utc"].dt.year().to_numpy()
    train_end = int(np.searchsorted(years, 2024, side="left"))
    val_end   = int(np.searchsorted(years, 2025, side="left"))

    train = target[:train_end]
    denom = float(np.nanmean(np.abs(train[24:] - train[:-24])))

    return BAData(
        ba=ba, ts_utc=ts, target=target.astype(np.float32),
        covariates=covariates, future_keys=future_keys,
        train_end=train_end, val_end=val_end, denom_mae=denom,
    )


def load_multi_ba(bas: list[str], *, with_gen: bool = True) -> dict[str, BAData]:
    return {ba: _join_ba(ba, with_gen=with_gen) for ba in bas}
