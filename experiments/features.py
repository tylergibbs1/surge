"""Build joined multi-BA datasets: load + temperature + calendar features.

Returned layout per BA:
    {
      "target":           np.ndarray[T] load in MW,
      "past_covariates":  {name: np.ndarray[T]},
      "future_keys":      set[str]  # covariates supplied over the horizon
    }

Covariate futures: what a forecaster may legitimately know
----------------------------------------------------------
Every covariate carries a *policy* saying how its values over the forecast
horizon are produced. This matters because the underlying series are all
**realized observations** — `temp_c` is observed ASOS station data (Iowa
Mesonet), and `wind_mw`/`solar_mw` are EIA-930 *actual* generation. Slicing
them over the forecast window and passing them as future covariates leaks the
future into the input.

Policies:
    known        deterministic from the timestamp alone — calendar features.
                 No leakage.
    persistence  held flat at the last value observed before the forecast
                 origin. This is a real, causal forecast, and it is what the
                 production API does (see surge.api.forecaster).
    oracle       realized future values, i.e. perfect foresight. Leaks. Only
                 valid as an explicitly-labelled upper bound, never as a
                 headline number or in a comparison against a real forecaster.
    past_only    supplied as history only, never over the horizon.

`future_mode` picks the policy set for the non-calendar covariates:
    "persistence"  temp_c=persistence, wind/solar=past_only   (default; causal)
    "none"         temp_c=past_only,   wind/solar=past_only
    "oracle"       temp_c=oracle,      wind/solar=oracle       (leaky; opt-in)

Because `persistence` depends on the forecast origin, future covariates cannot
be precomputed as a flat array — call `BAData.future_at(origin, horizon)`.
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
        # Backfill leading NaNs with the first real value; if the array is
        # entirely NaN (e.g. a BA with no weather data yet), fall back to 0
        # so training doesn't crash — the model treats it as a flat covariate
        # with no signal.
        real = out[~mask]
        out[mask] = real[0] if real.size else 0.0
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


FUTURE_MODES = ("persistence", "lag24", "none", "oracle")

CALENDAR_KEYS = ("hour_sin", "hour_cos", "dow_sin", "dow_cos",
                 "is_weekend", "is_holiday")


def _lag_idx(origin: int, horizon: int, n: int, lag: int = 24) -> np.ndarray:
    """Indices of the same-hour-yesterday profile for [origin, origin+horizon).

    Every returned index is strictly < `origin`, so this is observable at
    forecast time. When the horizon runs past the lag, the most recent
    available day is tiled forward rather than reaching into the future.
    """
    idx = np.arange(origin, origin + horizon) - lag
    while idx.max(initial=-1) >= origin:
        idx = np.where(idx >= origin, idx - lag, idx)
    return np.clip(idx, 0, n - 1)


def _resolve_policy(mode: str, gen_cols: list[str]) -> dict[str, str]:
    """Covariate name -> policy, for a given `future_mode`."""
    if mode not in FUTURE_MODES:
        raise ValueError(f"future_mode must be one of {FUTURE_MODES}, got {mode!r}")

    policy = {k: "known" for k in CALENDAR_KEYS}
    if mode == "oracle":
        policy["temp_c"] = "oracle"
        for c in gen_cols:
            policy[c] = "oracle"
    elif mode == "persistence":
        policy["temp_c"] = "persistence"
        # Actual renewable generation is not knowable at forecast time and the
        # production API does not supply it, so it stays history-only.
        for c in gen_cols:
            policy[c] = "past_only"
    elif mode == "lag24":
        policy["temp_c"] = "lag24"
        for c in gen_cols:
            policy[c] = "past_only"
    else:  # "none"
        policy["temp_c"] = "past_only"
        for c in gen_cols:
            policy[c] = "past_only"
    return policy


@dataclass
class BAData:
    ba: str
    ts_utc: np.ndarray                   # (T,) datetime64
    target: np.ndarray                   # (T,) load MW
    covariates: dict[str, np.ndarray]    # each (T,)
    future_policy: dict[str, str]        # covariate name -> policy
    future_mode: str
    train_end: int
    val_end: int
    test_end: int
    denom_mae: float

    @property
    def future_keys(self) -> list[str]:
        """Covariates supplied over the horizon (any policy but past_only)."""
        return sorted(k for k, p in self.future_policy.items() if p != "past_only")

    @property
    def leaks_future(self) -> bool:
        return any(p == "oracle" for p in self.future_policy.values())

    def slice(self, start: int, end: int) -> dict:
        return {
            "target": self.target[start:end],
            "past_covariates": {k: v[start:end] for k, v in self.covariates.items()},
        }

    def future_at(self, origin: int, horizon: int) -> dict[str, np.ndarray]:
        """Future covariates for the window [origin, origin+horizon), per policy.

        `origin` is the first *forecast* index; everything at index < origin is
        observable. Under `persistence` the value is frozen at origin-1.
        """
        out: dict[str, np.ndarray] = {}
        for k, pol in self.future_policy.items():
            if pol == "past_only":
                continue
            v = self.covariates[k]
            if pol in ("known", "oracle"):
                out[k] = v[origin:origin + horizon]
            elif pol == "persistence":
                last = v[origin - 1] if origin > 0 else v[0]
                out[k] = np.full(horizon, last, dtype=v.dtype)
            elif pol == "lag24":
                out[k] = v[_lag_idx(origin, horizon, len(v))]
            else:
                raise ValueError(f"unknown policy {pol!r} for {k!r}")
        return out

    def temp_at(self, origin: int, horizon: int) -> np.ndarray:
        """Temperature over the horizon under this BA's policy.

        Lets the classical baselines consume weather on the same terms as the
        foundation models, so the leaderboard stays like-for-like.
        """
        pol = self.future_policy.get("temp_c", "past_only")
        v = self.covariates["temp_c"]
        if pol == "oracle":
            return v[origin:origin + horizon]
        if pol == "lag24":
            return v[_lag_idx(origin, horizon, len(v))]
        last = v[origin - 1] if origin > 0 else v[0]
        # Under "persistence" this is exactly what the Chronos models receive.
        # Under "none" the Chronos models get no temperature at all, while these
        # baselines have temp baked in as a trained feature and need *some*
        # value — persistence is the weakest causal stand-in, so "none" leaves
        # the baselines marginally advantaged. Compare on "persistence".
        return np.full(horizon, last, dtype=v.dtype)


def _join_ba(ba: str, *, with_gen: bool = True,
             future_mode: str = "persistence") -> BAData:
    # Dedupe on the business key: the store is append-only, so overlapping
    # backfill windows and EIA in-place revisions leave several rows per hour.
    # Without this the target series carries repeated hours, which silently
    # breaks the lag/seasonal-naive arithmetic and the split indices.
    load = (store.scan("load_hourly", dedupe_on=["ts_utc", "ba"])
              .filter(pl.col("ba") == ba)
              .select("ts_utc", "load_mw")
              .sort("ts_utc")
              .collect())
    load = load.with_columns(
        pl.when(pl.col("load_mw") > 200_000).then(None).otherwise(pl.col("load_mw")).alias("load_mw")
    )
    weather = (store.scan("weather_hourly", dedupe_on=["ts_utc", "ba"])
                 .filter(pl.col("ba") == ba)
                 .select("ts_utc", "temp_c")
                 .sort("ts_utc")
                 .collect())
    j = load.join(weather, on="ts_utc", how="left")

    if with_gen:
        gen = (store.scan("gen_by_fuel", dedupe_on=["ts_utc", "ba", "fuel"])
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
    gen_cols: list[str] = []

    if with_gen:
        for col in ("wind_mw", "solar_mw"):
            if col in j.columns:
                v = _ffill_np(j[col].to_numpy().astype(np.float64))
                covariates[col] = v.astype(np.float32)
                gen_cols.append(col)

    future_policy = _resolve_policy(future_mode, gen_cols)

    years = j["ts_utc"].dt.year().to_numpy()
    train_end = int(np.searchsorted(years, 2024, side="left"))
    val_end   = int(np.searchsorted(years, 2025, side="left"))
    # The test split is calendar 2025, closed at the top. Without this bound it
    # would run to the end of the store, so every fresh ingest would silently
    # redefine "test MASE" and published numbers would stop being comparable.
    test_end  = int(np.searchsorted(years, 2026, side="left"))

    train = target[:train_end]
    denom = float(np.nanmean(np.abs(train[24:] - train[:-24])))

    return BAData(
        ba=ba, ts_utc=ts, target=target.astype(np.float32),
        covariates=covariates, future_policy=future_policy, future_mode=future_mode,
        train_end=train_end, val_end=val_end, test_end=test_end, denom_mae=denom,
    )


# Number of correlated peer BAs whose PAST load is attached as a covariate.
# Their history is observable at forecast time, so this is causal; only their
# future would be off-limits. 0 disables.
N_NEIGHBORS = 2


def _align_to(src_ts: np.ndarray, src_val: np.ndarray,
              dst_ts: np.ndarray) -> np.ndarray:
    """Reindex `src_val` onto `dst_ts` by exact timestamp match, then ffill."""
    idx = np.clip(np.searchsorted(src_ts, dst_ts), 0, len(src_ts) - 1)
    hit = src_ts[idx] == dst_ts
    return _ffill_np(np.where(hit, src_val[idx], np.nan))


def _attach_neighbors(bas: dict[str, BAData], k: int) -> None:
    """Add the k most train-correlated peer BAs' load as past-only covariates.

    Peers are chosen on the *train* split only — picking them on validation
    would tune the feature set to the split being scored.
    """
    if k <= 0 or len(bas) < 2:
        return

    codes = list(bas)
    aligned: dict[tuple[str, str], np.ndarray] = {}
    for tgt in codes:
        bd = bas[tgt]
        for src in codes:
            if src == tgt:
                continue
            aligned[(tgt, src)] = _align_to(
                bas[src].ts_utc, bas[src].target.astype(np.float64), bd.ts_utc)

    for tgt in codes:
        bd = bas[tgt]
        own = bd.target[:bd.train_end].astype(np.float64)
        scored = []
        for src in codes:
            if src == tgt:
                continue
            peer = aligned[(tgt, src)][:bd.train_end]
            if peer.std() == 0 or own.std() == 0:
                continue
            r = float(np.corrcoef(own, peer)[0, 1])
            if np.isfinite(r):
                scored.append((abs(r), src))
        scored.sort(reverse=True)

        for rank, (_, src) in enumerate(scored[:k], start=1):
            name = f"nbr{rank}_load"
            bd.covariates[name] = aligned[(tgt, src)].astype(np.float32)
            bd.future_policy[name] = "past_only"


def load_multi_ba(bas: list[str], *, with_gen: bool = True,
                  future_mode: str = "persistence") -> dict[str, BAData]:
    out = {ba: _join_ba(ba, with_gen=with_gen, future_mode=future_mode)
           for ba in bas}
    _attach_neighbors(out, N_NEIGHBORS)
    return out
