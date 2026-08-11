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

import os
from dataclasses import dataclass
from datetime import date

import holidays
import numpy as np
import polars as pl

from surge import store


US_HOLIDAYS = holidays.UnitedStates()


def _ffill_np(x: np.ndarray) -> np.ndarray:
    # Vectorised forward fill. This runs thousands of times per evaluation (once
    # per BA pair when peer covariates are built), so the original per-element
    # Python loop dominated wall-clock. Semantics are unchanged.
    out = x.astype(np.float64).copy()
    if out.size:
        valid = ~np.isnan(out)
        idx = np.where(valid, np.arange(out.size), 0)
        np.maximum.accumulate(idx, out=idx)
        # Positions before the first real value have no source to carry
        # forward; they stay NaN here and are backfilled below.
        carried = out[idx]
        first = int(np.argmax(valid)) if valid.any() else out.size
        carried[:first] = np.nan
        out = carried
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

    # --- special-day structure -------------------------------------------
    # A single is_holiday flag is not enough. The published weakness of
    # covariate-informed TSFMs on transmission-level load is precisely that they
    # miss the *indirect* effects around holidays — bridge days, long weekends,
    # the run-up to a holiday — not the holiday itself (arXiv 2607.15705). The
    # load-forecasting literature reports materially lower error when bridge and
    # proximity days are encoded separately rather than lumped together.
    #
    # All of these are pure functions of the timestamp, so they are legitimately
    # known arbitrarily far ahead and carry no leakage risk.
    day_ord = ts.astype("datetime64[D]").astype(np.int64)
    uniq_days, inv = np.unique(day_ord, return_inverse=True)
    d_hol = np.zeros(len(uniq_days), dtype=bool)
    d_wknd = np.zeros(len(uniq_days), dtype=bool)
    for i, dnum in enumerate(uniq_days):
        dt = np.datetime64(int(dnum), "D").astype("datetime64[s]").astype("O")
        d_hol[i] = date(dt.year, dt.month, dt.day) in US_HOLIDAYS
        d_wknd[i] = dt.weekday() >= 5

    off = d_hol | d_wknd                      # "not a normal working day"
    prev_off = np.concatenate([[False], off[:-1]])
    next_off = np.concatenate([off[1:], [False]])
    # A bridge day is a working day wedged between two non-working days.
    d_bridge = (~off) & prev_off & next_off

    # Signed distance in days to the nearest holiday, clipped to a week. Negative
    # before, positive after, so the model can learn asymmetric run-up/run-down.
    hol_idx = np.flatnonzero(d_hol)
    if hol_idx.size:
        pos = np.arange(len(uniq_days))
        nearest = hol_idx[np.clip(np.searchsorted(hol_idx, pos), 0, hol_idx.size - 1)]
        prev_h = hol_idx[np.clip(np.searchsorted(hol_idx, pos) - 1, 0, hol_idx.size - 1)]
        d_next = nearest - pos
        d_prev = pos - prev_h
        signed = np.where(d_next <= d_prev, -d_next, d_prev).astype(np.float32)
    else:
        signed = np.full(len(uniq_days), 7.0, dtype=np.float32)
    d_prox = np.clip(signed, -7, 7) / 7.0

    two_pi = 2 * np.pi
    return {
        "hour_sin": np.sin(two_pi * hour / 24).astype(np.float32),
        "hour_cos": np.cos(two_pi * hour / 24).astype(np.float32),
        "dow_sin":  np.sin(two_pi * dow / 7).astype(np.float32),
        "dow_cos":  np.cos(two_pi * dow / 7).astype(np.float32),
        "is_weekend": weekend,
        "is_holiday": holiday,
        **({"is_bridge": d_bridge[inv].astype(np.float32),
            "hol_prox": d_prox[inv].astype(np.float32)}
           if SPECIAL_DAY_FEATURES else {}),
    }


FUTURE_MODES = ("persistence", "lag24", "forecast", "forecast_full",
                "analysis_only", "none", "oracle", "oracle_om")

# Modes whose PAST temperature channel comes from Open-Meteo rather than ASOS.
# "forecast" and its control must share the same past channel, otherwise a
# comparison between them conflates two changes: a new future channel and a
# new past one.
_OPENMETEO_PAST_MODES = frozenset({"forecast", "forecast_full",
                                  "analysis_only", "oracle_om"})

# Extra forecast channels beyond temperature: causal proxies for the realized
# wind/solar generation this repo used to leak.
#
# `temp_spread` (GFS/ICON/ECMWF disagreement) is deliberately NOT in this tuple.
# It was measured on the 7 RTOs and does nothing: spread alone scores 0.5044 vs
# 0.5043 for temperature alone, and adding it to the renewables channels makes
# them slightly worse (0.4898 vs 0.4888). It did not improve interval coverage
# either, which was the entire motivation. The likely reason is that Chronos-2
# was never trained to read an uncertainty channel, and conformal calibration
# already handles interval width. Still collected by the scraper, so it can be
# revisited with a model trained to use it.
FCST_EXTRA_KEYS = ("rad_fcst", "wind_fcst")

# Toggle for the special-day features so an A/B against the previous schema is
# possible; the covariate set changes, so results are not comparable across it.
# Env-driven so a control run needs no file edit: SURGE_SPECIAL_DAYS=0.
# Default OFF: measured slightly WORSE for both the fine-tune (0.5625 vs 0.5604)
# and zero-shot Chronos-2 (0.5726 vs 0.5713) on the 7 RTOs at a 24 h horizon.
# The documented TSFM weakness on special events (arXiv 2607.15705) was at
# *longer* horizons on German TSO load, where school-holiday and industrial
# effects are stronger; it does not transfer to US RTOs day-ahead. Kept behind
# the flag because it may still pay off at longer horizons or with a retrain.
SPECIAL_DAY_FEATURES = os.environ.get("SURGE_SPECIAL_DAYS", "0") != "0"

_BASE_CALENDAR_KEYS = ("hour_sin", "hour_cos", "dow_sin", "dow_cos",
                       "is_weekend", "is_holiday")
_SPECIAL_DAY_KEYS = ("is_bridge", "hol_prox")
CALENDAR_KEYS = (_BASE_CALENDAR_KEYS + _SPECIAL_DAY_KEYS
                 if SPECIAL_DAY_FEATURES else _BASE_CALENDAR_KEYS)


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
    elif mode == "forecast":
        # A real archived day-ahead NWP forecast. Causal: the value for hour t
        # was issued ~24h before t. See surge.scrapers.openmeteo.
        policy["temp_c"] = "forecast"
        for c in gen_cols:
            policy[c] = "past_only"
    elif mode == "forecast_full":
        # Temperature forecast plus forecast irradiance, 100 m wind and the
        # cross-model temperature spread. All are genuine day-ahead forecasts.
        policy["temp_c"] = "forecast"
        for k in FCST_EXTRA_KEYS:
            policy[k] = "fcst_extra"
        for c in gen_cols:
            policy[c] = "past_only"
    elif mode == "oracle_om":
        # Perfect foresight of the SAME Open-Meteo channel the forecast mode
        # uses. This is the meaningful ceiling for that configuration: the old
        # "oracle" was measured against ASOS, which is zero-filled for 46 of 53
        # BAs, so it was never a valid upper bound for a populated channel.
        # Leaks by construction; the causal guard will reject it, which is why
        # it can only be run through run_c2, never through research_eval.
        policy["temp_c"] = "oracle"
        for c_ in gen_cols:
            policy[c_] = "past_only"
    elif mode == "analysis_only":
        # Control for "forecast": identical past channel (Open-Meteo analysis),
        # no future temperature at all. The difference between the two is
        # attributable solely to the forecast covariate.
        policy["temp_c"] = "past_only"
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
    # (T,) archived day-ahead FORECAST temperature. Deliberately NOT a member of
    # `covariates`: it is not a realized observation, so the causal guard — which
    # perturbs actuals at and after the origin — must not treat it as one. Its
    # value for hour t was issued ~24 h before t, so slicing it over a horizon of
    # 24 h or less reads only information available at forecast time.
    temp_fcst: np.ndarray | None = None
    # Additional forecast channels, keyed by covariate name. Held outside
    # `covariates` for the same reason as temp_fcst: they are forecasts, not
    # realized observations, so the causal guard must not treat them as actuals.
    fcst_extra: dict[str, np.ndarray] | None = None
    # Lead time of the forecast channels, in hours. The guard cannot validate a
    # forecast channel (it is not a realized series), so causality is enforced
    # here instead: a channel forecast N hours ahead may only be served over a
    # horizon of at most N hours.
    fcst_lead_hours: int = 0
    # First index with real forecast coverage. Training must not sample before
    # this, or it falls back to observed temperature and recreates exactly the
    # train/serve mismatch this channel exists to remove.
    fcst_start: int = 0

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

    def _assert_lead(self, horizon: int, key: str) -> None:
        """A forecast issued N hours ahead cannot cover a horizon longer than N.

        Without this, evaluating at horizon=48 against a `previous_day1` channel
        (24 h lead) would quietly serve values that were not yet forecast at the
        origin — a leak the causal guard cannot see, because a forecast series is
        not a realized one.
        """
        if self.fcst_lead_hours and horizon > self.fcst_lead_hours:
            raise ValueError(
                f"{self.ba}: covariate {key!r} has a {self.fcst_lead_hours}h forecast "
                f"lead but the horizon is {horizon}h. Re-fetch with "
                f"--lead-days {int(np.ceil(horizon / 24))} or shorten the horizon."
            )

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
            elif pol == "fcst_extra":
                self._assert_lead(horizon, k)
                if not self.fcst_extra or k not in self.fcst_extra:
                    raise ValueError(
                        f"{self.ba}: future_mode='forecast_full' needs {k}; re-run "
                        "scripts/backfill_weather_forecast.py")
                out[k] = self.fcst_extra[k][origin:origin + horizon]
            elif pol == "forecast":
                self._assert_lead(horizon, k)
                if self.temp_fcst is None:
                    raise ValueError(
                        f"{self.ba}: future_mode='forecast' needs weather_fcst_hourly — "
                        "run scripts/backfill_weather_forecast.py first")
                out[k] = self.temp_fcst[origin:origin + horizon]
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
        if pol == "forecast" and self.temp_fcst is not None:
            return self.temp_fcst[origin:origin + horizon]
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
    try:
        wx_fcst = (store.scan("weather_fcst_hourly", dedupe_on=["ts_utc", "ba"])
                     .filter(pl.col("ba") == ba)
                     .select("ts_utc", "temp_c_fcst", "temp_c_anal",
                             "rad_wm2_fcst", "wind100_fcst", "temp_spread_c",
                             "lead_hours")
                     .sort("ts_utc")
                     .collect())
    except Exception:
        wx_fcst = None
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
            else:
                # Zero-fill so every BA exposes the same covariate keys.
                # Chronos-2's fit() rejects heterogeneous key sets across tasks,
                # and a uniform schema is also what keeps train and serve aligned.
                covariates[col] = np.zeros(len(target), dtype=np.float32)
            gen_cols.append(col)

    # Align the archived forecast onto this BA's hourly index. Hours before the
    # archive begins (GFS 2m temp starts ~2021-03) fall back to observed
    # temperature purely so the array is well-formed; `fcst_start` marks where
    # real forecast coverage begins and training must not sample before it.
    temp_fcst = None
    fcst_extra = None
    fcst_start = 0
    fcst_lead_hours = 0
    if wx_fcst is not None and wx_fcst.height:
        src_ts = wx_fcst["ts_utc"].to_numpy()
        src_v = wx_fcst["temp_c_fcst"].to_numpy().astype(np.float64)
        src_anal = wx_fcst["temp_c_anal"].to_numpy().astype(np.float64)
        idx = np.clip(np.searchsorted(src_ts, ts), 0, len(src_ts) - 1)
        hit = src_ts[idx] == ts
        aligned = np.where(hit, src_v[idx], np.nan)
        aligned_anal = np.where(hit, src_anal[idx], np.nan)
        covered = np.flatnonzero(~np.isnan(aligned))
        if covered.size:
            fcst_start = int(covered[0])
            # Interior gaps carry the last real forecast forward; the
            # pre-coverage prefix takes observed temperature and is excluded
            # from training by fcst_start.
            aligned = _ffill_np(aligned)
            aligned[:fcst_start] = temp[:fcst_start]
            temp_fcst = aligned.astype(np.float32)

            # Replace the PAST temperature channel with the analysis from the
            # same grid point as the forecast. Otherwise the model sees ASOS
            # station history and then a centroid forecast, which for PJM
            # differ by ~5 C on average — a discontinuity exactly at the
            # forecast boundary, which is the worst possible place for one.
            if future_mode in _OPENMETEO_PAST_MODES and not np.all(np.isnan(aligned_anal)):
                anal = _ffill_np(aligned_anal)
                anal[:fcst_start] = temp[:fcst_start]
                covariates["temp_c"] = anal.astype(np.float32)

            # Extra forecast channels. Each is stored twice on purpose: in
            # `covariates` so the model sees its history, and in `fcst_extra` so
            # future_at can serve it over the horizon. The guard perturbs only the
            # `covariates` copy, which is correct — these are forecasts, and their
            # causality comes from the lead time (see lead_hours), not from being
            # withheld.
            src_cols = {"rad_fcst": "rad_wm2_fcst",
                        "wind_fcst": "wind100_fcst",
                        "temp_spread": "temp_spread_c"}
            extra: dict[str, np.ndarray] = {}
            for name, col in src_cols.items():
                if col not in wx_fcst.columns:
                    continue
                raw = wx_fcst[col].to_numpy().astype(np.float64)
                al = np.where(hit, raw[idx], np.nan)
                if np.all(np.isnan(al)):
                    continue
                al = _ffill_np(al)
                al[:fcst_start] = al[fcst_start] if fcst_start < len(al) else 0.0
                arr = al.astype(np.float32)
                extra[name] = arr
                covariates[name] = arr
            if extra:
                fcst_extra = extra
            if "lead_hours" in wx_fcst.columns:
                lead = wx_fcst["lead_hours"].drop_nulls()
                if len(lead):
                    fcst_lead_hours = int(lead.min())

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
        temp_fcst=temp_fcst, fcst_extra=fcst_extra, fcst_start=fcst_start,
        fcst_lead_hours=fcst_lead_hours,
    )


# Number of correlated peer BAs whose PAST load is attached as a covariate.
# Their history is observable at forecast time, so this is causal; only their
# future would be off-limits. 0 disables.
N_NEIGHBORS = 8


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

        # Always emit exactly k channels, zero-filled when a BA has too few
        # usable peers, so the covariate key set is identical across all tasks.
        for rank in range(1, k + 1):
            name = f"nbr{rank}_load"
            if rank <= len(scored):
                src = scored[rank - 1][1]
                bd.covariates[name] = aligned[(tgt, src)].astype(np.float32)
            else:
                bd.covariates[name] = np.zeros(len(bd.target), dtype=np.float32)
            bd.future_policy[name] = "past_only"


def load_multi_ba(bas: list[str], *, with_gen: bool = True,
                  future_mode: str = "persistence") -> dict[str, BAData]:
    out = {ba: _join_ba(ba, with_gen=with_gen, future_mode=future_mode)
           for ba in bas}
    _attach_neighbors(out, N_NEIGHBORS)
    return out
