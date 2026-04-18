"""Rolling 24h-ahead eval harness with calibration and bootstrap CIs.

Protocol:
    - Load cleaned hourly series for one or more BAs.
    - For each step in the eval window, take the preceding `context` hours as
      history and produce a `horizon`-step probabilistic forecast.
    - Compute MASE (m=24), RMSE, CRPS, PI coverage at [50, 80, 90]%.
    - Optional bootstrap CI on MASE via window-level resampling.

MASE denominator uses the *per-BA train-set* seasonal-naive (m=24) absolute
error, per Hyndman's definition. Important: this is computed on train only
so the scaler doesn't leak eval-period information.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import polars as pl

from surge import store


@dataclass
class BASplit:
    ba: str
    full: np.ndarray
    train_end: int
    val_end: int
    denom_mae: float   # train-set seasonal-naive MAE (m=24)

    @property
    def train(self) -> np.ndarray: return self.full[:self.train_end]
    @property
    def val(self)   -> np.ndarray: return self.full[self.train_end:self.val_end]
    @property
    def test(self)  -> np.ndarray: return self.full[self.val_end:]


@dataclass
class Split:
    """Multi-BA split. The single-BA convenience properties (train/val/test)
    concatenate across BAs for training-data construction."""
    bas: dict[str, BASplit] = field(default_factory=dict)

    @property
    def train(self) -> np.ndarray:
        return np.concatenate([b.train for b in self.bas.values()])

    @property
    def denom_mae(self) -> float:
        """Global denominator — mean of per-BA train MASE denominators."""
        return float(np.mean([b.denom_mae for b in self.bas.values()]))


def _ffill(x: np.ndarray) -> np.ndarray:
    out = x.copy()
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


def load_split(bas: list[str] | str = "PJM") -> Split:
    if isinstance(bas, str):
        bas = [bas]
    out: dict[str, BASplit] = {}
    for ba in bas:
        df = (store.scan("load_hourly")
                .filter(pl.col("ba") == ba)
                .sort("ts_utc")
                .collect())
        # Drop known-corrupt rows (absurd magnitudes).
        df = df.with_columns(
            pl.when(pl.col("load_mw") > 200_000)
              .then(None)
              .otherwise(pl.col("load_mw"))
              .alias("load_mw")
        )
        y = _ffill(df["load_mw"].to_numpy().astype(np.float64))

        years = df["ts_utc"].dt.year().to_numpy()
        train_end = int(np.searchsorted(years, 2024, side="left"))
        val_end   = int(np.searchsorted(years, 2025, side="left"))

        train = y[:train_end]
        denom = float(np.nanmean(np.abs(train[24:] - train[:-24])))
        out[ba] = BASplit(ba=ba, full=y, train_end=train_end, val_end=val_end,
                          denom_mae=denom)
    return Split(bas=out)


Forecaster = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]


def _collect_windows(ba_split: BASplit, on: str, context: int, horizon: int, step: int):
    full = ba_split.full
    if on == "test":
        eval_start, eval_end = ba_split.val_end, len(full)
    elif on == "val":
        eval_start, eval_end = ba_split.train_end, ba_split.val_end
    else:
        raise ValueError(on)

    contexts, targets = [], []
    origin = eval_start
    while origin + horizon <= eval_end:
        if origin - context >= 0:
            contexts.append(full[origin - context:origin])
            targets.append(full[origin:origin + horizon])
        origin += step
    return np.stack(contexts), np.stack(targets)


def _forecast_all(model: Forecaster, ctxs: np.ndarray, horizon: int,
                  q_levels: list[float], batch_size: int):
    N = len(ctxs)
    all_quants = np.empty((N, horizon, len(q_levels)), dtype=np.float32)
    all_means = np.empty((N, horizon), dtype=np.float32)

    if getattr(model, "batched", False):
        for i in range(0, N, batch_size):
            q, m = model(ctxs[i:i + batch_size], horizon)
            all_quants[i:i + q.shape[0]] = q
            all_means[i:i + m.shape[0]] = m
    else:
        for i in range(N):
            q, m = model(ctxs[i], horizon)
            all_quants[i] = q
            all_means[i] = m
    return all_quants, all_means


def _metrics_for_ba(quants: np.ndarray, means: np.ndarray, truths: np.ndarray,
                    denom_mae: float, q_levels: list[float],
                    pi_levels: Iterable[float] = (0.5, 0.8, 0.9)) -> dict[str, float]:
    diff = truths - means
    valid = ~np.isnan(truths)
    abs_err = np.abs(diff)[valid]
    sq_err = (diff * diff)[valid]

    pb_total = 0.0
    for i, tau in enumerate(q_levels):
        qi = quants[..., i]
        pb = np.where(truths >= qi, tau * (truths - qi), (1 - tau) * (qi - truths))
        pb_total += pb[valid].mean()
    crps = float(pb_total / len(q_levels))

    # PI coverage — assumes q_levels includes symmetric quantiles (e.g. 0.1, 0.5, 0.9).
    cov: dict[str, float] = {}
    low_idx = q_levels.index(min(q_levels))
    high_idx = q_levels.index(max(q_levels))
    if abs(q_levels[low_idx] + q_levels[high_idx] - 1) < 1e-9:
        pi = q_levels[high_idx] - q_levels[low_idx]
        hit = ((truths >= quants[..., low_idx]) & (truths <= quants[..., high_idx]))[valid]
        cov[f"cov_pi{int(pi*100)}"] = float(hit.mean())

    mae = float(abs_err.mean())
    rmse = float(math.sqrt(sq_err.mean()))
    return {"mae": mae, "rmse": rmse, "mase": mae / denom_mae, "crps": crps,
            "n_points": int(valid.sum()), **cov}


def rolling_eval(
    model: Forecaster,
    split: Split,
    *,
    on: str = "test",
    context: int = 168,
    horizon: int = 24,
    step: int = 24,
    quantile_levels: Iterable[float] = (0.1, 0.5, 0.9),
    batch_size: int = 128,
    bootstrap: int = 0,
    seed: int = 0,
) -> dict[str, float]:
    """Evaluate across all BAs in split. Returns macro-averaged metrics,
    plus per-BA breakdown under `per_ba`, plus bootstrap CI on MASE if >0."""
    q_levels = list(quantile_levels)
    per_ba: dict[str, dict[str, float]] = {}
    all_abs_err_by_ba: dict[str, np.ndarray] = {}

    for ba, bs in split.bas.items():
        ctxs, truths = _collect_windows(bs, on, context, horizon, step)
        quants, means = _forecast_all(model, ctxs, horizon, q_levels, batch_size)
        m = _metrics_for_ba(quants, means, truths, bs.denom_mae, q_levels)
        m["n_windows"] = len(ctxs)
        per_ba[ba] = m

        # Save per-window abs errors for bootstrap.
        diff = truths - means
        valid_mask = ~np.isnan(truths)
        window_abs_err = np.where(valid_mask, np.abs(diff), 0).sum(axis=1) / \
                         np.maximum(valid_mask.sum(axis=1), 1)
        # Scale by BA denom for MASE.
        all_abs_err_by_ba[ba] = window_abs_err / bs.denom_mae

    macro = {k: float(np.mean([v[k] for v in per_ba.values()]))
             for k in ("mae", "rmse", "mase", "crps")}
    cov_keys = {k for v in per_ba.values() for k in v if k.startswith("cov_")}
    for k in cov_keys:
        macro[k] = float(np.mean([v[k] for v in per_ba.values()]))

    out: dict = {**macro, "per_ba": per_ba, "n_bas": len(split.bas)}
    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        # Stack window-level MASE values across BAs.
        pooled = np.concatenate(list(all_abs_err_by_ba.values()))
        boots = np.empty(bootstrap)
        for b in range(bootstrap):
            idx = rng.integers(0, len(pooled), len(pooled))
            boots[b] = pooled[idx].mean()
        out["mase_ci_low"]  = float(np.quantile(boots, 0.025))
        out["mase_ci_high"] = float(np.quantile(boots, 0.975))
    return out
