"""Rolling 24h eval for Chronos-2 with covariates.

Differs from `eval.py` only in how model inputs are shaped: we pass
`{target, past_covariates, future_covariates}` dicts instead of raw arrays.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from experiments.features import BAData, load_multi_ba


def _predict_windows(pipe, bd: BAData, origins, *, context: int, horizon: int,
                     q_levels: list[float], batch_size: int):
    """Run the model over `origins`. Returns (quants (N,H,Q), truths (N,H))."""
    q_out, t_out = [], []
    for i in range(0, len(origins), batch_size):
        batch = origins[i:i + batch_size]
        tasks, truths = [], []
        for o in batch:
            tasks.append({
                "target": bd.target[o - context:o].astype(np.float32),
                "past_covariates": {k: v[o - context:o] for k, v in bd.covariates.items()},
                "future_covariates": bd.future_at(o, horizon),
            })
            truths.append(bd.target[o:o + horizon].astype(np.float32))
        quants_list, _ = pipe.predict_quantiles(
            tasks, prediction_length=horizon, quantile_levels=q_levels,
            batch_size=len(tasks))
        q_out.append(torch.stack([q.squeeze(0) for q in quants_list]).float().cpu().numpy())
        t_out.append(np.stack(truths))
    if not q_out:
        return np.empty((0, horizon, len(q_levels))), np.empty((0, horizon))
    return np.concatenate(q_out), np.concatenate(t_out)


def _conformal_delta(pipe, bd: BAData, *, on: str, context: int, horizon: int,
                     step: int, q_levels: list[float], batch_size: int) -> float:
    """Split-conformal (CQR) widening for the outer interval, fitted per BA.

    Conformity score s = max(q_lo - y, y - q_hi); the returned delta is its
    (1-alpha) empirical quantile with the finite-sample correction. Adding it to
    q_hi and subtracting from q_lo restores nominal coverage.

    Fitted on the split *before* the one being scored, so the calibration data is
    never the evaluation data: scoring test (2025) calibrates on val (2024), and
    scoring val calibrates on the tail of train.
    """
    if on == "test":
        lo, hi = bd.train_end, bd.val_end
    else:
        lo, hi = max(bd.train_end - 8760, context), bd.train_end
    origins = [o for o in range(lo, hi - horizon + 1, step) if o - context >= 0]
    if not origins:
        return 0.0

    quants, truths = _predict_windows(pipe, bd, origins, context=context,
                                      horizon=horizon, q_levels=q_levels,
                                      batch_size=batch_size)
    li, hi_i = q_levels.index(min(q_levels)), q_levels.index(max(q_levels))
    scores = np.maximum(quants[..., li] - truths, truths - quants[..., hi_i])
    scores = scores[~np.isnan(scores)]
    if scores.size == 0:
        return 0.0
    alpha = 1.0 - (max(q_levels) - min(q_levels))
    n = scores.size
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level))


def rolling_eval_c2(
    pipe,
    bas: dict[str, BAData],
    *,
    on: str = "val",
    context: int = 2048,
    horizon: int = 24,
    step: int = 24,
    quantile_levels: Iterable[float] = (0.1, 0.5, 0.9),
    batch_size: int = 16,
    bootstrap: int = 0,
    seed: int = 0,
    per_step: bool = False,
    conformal: bool = False,
) -> dict:
    q_levels = list(quantile_levels)
    per_ba: dict[str, dict[str, float]] = {}
    conformal_deltas: dict[str, float] = {}
    all_mase_windows: list[np.ndarray] = []
    per_step_abs_by_ba: dict[str, np.ndarray] = {}

    for ba, bd in bas.items():
        eval_start = bd.train_end if on == "val" else bd.val_end
        eval_end   = bd.val_end   if on == "val" else bd.test_end

        origins = [o for o in range(eval_start, eval_end - horizon + 1, step)
                   if o - context >= 0]
        if not origins:
            continue

        delta = 0.0
        if conformal:
            delta = _conformal_delta(pipe, bd, on=on, context=context,
                                     horizon=horizon, step=step,
                                     q_levels=q_levels, batch_size=batch_size)
            conformal_deltas[ba] = round(delta, 2)

        abs_err_list: list[np.ndarray] = []
        sq_err_list: list[np.ndarray] = []
        pb_lists: list[list[np.ndarray]] = [[] for _ in q_levels]
        cov_hits: list[np.ndarray] = []

        # Batch origins
        per_step_sums = np.zeros(horizon, dtype=np.float64)
        per_step_counts = np.zeros(horizon, dtype=np.int64)
        for i in range(0, len(origins), batch_size):
            batch_origins = origins[i:i + batch_size]
            tasks = []
            truths = []
            for o in batch_origins:
                past = {k: v[o - context:o] for k, v in bd.covariates.items()}
                future = bd.future_at(o, horizon)
                tasks.append({
                    "target": bd.target[o - context:o].astype(np.float32),
                    "past_covariates": past,
                    "future_covariates": future,
                })
                truths.append(bd.target[o:o + horizon].astype(np.float32))
            truths = np.stack(truths)  # (B, H)

            quants_list, means_list = pipe.predict_quantiles(
                tasks, prediction_length=horizon, quantile_levels=q_levels,
                batch_size=len(tasks),
            )
            quants = torch.stack([q.squeeze(0) for q in quants_list]).float().cpu().numpy()  # (B, H, Q)
            if delta:
                # Widen only the outer pair; the median is untouched, so MASE and
                # MAE are unaffected and only coverage/CRPS move.
                _lo_i = q_levels.index(min(q_levels))
                _hi_i = q_levels.index(max(q_levels))
                quants[..., _lo_i] -= delta
                quants[..., _hi_i] += delta
            means  = torch.stack([m.squeeze(0) for m in means_list]).float().cpu().numpy()   # (B, H)

            diff = truths - means
            valid = ~np.isnan(truths)
            abs_err_list.append(np.abs(diff)[valid])
            sq_err_list.append((diff * diff)[valid])

            # Per-step bucketing: sum |err| and count valid points per step_ahead.
            abs_err_batch = np.abs(diff)
            per_step_sums += np.where(valid, abs_err_batch, 0).sum(axis=0)
            per_step_counts += valid.sum(axis=0)
            for qi_idx, tau in enumerate(q_levels):
                qi = quants[..., qi_idx]
                pb = np.where(truths >= qi, tau * (truths - qi), (1 - tau) * (qi - truths))
                pb_lists[qi_idx].append(pb[valid])

            # PI coverage on [low, high]
            low_i  = q_levels.index(min(q_levels))
            high_i = q_levels.index(max(q_levels))
            hit = ((truths >= quants[..., low_i]) & (truths <= quants[..., high_i]))[valid]
            cov_hits.append(hit)

            # per-window abs err for bootstrap
            win_abs = np.abs(diff).mean(axis=1)
            all_mase_windows.append(win_abs / bd.denom_mae)

        abs_err = np.concatenate(abs_err_list)
        sq_err = np.concatenate(sq_err_list)
        pb_means = [float(np.concatenate(l).mean()) for l in pb_lists]

        if per_step:
            # Per-step MASE uses the BA's seasonal-naive denom.
            per_step_mae = per_step_sums / np.maximum(per_step_counts, 1)
            per_step_abs_by_ba[ba] = per_step_mae / bd.denom_mae

        mae = float(abs_err.mean())
        rmse = float(math.sqrt(sq_err.mean()))
        pi = max(q_levels) - min(q_levels)
        per_ba[ba] = {
            "mae": mae, "rmse": rmse, "mase": mae / bd.denom_mae,
            "crps": float(np.mean(pb_means)),
            f"cov_pi{int(pi*100)}": float(np.concatenate(cov_hits).mean()),
            "n_windows": len(origins), "n_points": int(len(abs_err)),
        }

    macro = {k: float(np.mean([v[k] for v in per_ba.values()]))
             for k in ("mae", "rmse", "mase", "crps")}
    cov_keys = {k for v in per_ba.values() for k in v if k.startswith("cov_")}
    for k in cov_keys:
        macro[k] = float(np.mean([v[k] for v in per_ba.values()]))
    macro["per_ba"] = per_ba
    if conformal_deltas:
        macro["conformal_deltas_mw"] = conformal_deltas
    macro["n_bas"] = len(per_ba)

    if bootstrap > 0 and all_mase_windows:
        rng = np.random.default_rng(seed)
        pooled = np.concatenate(all_mase_windows)
        boots = np.empty(bootstrap)
        for b in range(bootstrap):
            idx = rng.integers(0, len(pooled), len(pooled))
            boots[b] = pooled[idx].mean()
        macro["mase_ci_low"]  = float(np.quantile(boots, 0.025))
        macro["mase_ci_high"] = float(np.quantile(boots, 0.975))

    if per_step and per_step_abs_by_ba:
        # Macro-average per-step MASE across BAs.
        macro["per_step_mase"] = np.mean(
            np.stack(list(per_step_abs_by_ba.values())), axis=0
        ).tolist()

    return macro
