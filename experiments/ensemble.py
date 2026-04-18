"""Ensemble of Chronos-2 variants. Quantile-averages multiple pipelines
across the same rolling windows, then scores the ensemble.
"""
from __future__ import annotations

import json
import math
import sys
import time
from typing import Iterable

import numpy as np
import torch

from chronos import BaseChronosPipeline
from experiments.features import load_multi_ba
from experiments.eval_c2 import rolling_eval_c2


def ensemble_forecast(pipes, bas, on, context, horizon, batch_size,
                      q_levels: Iterable[float] = (0.1, 0.5, 0.9)):
    """Produce ensemble forecasts by averaging each model's quantiles."""
    q_levels = list(q_levels)
    per_ba = {}
    all_win_mase = []
    for ba, bd in bas.items():
        start = bd.train_end if on == "val" else bd.val_end
        end = bd.val_end if on == "val" else len(bd.target)
        origins = [o for o in range(start, end - horizon + 1, horizon)
                   if o - context >= 0]
        if not origins:
            continue

        truths_list = []
        all_means = np.zeros((len(origins), horizon), dtype=np.float32)
        all_quants = np.zeros((len(origins), horizon, len(q_levels)), dtype=np.float32)

        for pipe in pipes:
            for i in range(0, len(origins), batch_size):
                batch_origins = origins[i:i + batch_size]
                tasks = []
                for o in batch_origins:
                    past = {k: v[o - context:o] for k, v in bd.covariates.items()}
                    future = {k: bd.covariates[k][o:o + horizon] for k in bd.future_keys}
                    tasks.append({
                        "target": bd.target[o - context:o].astype(np.float32),
                        "past_covariates": past,
                        "future_covariates": future,
                    })
                    if len(pipes) == 1 or pipe is pipes[0]:
                        truths_list.append(bd.target[o:o + horizon].astype(np.float32))
                quants_list, means_list = pipe.predict_quantiles(
                    tasks, prediction_length=horizon, quantile_levels=q_levels,
                    batch_size=len(tasks),
                )
                q = torch.stack([x.squeeze(0) for x in quants_list]).float().cpu().numpy()
                m = torch.stack([x.squeeze(0) for x in means_list]).float().cpu().numpy()
                all_means[i:i + len(batch_origins)] += m / len(pipes)
                all_quants[i:i + len(batch_origins)] += q / len(pipes)

        truths = np.stack(truths_list)
        diff = truths - all_means
        valid = ~np.isnan(truths)
        abs_err = np.abs(diff)[valid]
        sq_err = (diff * diff)[valid]

        pb_total = 0.0
        for qi_idx, tau in enumerate(q_levels):
            qi = all_quants[..., qi_idx]
            pb = np.where(truths >= qi, tau * (truths - qi), (1 - tau) * (qi - truths))
            pb_total += pb[valid].mean()
        crps = float(pb_total / len(q_levels))

        low_i, high_i = q_levels.index(min(q_levels)), q_levels.index(max(q_levels))
        pi = q_levels[high_i] - q_levels[low_i]
        cov = float(((truths >= all_quants[..., low_i]) &
                     (truths <= all_quants[..., high_i]))[valid].mean())

        mae = float(abs_err.mean())
        rmse = float(math.sqrt(sq_err.mean()))
        per_ba[ba] = {
            "mae": mae, "rmse": rmse, "mase": mae / bd.denom_mae,
            "crps": crps, f"cov_pi{int(pi * 100)}": cov,
            "n_windows": len(origins), "n_points": int(len(abs_err)),
        }
        win_mae = np.where(valid, np.abs(diff), 0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
        all_win_mase.append(win_mae / bd.denom_mae)

    macro = {k: float(np.mean([v[k] for v in per_ba.values()]))
             for k in ("mae", "rmse", "mase", "crps")}
    cov_keys = {k for v in per_ba.values() for k in v if k.startswith("cov_")}
    for k in cov_keys:
        macro[k] = float(np.mean([v[k] for v in per_ba.values()]))
    macro["per_ba"] = per_ba
    macro["n_bas"] = len(per_ba)

    pooled = np.concatenate(all_win_mase)
    rng = np.random.default_rng(42)
    boots = np.empty(2000)
    for b in range(2000):
        idx = rng.integers(0, len(pooled), len(pooled))
        boots[b] = pooled[idx].mean()
    macro["mase_ci_low"] = float(np.quantile(boots, 0.025))
    macro["mase_ci_high"] = float(np.quantile(boots, 0.975))
    return macro


def main() -> None:
    cfg = json.loads(sys.argv[1])
    bas = load_multi_ba(cfg["bas"], with_gen=cfg.get("with_gen", False))
    pipes = []
    for p in cfg["paths"]:
        pipe = BaseChronosPipeline.from_pretrained(p, device_map="cuda", torch_dtype=torch.bfloat16)
        pipes.append(pipe)
        print(f"[load] {p}", flush=True)
    t0 = time.time()
    m = ensemble_forecast(pipes, bas, cfg["on"], cfg["context"], cfg["horizon"], cfg.get("batch_size", 16))
    m["eval_s"] = round(time.time() - t0, 1)
    print("METRIC:", json.dumps({k: (round(v, 4) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                                  for k, v in m.items() if k != "per_ba"} | {"per_ba": m.get("per_ba")}))


if __name__ == "__main__":
    main()
