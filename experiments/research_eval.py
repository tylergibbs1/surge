"""Scoring entrypoint for autonomous accuracy search.

    python -m experiments.research_eval <exp_name> '<config_json>'

Contract, and the reason for each rule:

* **Optimise on validation (calendar 2024), never test.** The search may run
  hundreds of evaluations; scoring on 2025 would tune straight into the test
  set and the reported number would stop meaning anything. `on` is forced to
  "val" unless `confirm: true` is passed, which is for a single final check.
* **Causality is verified, not assumed.** Every run is audited by
  `experiments.causal_guard`, which perturbs post-origin actuals and requires
  the future covariates not to move. `future_mode: "oracle"` is refused.
* **Report generalisation gaps, not just the score.** Emits the held-out-BA
  score alongside the tuned-BA score so overfitting to particular BAs is
  visible in the metric line itself.

Config keys:
    base         str   HF id or local checkpoint path
    scope        "all53" (default) | "rto7"
    future_mode  causal policy name; "oracle" is rejected
    context      int   2048
    horizon      int   24
    with_gen     bool  True
    holdout_frac float 0.25 — fraction of BAs reserved to measure generalisation
    confirm      bool  False — set once, at the end, to score the test split
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from chronos import BaseChronosPipeline

import experiments.features as F
from experiments.causal_guard import audit
from experiments.eval_c2 import rolling_eval_c2
from surge import bas as _bas

RTOS = ["PJM", "CISO", "ERCO", "MISO", "NYIS", "ISNE", "SWPP"]


def _split_bas(codes: list[str], holdout_frac: float) -> tuple[list[str], list[str]]:
    """Deterministic tune/holdout split, interleaved so both get big and small BAs."""
    if holdout_frac <= 0:
        return list(codes), []
    ordered = sorted(codes)
    stride = max(int(round(1 / holdout_frac)), 2)
    holdout = [c for i, c in enumerate(ordered) if i % stride == stride - 1]
    tune = [c for c in ordered if c not in set(holdout)]
    return tune, holdout


def main() -> None:
    exp = sys.argv[1] if len(sys.argv) > 1 else "exp"
    cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

    base = cfg.get("base", "amazon/chronos-2")
    scope = cfg.get("scope", "all53")
    future_mode = cfg.get("future_mode", "persistence")
    context = int(cfg.get("context", 2048))
    horizon = int(cfg.get("horizon", 24))
    with_gen = bool(cfg.get("with_gen", True))
    holdout_frac = float(cfg.get("holdout_frac", 0.25))
    confirm = bool(cfg.get("confirm", False))

    if future_mode == "oracle":
        print("METRIC: " + json.dumps({
            "exp": exp, "status": "rejected",
            "error": "future_mode='oracle' replays realized future values. That is "
                     "the leakage this benchmark exists to avoid; it is not a "
                     "legitimate improvement.",
        }), flush=True)
        sys.exit(2)

    codes = RTOS if scope == "rto7" else _bas.demand_codes()
    tune, holdout = _split_bas(codes, holdout_frac)
    on = "test" if confirm else "val"

    bas = F.load_multi_ba(codes, with_gen=with_gen, future_mode=future_mode)

    # Prove causality on the exact objects about to be scored.
    try:
        guard = audit(bas, horizon=horizon)
    except AssertionError as e:
        print("METRIC: " + json.dumps({
            "exp": exp, "status": "rejected", "error": f"leakage detected: {e}",
        }), flush=True)
        sys.exit(2)

    pipe = BaseChronosPipeline.from_pretrained(
        base, device_map="cuda", torch_dtype=torch.bfloat16)

    t0 = time.time()
    m = rolling_eval_c2(pipe, bas, on=on, context=context, horizon=horizon,
                        batch_size=int(cfg.get("batch_size", 32)),
                        bootstrap=int(cfg.get("bootstrap", 1000)), seed=42)
    per_ba = {b: v["mase"] for b, v in m["per_ba"].items()}

    def macro(subset):
        vals = [per_ba[b] for b in subset if b in per_ba]
        return round(float(np.mean(vals)), 4) if vals else None

    tuned_mase, holdout_mase = macro(tune), macro(holdout)
    gap = (round(holdout_mase - tuned_mase, 4)
           if tuned_mase is not None and holdout_mase is not None else None)

    out = {
        "exp": exp, "status": "ok", "split": on, "scope": scope,
        "base": base, "future_mode": future_mode, "context": context,
        "with_gen": with_gen, "causal_audit": guard,
        # `mase` is the objective the search minimises.
        "mase": round(m["mase"], 4),
        "mase_tuned_bas": tuned_mase,
        "mase_holdout_bas": holdout_mase,
        "holdout_gap": gap,
        "n_tune": len(tune), "n_holdout": len(holdout),
        "mae": round(m["mae"], 1), "crps": round(m["crps"], 2),
        "cov_pi80": round(m.get("cov_pi80", float("nan")), 4),
        "mase_ci": [round(m.get("mase_ci_low", float("nan")), 4),
                    round(m.get("mase_ci_high", float("nan")), 4)],
        "n_bas": m["n_bas"], "secs": round(time.time() - t0, 1),
    }
    print("METRIC: " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
