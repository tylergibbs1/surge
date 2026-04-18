"""Experiment runner. Usage: python -m experiments.run <model> <config_json>

Prints the final line as:
    METRIC: {json}

so the outer loop can grep a single line instead of parsing full stdout.
"""
from __future__ import annotations

import json
import sys
import time

from experiments import eval as eev
from experiments import models


def main() -> None:
    exp_name = sys.argv[1]
    cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

    model_name = cfg.get("model", exp_name)
    context = cfg.get("context", 168)
    horizon = cfg.get("horizon", 24)
    on = cfg.get("on", "test")
    batch_size = cfg.get("batch_size", 128)

    bas = cfg.get("bas", ["PJM"])
    split = eev.load_split(bas)
    print(f"[data] bas={list(split.bas)} denom_mae(macro)={split.denom_mae:.2f}", flush=True)

    t0 = time.time()
    forecaster = models.get(model_name, cfg)
    load_s = time.time() - t0

    bootstrap = cfg.get("bootstrap", 0)
    seed = cfg.get("seed", 0)
    t0 = time.time()
    m = eev.rolling_eval(forecaster, split, on=on, context=context, horizon=horizon,
                         batch_size=batch_size, bootstrap=bootstrap, seed=seed)
    eval_s = time.time() - t0

    out = {
        "exp": exp_name, "model": model_name, "on": on,
        "bas": list(split.bas), "context": context, "horizon": horizon,
        "load_s": round(load_s, 2), "eval_s": round(eval_s, 2),
        **{k: round(v, 4) if isinstance(v, (float, int)) and not isinstance(v, bool) else v
           for k, v in m.items() if k != "per_ba"},
    }
    if "per_ba" in m:
        out["per_ba"] = {ba: {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in stats.items()}
                         for ba, stats in m["per_ba"].items()}
    print("METRIC:", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
