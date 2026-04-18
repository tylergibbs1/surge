"""Evaluation runner for Chronos-2 with covariates.

    python -m experiments.run_c2 <exp_name> <config_json>

Config keys:
    base          str   HF model id or local path
    bas           list[str]  BAs to evaluate on
    context       int   2048
    horizon       int   24
    on            "val" | "test"
    bootstrap     int   0 for none, else n resamples
    batch_size    int   16
"""
from __future__ import annotations

import json
import sys
import time

import torch

from chronos import BaseChronosPipeline
from experiments.features import load_multi_ba
from experiments.eval_c2 import rolling_eval_c2


def main() -> None:
    exp_name = sys.argv[1]
    cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

    base = cfg.get("base", "amazon/chronos-2")
    bas_list = cfg.get("bas", ["PJM", "CISO", "ERCO", "MISO", "NYIS", "ISNE", "SWPP"])
    context = cfg.get("context", 2048)
    horizon = cfg.get("horizon", 24)
    on = cfg.get("on", "val")
    batch_size = cfg.get("batch_size", 16)
    bootstrap = cfg.get("bootstrap", 0)
    seed = cfg.get("seed", 42)

    bas = load_multi_ba(bas_list, with_gen=cfg.get("with_gen", True))

    t0 = time.time()
    pipe = BaseChronosPipeline.from_pretrained(base, device_map="cuda", torch_dtype=torch.bfloat16)
    load_s = time.time() - t0

    t0 = time.time()
    m = rolling_eval_c2(pipe, bas, on=on, context=context, horizon=horizon,
                        batch_size=batch_size, bootstrap=bootstrap, seed=seed,
                        per_step=cfg.get("per_step", False))
    eval_s = time.time() - t0

    out = {
        "exp": exp_name, "base": base, "bas": bas_list, "on": on,
        "context": context, "horizon": horizon,
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
