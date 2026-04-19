"""Fine-tune Chronos-2 with covariates on multi-BA load.

Uses Chronos-2's official `.fit()` API. For each BA we build one training
task (one dict) containing:
  target            full train series
  past_covariates   temp + calendar, full train series
  future_covariates same keys, full train series
    (values are read by the trainer to identify known-future features;
     the trainer slices windows internally).

For validation, we pass the same structure restricted to train+val range
so windows sampled at the train/val boundary are scored.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from chronos import BaseChronosPipeline
from experiments.features import load_multi_ba
from surge import bas as _bas


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _task(bd, start: int, end: int) -> dict:
    past = {k: v[start:end].astype(np.float32) for k, v in bd.covariates.items()}
    future = {k: np.array([], dtype=np.float32) for k in bd.future_keys}
    return {
        "target": bd.target[start:end].astype(np.float32),
        "past_covariates": past,
        "future_covariates": future,   # empty values, keys only (per Chronos-2 docs)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="amazon/chronos-2")
    # Default: every BA with a demand series (see surge.bas). Override with
    # --bas for targeted experiments or v1 reproduction (--bas PJM CISO ERCO
    # MISO NYIS ISNE SWPP).
    ap.add_argument("--bas", nargs="+", default=_bas.demand_codes())
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--mode", choices=["full", "lora"], default="lora")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--num-steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    print(f"[args] {vars(args)}", flush=True)
    set_seed(args.seed)

    bas = load_multi_ba(args.bas)
    print(f"[data] loaded BAs: {list(bas)}", flush=True)

    train_inputs = [_task(bd, 0, bd.train_end) for bd in bas.values()]
    val_inputs   = [_task(bd, 0, bd.val_end)   for bd in bas.values()]
    print(f"[data] train tasks: {len(train_inputs)} | val tasks: {len(val_inputs)}", flush=True)

    pipe = BaseChronosPipeline.from_pretrained(args.base, device_map="cuda",
                                               torch_dtype=torch.bfloat16)
    print(f"[model] chronos-2 loaded, params: "
          f"{sum(p.numel() for p in pipe.model.parameters())/1e6:.1f}M", flush=True)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ft_pipe = pipe.fit(
        inputs=train_inputs,
        validation_inputs=val_inputs,
        prediction_length=args.horizon,
        context_length=args.context,
        finetune_mode=args.mode,
        learning_rate=args.lr,
        num_steps=args.num_steps,
        batch_size=args.batch,
        output_dir=str(out_dir),
        finetuned_ckpt_name="best",
        disable_data_parallel=True,
    )
    elapsed = time.time() - t0

    # Promote the best checkpoint directory to the out root for eval.
    best = out_dir / "best"
    if best.exists():
        # Don't move — just note the path; eval code can consume it directly.
        pass

    print("FINETUNE_DONE:", json.dumps({
        "wall_s": round(elapsed, 1),
        "out": str(out_dir),
        "best": str(best),
    }))


if __name__ == "__main__":
    main()
