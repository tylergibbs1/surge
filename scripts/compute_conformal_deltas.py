"""Fit the per-BA split-conformal deltas the API serves.

    python scripts/compute_conformal_deltas.py --bas all

Needs a GPU and the parquet store (load_hourly, weather_hourly,
weather_fcst_hourly), so this runs on the remote box — not in CI, and not as
part of a deploy. Output is a small JSON artifact committed alongside the code:
src/surge/api/conformal_deltas.json, read at serve time by surge.api.conformal.

The math is not reimplemented here. It imports
`experiments.eval_c2._conformal_delta`, the same function whose deltas moved
offline cov_pi80 from 0.750 to 0.8125, so the served interval cannot drift from
the measured one. That function's `on=` argument names the split being *scored*,
not the split being fitted: `on="test"` fits on [train_end, val_end), i.e.
calendar 2024 — the validation split. Test (2025) is never read, so the interval
width is not tuned on the data it will be reported against.

Covariate defaults match the SERVING path (surge.api.forecaster), not the
research eval: real day-ahead forecast weather (temp + irradiance + 100 m wind),
calendar features, no generation channels, no peer-BA channels. A delta fitted on
a richer input describes a different predictive distribution than the API serves,
and would be the wrong width for it.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from chronos import BaseChronosPipeline

from experiments import features
from experiments.eval_c2 import _conformal_delta
from experiments.features import load_multi_ba
from surge import bas as _bas
from surge.api import forecaster

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "src" / "surge" / "api" / "conformal_deltas.json"
Q_LEVELS = [0.1, 0.5, 0.9]          # the levels forecast_ba() asks the model for


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bas", nargs="+", default=["all"],
                    help="BA codes; 'all' for every demand-reporting BA")
    ap.add_argument("--model", default=forecaster.MODEL_PATH,
                    help="checkpoint to calibrate; defaults to the served one")
    ap.add_argument("--context", type=int, default=forecaster.CONTEXT_LENGTH)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--step", type=int, default=24,
                    help="hours between calibration origins; 24 keeps them "
                         "non-overlapping, which keeps the scores near-exchangeable")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--future-mode", default="forecast_full",
                    help="see experiments.features; forecast_full is what the "
                         "API feeds (real day-ahead NWP over the horizon)")
    ap.add_argument("--with-gen", action="store_true",
                    help="attach generation covariates (the API does not)")
    ap.add_argument("--neighbors", type=int, default=0,
                    help="peer-BA load channels (the API sends none)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    codes = _bas.demand_codes() if args.bas == ["all"] else [b.upper() for b in args.bas]

    # Read as a module global inside load_multi_ba, so this is the only way to
    # turn peer channels off without duplicating the loader.
    features.N_NEIGHBORS = args.neighbors
    bas = load_multi_ba(codes, with_gen=args.with_gen, future_mode=args.future_mode)

    load_kwargs: dict = {"device_map": "cuda", "torch_dtype": torch.bfloat16}
    # Pin the revision when calibrating an HF repo id, for the same reason the
    # API does: `main` is mutable, and a delta belongs to one set of weights.
    if "/" in args.model and not args.model.startswith("/"):
        load_kwargs["revision"] = forecaster.MODEL_REVISION
    pipe = BaseChronosPipeline.from_pretrained(args.model, **load_kwargs)

    deltas: dict[str, float] = {}
    for ba, bd in bas.items():
        # on="test" == "calibrate for scoring test" == fit on val (2024).
        delta = _conformal_delta(pipe, bd, on="test", context=args.context,
                                 horizon=args.horizon, step=args.step,
                                 q_levels=Q_LEVELS, batch_size=args.batch_size)
        if delta == 0.0:
            # _conformal_delta returns exactly 0.0 when it had no usable
            # calibration windows. Indistinguishable from a real zero delta, and
            # a real zero is a no-op anyway, so leave the BA out of the artifact
            # rather than publish a width nothing was fitted on.
            print(f"[{ba}] no calibration windows; omitted", flush=True)
            continue
        deltas[ba] = round(delta, 2)
        print(f"[{ba}] delta {delta:8.2f} MW", flush=True)

    payload = {
        "fit_on": "val (calendar 2024)",
        "generated_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "model": args.model,
        "model_revision": load_kwargs.get("revision"),
        "context": args.context,
        "horizon": args.horizon,
        "step": args.step,
        "future_mode": args.future_mode,
        "with_gen": args.with_gen,
        "neighbors": args.neighbors,
        "quantile_levels": Q_LEVELS,
        "deltas_mw": dict(sorted(deltas.items())),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out} — {len(deltas)}/{len(codes)} BAs", flush=True)


if __name__ == "__main__":
    main()
