"""Score the open baselines and the blend on the 2024 validation lane.

Publishes what a reader needs to judge Surge: a naive floor, a strong non-
foundation baseline, the foundation model, and their combination — all on
identical origins and identical target hours.

The blend weight is fitted on the first chronological half of the cohort and
reported on the untouched second half, so the reported blend is never scored on
the data that chose it.

Validation-only. The locked lane is excluded in the query itself.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.features import AvailabilityMode, load_multi_ba
from experiments.gbm_baseline import (
    BASELINE_VERSION,
    DEFAULT_PARAMS,
    HORIZON,
    aligned_training_origins,
    build_design,
    fit,
    predict_levels,
)
from experiments.run_conformal_c2 import RTO_BAS, _validation_origins
from surge.features.splits import ACTIVE as ACTIVE_SPLIT

WEIGHT_GRID = np.linspace(0.0, 1.0, 101)


def _mape(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(100 * np.mean(np.abs(pred - actual) / np.abs(actual)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chronos-cache",
        type=Path,
        required=True,
        help="npz of foundation-model forecasts on the same origins",
    )
    parser.add_argument("--bas", nargs="+", default=list(RTO_BAS))
    parser.add_argument("--context", type=int, default=2_048)
    parser.add_argument(
        "--code-revision", default=os.environ.get("SURGE_CODE_REVISION", "unknown")
    )
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    blob = np.load(args.chronos_cache, allow_pickle=False)
    cached_bas = [str(value) for value in blob["bas"]]
    if cached_bas != list(args.bas):
        raise SystemExit(f"cache holds {cached_bas}, expected {list(args.bas)}")
    cached_origins = blob["origins"].astype("datetime64[us]")
    chronos_median = blob["median"]
    truth_grid = blob["truth"]

    data = load_multi_ba(
        args.bas,
        with_gen=False,
        availability_mode=AvailabilityMode.RETROSPECTIVE_FINAL,
        valid_before=datetime(ACTIVE_SPLIT.locked_test_from_year, 1, 1, tzinfo=UTC),
    )

    per_ba: dict[str, dict[str, float]] = {}
    for position, ba in enumerate(args.bas):
        series = data[ba]
        target = series.target.astype(np.float64)
        temp = series.covariates["temp_c"].astype(np.float64)
        evaluation_origins = _validation_origins(
            series, context=args.context, horizon=HORIZON
        )
        if len(evaluation_origins) != len(cached_origins):
            raise SystemExit(f"{ba}: cache and evaluation schedules differ")
        training_origins = aligned_training_origins(
            evaluation_origins=evaluation_origins,
            earliest=args.context,
            before=series.train_end,
        )
        model = fit(build_design(target, temp, series.ts_utc, training_origins))

        gbm_grid = np.full((len(evaluation_origins), HORIZON), np.nan)
        for row, origin in enumerate(evaluation_origins):
            design = build_design(target, temp, series.ts_utc, [origin])
            levels = predict_levels(model, design)
            gbm_grid[row, : len(levels)] = levels

        chronos = chronos_median[:, position, :]
        truth = truth_grid[:, position, :]
        naive = np.full_like(truth, np.nan)
        for row, origin in enumerate(evaluation_origins):
            for step in range(HORIZON):
                index = origin + step - 24
                if index >= 0:
                    naive[row, step] = target[index]

        scored = (
            np.isfinite(truth)
            & np.isfinite(gbm_grid)
            & np.isfinite(naive)
            & (np.abs(truth) > 0)
        )
        split = len(evaluation_origins) // 2
        inner = np.zeros_like(scored)
        inner[:split] = scored[:split]
        outer = np.zeros_like(scored)
        outer[split:] = scored[split:]

        losses = [
            np.mean(
                np.abs((w * chronos[inner] + (1 - w) * gbm_grid[inner]) - truth[inner])
            )
            for w in WEIGHT_GRID
        ]
        weight = float(WEIGHT_GRID[int(np.argmin(losses))])
        blended = weight * chronos[outer] + (1 - weight) * gbm_grid[outer]

        per_ba[ba] = {
            "seasonal_naive_24h": round(_mape(naive[outer], truth[outer]), 4),
            "gbm_baseline": round(_mape(gbm_grid[outer], truth[outer]), 4),
            "chronos2_zero_shot": round(_mape(chronos[outer], truth[outer]), 4),
            "blend": round(_mape(blended, truth[outer]), 4),
            "blend_weight_on_chronos": weight,
            "scored_points": int(outer.sum()),
        }
        print(f"{ba:>6} " + json.dumps(per_ba[ba], sort_keys=True), flush=True)

    def macro(key: str) -> float:
        return round(float(np.mean([v[key] for v in per_ba.values()])), 4)

    output = {
        "schema_version": 1,
        "protocol": "open-baseline-and-blend-on-untouched-outer-half",
        "baseline_version": BASELINE_VERSION,
        "baseline_params": DEFAULT_PARAMS,
        "availability_mode": AvailabilityMode.RETROSPECTIVE_FINAL.value,
        "held_out_test_lane": "not-opened",
        "active_split": ACTIVE_SPLIT.name,
        "code_revision": args.code_revision,
        "bas": list(args.bas),
        "weight_fitted_on": "first chronological half",
        "reported_on": "untouched second half",
        "per_ba": per_ba,
        "macro": {
            "seasonal_naive_24h": macro("seasonal_naive_24h"),
            "gbm_baseline": macro("gbm_baseline"),
            "chronos2_zero_shot": macro("chronos2_zero_shot"),
            "blend": macro("blend"),
        },
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(f".{args.out.name}.tmp-{os.getpid()}")
        temporary.write_text(text + "\n", encoding="utf-8")
        os.replace(temporary, args.out)
    print("BASELINE_RESULT:", json.dumps(output["macro"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
