"""Score one calibration policy on the 2024 validation lane.

Autonomous-research harness. Reads cached forecast arrays so an experiment costs
seconds instead of a GPU pass, then prints a single scalar:

    METRIC: <equal-BA macro interval score>

The metric is the Winkler interval score at the 80% level (Gneiting & Raftery
2007, eq. 43), averaged with equal weight per RTO so PJM and MISO cannot drown
the smaller balancing authorities. It is a strictly proper scoring rule: it
penalizes width and misses together, so it cannot be gamed by widening (as
coverage alone can) or by collapsing to a point forecast (as width alone can).

Coverage is reported alongside but is not the objective. Optimizing coverage
directly rewards the degenerate policies that a proper score rejects.

The 2025+ locked lane is never read: the loader cuts it out of the query.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.aci import (
    DEFAULT_ALPHA_SCOPE,
    DEFAULT_GAMMA,
    DEFAULT_MIN_HISTORY,
    DEFAULT_MIN_WIDTH_FRACTION,
    DEFAULT_WINDOW,
    aci_conformalize,
)
from experiments.run_conformal_c2 import (
    RTO_BAS,
    CandidateRun,
    _candidate_summary,
    _common_eligible,
    _partition_masks,
)
from surge.verification import MATURITY_HOURS

TARGET_COVERAGE = 0.8
COVERAGE_TOLERANCE = 0.02


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--alpha-scope", default=DEFAULT_ALPHA_SCOPE)
    parser.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    parser.add_argument(
        "--min-width-fraction", type=float, default=DEFAULT_MIN_WIDTH_FRACTION
    )
    parser.add_argument("--selection-fraction", type=float, default=0.5)
    return parser


def main() -> None:
    args = _parser().parse_args()
    blob = np.load(args.cache, allow_pickle=False)
    bas = [str(value) for value in blob["bas"]]
    if bas != list(RTO_BAS):
        raise ValueError(f"cache holds {bas}, expected the seven RTOs")
    origins = blob["origins"].astype("datetime64[us]")
    lower, median = blob["lower"], blob["median"]
    upper, truth = blob["upper"], blob["truth"]

    calibrated = aci_conformalize(
        lower,
        median,
        upper,
        truth,
        origin_times_utc=origins,
        outcome_delay_hours=MATURITY_HOURS,
        window=args.window,
        min_history=args.min_history,
        coverage=TARGET_COVERAGE,
        gamma=args.gamma,
        alpha_scope=args.alpha_scope,
        min_width_fraction=args.min_width_fraction,
    )
    run = CandidateRun(pooled=False, window=args.window, calibrated=calibrated)
    _, outer, _ = _partition_masks(
        _common_eligible([run]), selection_fraction=args.selection_fraction
    )
    summary = _candidate_summary(
        bas=bas,
        lower=lower,
        median=median,
        upper=upper,
        truth=truth,
        run=run,
        mask=outer,
        target_coverage=TARGET_COVERAGE,
        coverage_tolerance=COVERAGE_TOLERANCE,
    )

    per_ba = summary["per_ba"]
    macro_interval_score = float(
        np.mean([metrics["calibrated"]["interval_score"] for metrics in per_ba.values()])
    )
    baseline_macro = float(
        np.mean([metrics["baseline"]["interval_score"] for metrics in per_ba.values()])
    )
    report = {
        "macro_interval_score": round(macro_interval_score, 4),
        "uncalibrated_macro_interval_score": round(baseline_macro, 4),
        "coverage": summary["calibrated"]["coverage"],
        "max_per_ba_coverage_error": summary["max_per_ba_coverage_error"],
        "mean_width": summary["calibrated"]["mean_width"],
        "n_points": summary["calibrated"]["n_points"],
    }
    print("DETAIL:", json.dumps(report, sort_keys=True), flush=True)
    print(f"METRIC: {macro_interval_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
