"""Single-shot replication of a frozen calibration policy on an earlier cohort.

The adaptive-conformal ablation searched two candidate families against the
2024 validation cohort, so its reporting half has been inspected more than
once. This runner answers the only question that search cannot: does the frozen
winner still hold every RTO inside the coverage band on origins that no
calibration search has ever touched?

The cohort is a full earlier calendar year. Those years are training data for
the fine-tuning lane, but the calibrated model here is the pinned zero-shot
upstream base, which Surge never trains, so the cohort is genuinely unused by
this line of work. It is not the locked 2025 lane, which stays closed.

There is no selection step. The policy is supplied on the command line, run
once, and reported whatever it does.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from experiments.aci import ALPHA_SCOPES, aci_conformalize
from experiments.conformal import rolling_conformalize
from experiments.features import AvailabilityMode, BAData, load_multi_ba
from experiments.overfit import verify_code_checkout, verify_data_snapshot_manifest
from experiments.run_conformal_c2 import (
    PINNED_CHRONOS2_REVISION,
    RTO_BAS,
    TARGET_COVERAGE,
    CandidateRun,
    _align_predictions,
    _candidate_summary,
    _common_eligible,
    _eligible_origin_range,
    _predict_ba,
)
from surge.model_loader import load_chronos2
from surge.verification import MATURITY_HOURS, OUTCOME_POLICY_VERSION

DEFAULT_COVERAGE_TOLERANCE = 0.02
EARLIEST_COHORT_YEAR = 2019
LOCKED_LANE_YEAR = 2025


def validate_cohort_year(year: int) -> int:
    """Reject any cohort that would open the locked test lane."""
    if not EARLIEST_COHORT_YEAR <= year < LOCKED_LANE_YEAR:
        raise ValueError(
            f"cohort year must be in [{EARLIEST_COHORT_YEAR}, {LOCKED_LANE_YEAR}); "
            f"{LOCKED_LANE_YEAR} and later is the locked test lane"
        )
    return year


def cohort_origins(data: BAData, *, year: int, context: int, horizon: int) -> list[int]:
    """Non-overlapping origins whose whole target window lies inside ``year``."""
    years = data.ts_utc.astype("datetime64[Y]").astype(np.int64) + 1970
    inside = np.flatnonzero(years == year)
    if not len(inside):
        raise ValueError(f"{data.ba} has no {year} rows")
    start, end = int(inside[0]), int(inside[-1]) + 1
    origins = [
        origin
        for origin in range(start, end - horizon + 1, horizon)
        if origin - context >= 0
    ]
    if not origins:
        raise ValueError(f"no {year} origins for {data.ba}")
    origin_times = data.ts_utc[np.asarray(origins)].astype("datetime64[us]")
    if len(origin_times) > 1 and np.any(
        np.diff(origin_times) != np.timedelta64(horizon, "h")
    ):
        raise ValueError(f"{data.ba} {year} origins are not {horizon} hours apart")
    return origins


def aligned_cohort(
    data: dict[str, BAData],
    bas: list[str],
    *,
    year: int,
    context: int,
    horizon: int,
) -> dict[str, list[int]]:
    """Every RTO must share one exact origin schedule, as in the other lanes."""
    schedules = {
        ba: cohort_origins(data[ba], year=year, context=context, horizon=horizon)
        for ba in bas
    }
    reference_ba = bas[0]
    reference = data[reference_ba].ts_utc[np.asarray(schedules[reference_ba])].astype(
        "datetime64[us]"
    )
    mismatches = [
        f"{ba}={len(schedules[ba])} origins"
        for ba in bas[1:]
        if not np.array_equal(
            data[ba].ts_utc[np.asarray(schedules[ba])].astype("datetime64[us]"), reference
        )
    ]
    if mismatches:
        raise ValueError(
            f"{year} origin alignment is incomplete; {reference_ba}={len(reference)}; "
            + "; ".join(mismatches)
        )
    return schedules


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-year", type=int, default=2023)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--window", type=int, default=168)
    parser.add_argument("--alpha-scope", choices=list(ALPHA_SCOPES), default="per-series")
    parser.add_argument("--model", default="amazon/chronos-2")
    parser.add_argument("--model-revision", default=PINNED_CHRONOS2_REVISION)
    parser.add_argument("--bas", nargs="+", default=list(RTO_BAS))
    parser.add_argument("--context", type=int, default=2_048)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-history", type=int, default=28)
    parser.add_argument("--coverage", type=float, default=TARGET_COVERAGE)
    parser.add_argument(
        "--coverage-tolerance", type=float, default=DEFAULT_COVERAGE_TOLERANCE
    )
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--code-revision", default=os.environ.get("SURGE_CODE_REVISION", "unknown")
    )
    parser.add_argument(
        "--data-snapshot-sha256",
        default=os.environ.get("SURGE_DATA_SNAPSHOT_SHA256", "unknown"),
    )
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.horizon != 24:
        parser.error("the v0.2 calibration lane requires a 24-hour horizon")
    try:
        validate_cohort_year(args.cohort_year)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 < args.gamma < 1:
        parser.error("--gamma must be in (0, 1)")

    verify_code_checkout(Path(__file__).resolve().parents[1], args.code_revision)
    data_root_value = os.environ.get("SURGE_DATA_DIR")
    if not data_root_value:
        parser.error("SURGE_DATA_DIR is required to bind the calibration data snapshot")
    try:
        verify_data_snapshot_manifest(Path(data_root_value), args.data_snapshot_sha256)
    except ValueError as exc:
        parser.error(str(exc))

    # Cut the locked lane out of the query itself, exactly as the other lanes do.
    data = load_multi_ba(
        args.bas,
        with_gen=False,
        availability_mode=AvailabilityMode.RETROSPECTIVE_FINAL,
        valid_before=datetime(LOCKED_LANE_YEAR, 1, 1, tzinfo=UTC),
    )
    schedules = aligned_cohort(
        data, args.bas, year=args.cohort_year, context=args.context, horizon=args.horizon
    )

    import torch

    if args.device_map.startswith("cuda") and not torch.cuda.is_available():
        parser.error("--device-map requests CUDA but torch.cuda.is_available() is false")
    load_started = time.time()
    pipe = load_chronos2(
        args.model,
        device_map=args.device_map,
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
        revision=args.model_revision,
    )
    load_seconds = time.time() - load_started

    predict_started = time.time()
    predictions = {
        ba: _predict_ba(
            pipe,
            data[ba],
            origins=schedules[ba],
            context=args.context,
            horizon=args.horizon,
            batch_size=args.batch_size,
        )
        for ba in args.bas
    }
    predict_seconds = time.time() - predict_started
    origins, lower, median, upper, truth = _align_predictions(predictions, args.bas)

    shared = {
        "origin_times_utc": origins,
        "outcome_delay_hours": MATURITY_HOURS,
        "min_history": args.min_history,
        "coverage": args.coverage,
        "normalized": True,
    }
    runs = {
        "frozen-adaptive": CandidateRun(
            pooled=False,
            window=args.window,
            calibrated=aci_conformalize(
                lower,
                median,
                upper,
                truth,
                window=args.window,
                gamma=args.gamma,
                alpha_scope=args.alpha_scope,
                **shared,
            ),
        ),
        "reference-fixed": CandidateRun(
            pooled=False,
            window=args.window,
            calibrated=rolling_conformalize(
                lower, median, upper, truth, window=args.window, pooled_series=False, **shared
            ),
        ),
    }
    scored = _common_eligible(list(runs.values()))
    cohort_start, cohort_end = _eligible_origin_range(origins, scored, truth)

    results = {
        label: _candidate_summary(
            bas=args.bas,
            lower=lower,
            median=median,
            upper=upper,
            truth=truth,
            run=run,
            mask=scored,
            target_coverage=args.coverage,
            coverage_tolerance=args.coverage_tolerance,
        )
        for label, run in runs.items()
    }
    output = {
        "schema_version": 1,
        "protocol": "single-shot-replication-of-a-frozen-calibration-policy",
        "availability_mode": AvailabilityMode.RETROSPECTIVE_FINAL.value,
        "held_out_test_lane": "not-opened",
        "point_in_time_replay": False,
        "selection_performed": False,
        "outcome_timing_policy_reference": OUTCOME_POLICY_VERSION,
        "outcome_delay_hours": MATURITY_HOURS,
        "negative_adjustment_policy": "clamp-to-zero-no-interval-shrink",
        "frozen_policy": {
            "gamma": args.gamma,
            "window_mature_origins": args.window,
            "alpha_scope": args.alpha_scope,
            "min_history_mature_origins": args.min_history,
            "source": "artifacts/adaptive-conformal-validation.json",
        },
        "cohort": {
            "year": args.cohort_year,
            "origin_count": len(origins),
            "scored_origin_start_utc": cohort_start,
            "scored_origin_end_utc": cohort_end,
            "used_by_any_calibration_search": False,
            "note": (
                "training-lane years for the fine-tune, but the calibrated model is "
                "the pinned zero-shot upstream base, which Surge never trains"
            ),
        },
        "bas": list(args.bas),
        "model": args.model,
        "model_revision": args.model_revision,
        "code_revision": args.code_revision,
        "data_snapshot_sha256": args.data_snapshot_sha256,
        "context": args.context,
        "horizon": args.horizon,
        "target_coverage": args.coverage,
        "coverage_tolerance": args.coverage_tolerance,
        "versions": {"numpy": np.__version__, "torch": torch.__version__},
        "runtime": {
            "device_map": args.device_map,
            "dtype": args.dtype,
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if args.device_map.startswith("cuda") and torch.cuda.is_available()
                else None
            ),
            "load_seconds": round(load_seconds, 3),
            "predict_seconds": round(predict_seconds, 3),
        },
        "results": results,
        "replication_holds": bool(results["frozen-adaptive"]["coverage_constraint_satisfied"]),
    }
    payload: dict[str, Any] = output
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(f".{args.out.name}.tmp-{os.getpid()}")
        temporary.write_text(text + "\n", encoding="utf-8")
        os.replace(temporary, args.out)
    print("ACI_REPLICATION_RESULT:", json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
