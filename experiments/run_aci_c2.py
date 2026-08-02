"""Adaptive-conformal ablation on the 2024 validation lane.

Validation-only, like ``run_conformal_c2``: the 2025 lane is never read. This
runner reuses that module's audited loading, alignment, prediction, split and
scoring path, so the only variable under test is how the quantile level is
chosen.

Candidates are the shipped fixed-level calibration and adaptive calibration
over a gamma grid, evaluated at both alpha scopes. Selection uses the first
chronological half of the common eligible origins; the selected candidate alone
is reported on the untouched later half.
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
from experiments.features import AvailabilityMode, load_multi_ba
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
    _partition_masks,
    _predict_ba,
    _validate_origin_alignment,
)
from surge.model_loader import load_chronos2
from surge.verification import MATURITY_HOURS, OUTCOME_POLICY_VERSION

DEFAULT_GAMMAS = (0.002, 0.005, 0.01, 0.02, 0.05)
DEFAULT_WINDOWS = (56, 168)
DEFAULT_COVERAGE_TOLERANCE = 0.02


def _label(kind: str, *, scope: str = "", gamma: float = 0.0, window: int = 0) -> str:
    if kind == "fixed":
        return f"fixed/per-rto/w{window}"
    return f"aci/{scope}/g{gamma}/w{window}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="amazon/chronos-2")
    parser.add_argument("--model-revision", default=PINNED_CHRONOS2_REVISION)
    parser.add_argument("--bas", nargs="+", default=list(RTO_BAS))
    parser.add_argument("--context", type=int, default=2_048)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--gammas", nargs="+", type=float, default=list(DEFAULT_GAMMAS))
    parser.add_argument("--min-history", type=int, default=28)
    parser.add_argument("--coverage", type=float, default=TARGET_COVERAGE)
    parser.add_argument(
        "--coverage-tolerance", type=float, default=DEFAULT_COVERAGE_TOLERANCE
    )
    parser.add_argument("--selection-fraction", type=float, default=0.5)
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
    if not 0 < args.selection_fraction < 1:
        parser.error("--selection-fraction must be in (0, 1)")
    if any(gamma <= 0 or gamma >= 1 for gamma in args.gammas):
        parser.error("every --gammas value must be in (0, 1)")

    verify_code_checkout(Path(__file__).resolve().parents[1], args.code_revision)
    data_root_value = os.environ.get("SURGE_DATA_DIR")
    if not data_root_value:
        parser.error("SURGE_DATA_DIR is required to bind the calibration data snapshot")
    try:
        verify_data_snapshot_manifest(Path(data_root_value), args.data_snapshot_sha256)
    except ValueError as exc:
        parser.error(str(exc))

    # The held-out lane is excluded in the query itself, not by the scheduler.
    data = load_multi_ba(
        args.bas,
        with_gen=False,
        availability_mode=AvailabilityMode.RETROSPECTIVE_FINAL,
        valid_before=datetime(2025, 1, 1, tzinfo=UTC),
    )
    schedules = _validate_origin_alignment(
        data, args.bas, context=args.context, horizon=args.horizon
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
    runs: dict[str, CandidateRun] = {}
    for window in args.windows:
        runs[_label("fixed", window=window)] = CandidateRun(
            pooled=False,
            window=window,
            calibrated=rolling_conformalize(
                lower, median, upper, truth, window=window, pooled_series=False, **shared
            ),
        )
    for window in args.windows:
        for gamma in args.gammas:
            for scope in ALPHA_SCOPES:
                tag = "lead" if scope == "per-lead" else "series"
                runs[_label("aci", scope=tag, gamma=gamma, window=window)] = CandidateRun(
                    pooled=False,
                    window=window,
                    calibrated=aci_conformalize(
                        lower,
                        median,
                        upper,
                        truth,
                        window=window,
                        gamma=gamma,
                        alpha_scope=scope,
                        **shared,
                    ),
                )

    common = _common_eligible(list(runs.values()))
    selection_mask, outer_mask, split_index = _partition_masks(
        common, selection_fraction=args.selection_fraction
    )

    def summarize(run: CandidateRun, mask: np.ndarray) -> dict[str, Any]:
        return _candidate_summary(
            bas=args.bas,
            lower=lower,
            median=median,
            upper=upper,
            truth=truth,
            run=run,
            mask=mask,
            target_coverage=args.coverage,
            coverage_tolerance=args.coverage_tolerance,
        )

    inner = {label: summarize(run, selection_mask) for label, run in runs.items()}
    feasible = [
        label for label, summary in inner.items() if summary["coverage_constraint_satisfied"]
    ]
    pool = feasible or list(inner)
    rule_applied = (
        "predeclared-minimum-macro-relative-interval-score-within-coverage-tolerance"
        if feasible
        else "fallback-minimum-relative-score-no-candidate-met-coverage"
    )
    selected = min(pool, key=lambda label: inner[label]["macro_relative_interval_score"])

    selection_start, selection_end = _eligible_origin_range(origins, selection_mask, truth)
    outer_start, outer_end = _eligible_origin_range(origins, outer_mask, truth)
    output = {
        "schema_version": 1,
        "protocol": "validation-only-adaptive-conformal-ablation",
        "availability_mode": AvailabilityMode.RETROSPECTIVE_FINAL.value,
        "held_out_test_lane": "not-opened",
        "point_in_time_replay": False,
        "outcome_timing_policy_reference": OUTCOME_POLICY_VERSION,
        "outcome_delay_hours": MATURITY_HOURS,
        "score_availability_rule": "prior forecast last target +72h <= current origin",
        "negative_adjustment_policy": "clamp-to-zero-no-interval-shrink",
        "bas": list(args.bas),
        "model": args.model,
        "model_revision": args.model_revision,
        "code_revision": args.code_revision,
        "data_snapshot_sha256": args.data_snapshot_sha256,
        "context": args.context,
        "horizon": args.horizon,
        "batch_size": args.batch_size,
        "target_coverage": args.coverage,
        "coverage_tolerance": args.coverage_tolerance,
        "min_history_mature_origins": args.min_history,
        "windows": list(args.windows),
        "gammas": list(args.gammas),
        "alpha_scopes": list(ALPHA_SCOPES),
        "origin_count": len(origins),
        "origin_start_utc": str(origins[0]),
        "origin_end_utc": str(origins[-1]),
        "research_basis": [
            {
                "id": "arXiv:2605.08422v1",
                "title": (
                    "Rolling-Origin Conformal Prediction under Local Stationarity "
                    "and Weak Dependence"
                ),
                "url": "https://arxiv.org/abs/2605.08422v1",
                "relationship": "fixed-level baseline this ablation is measured against",
            },
            {
                "id": "arXiv:2604.13253v1",
                "title": (
                    "Bias-Corrected Adaptive Conformal Inference for Multi-Horizon "
                    "Time Series Forecasting"
                ),
                "url": "https://arxiv.org/abs/2604.13253v1",
                "relationship": (
                    "source of the online level update; its EWMA recentering is "
                    "deliberately not implemented because Surge may not move an "
                    "interval off p50"
                ),
            },
        ],
        "versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
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
        "selection": {
            "chronological_fraction": args.selection_fraction,
            "split_origin_index": split_index,
            "scored_origin_start_utc": selection_start,
            "scored_origin_end_utc": selection_end,
            "rule_predeclared": (
                "minimum equal-BA macro relative interval score among candidates with "
                "every BA within coverage tolerance"
            ),
            "rule_applied": rule_applied,
            "feasible_candidate_count": len(feasible),
            "candidate_count": len(inner),
            "selected": selected,
        },
        "inner_selection": inner,
        "outer_validation": {
            "used_for_selection": False,
            "scored_origin_start_utc": outer_start,
            "scored_origin_end_utc": outer_end,
            "result": summarize(runs[selected], outer_mask),
        },
    }
    payload = json.dumps(output, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(f".{args.out.name}.tmp-{os.getpid()}")
        temporary.write_text(payload + "\n", encoding="utf-8")
        os.replace(temporary, args.out)
    print("ACI_ABLATION_RESULT:", json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
