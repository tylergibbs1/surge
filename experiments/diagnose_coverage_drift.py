"""Per-month interval-coverage diagnostic for one cohort year.

The calibration work left one open question it could not answer: per-RTO
coverage is stable in 2021-2022 and drifts in 2023-2024, worst for ISNE. Is that
a step change at some date, or a seasonal pattern that simply got worse?

This reports uncalibrated coverage by RTO and month, so the shape of the drift
is visible rather than averaged away. It is a diagnostic, not an evaluation: no
policy is selected and no candidate is compared.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.features import AvailabilityMode, load_multi_ba
from experiments.overfit import verify_code_checkout, verify_data_snapshot_manifest
from experiments.run_aci_replication_c2 import aligned_cohort, validate_cohort_year
from experiments.run_conformal_c2 import (
    PINNED_CHRONOS2_REVISION,
    RTO_BAS,
    _align_predictions,
    _predict_ba,
)
from surge.features.splits import ACTIVE as ACTIVE_SPLIT
from surge.model_loader import load_chronos2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-year", type=int, required=True)
    parser.add_argument("--model", default="amazon/chronos-2")
    parser.add_argument("--model-revision", default=PINNED_CHRONOS2_REVISION)
    parser.add_argument("--bas", nargs="+", default=list(RTO_BAS))
    parser.add_argument("--context", type=int, default=2_048)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="float32")
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
    try:
        validate_cohort_year(args.cohort_year)
    except ValueError as exc:
        parser.error(str(exc))

    verify_code_checkout(Path(__file__).resolve().parents[1], args.code_revision)
    data_root_value = os.environ.get("SURGE_DATA_DIR")
    if not data_root_value:
        parser.error("SURGE_DATA_DIR is required to bind the data snapshot")
    try:
        verify_data_snapshot_manifest(Path(data_root_value), args.data_snapshot_sha256)
    except ValueError as exc:
        parser.error(str(exc))

    data = load_multi_ba(
        args.bas,
        with_gen=False,
        availability_mode=AvailabilityMode.RETROSPECTIVE_FINAL,
        valid_before=datetime(ACTIVE_SPLIT.locked_test_from_year, 1, 1, tzinfo=UTC),
    )
    schedules = aligned_cohort(
        data, args.bas, year=args.cohort_year, context=args.context, horizon=args.horizon
    )

    import torch

    pipe = load_chronos2(
        args.model,
        device_map=args.device_map,
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
        revision=args.model_revision,
    )
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
    origins, lower, _median, upper, truth = _align_predictions(predictions, args.bas)

    covered = (truth >= lower) & (truth <= upper)
    finite = np.isfinite(truth)
    months = origins.astype("datetime64[M]").astype(int) % 12 + 1

    by_month: dict[str, dict[str, float | int]] = {}
    for index, ba in enumerate(args.bas):
        per_month: dict[str, float | int] = {}
        for month in range(1, 13):
            rows = months == month
            valid = finite[rows, index, :]
            if not valid.any():
                continue
            hits = covered[rows, index, :][valid]
            per_month[f"{month:02d}"] = round(float(hits.mean()), 6)
        year_valid = finite[:, index, :]
        by_month[ba] = {
            "year": round(float(covered[:, index, :][year_valid].mean()), 6),
            "months": per_month,
        }

    output = {
        "schema_version": 1,
        "protocol": "uncalibrated-coverage-diagnostic",
        "diagnostic_only": True,
        "selection_performed": False,
        "cohort_year": args.cohort_year,
        "active_split": ACTIVE_SPLIT.name,
        "nominal_coverage": 0.8,
        "bas": list(args.bas),
        "model": args.model,
        "model_revision": args.model_revision,
        "code_revision": args.code_revision,
        "data_snapshot_sha256": args.data_snapshot_sha256,
        "origin_count": len(origins),
        "runtime": {"device_map": args.device_map, "dtype": args.dtype},
        "uncalibrated_coverage": by_month,
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(f".{args.out.name}.tmp-{os.getpid()}")
        temporary.write_text(text + "\n", encoding="utf-8")
        os.replace(temporary, args.out)
    print("COVERAGE_DIAGNOSTIC:", json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
