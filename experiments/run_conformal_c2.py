"""Validation-only rolling conformal experiment for Chronos-2 forecasts.

The first chronological half of 2024 selects the calibration policy; the
second half reports that already-selected policy prequentially. Forecast errors
enter calibration only after the complete 24-hour target window has matured
under Surge's +72-hour outcome policy. The held-out 2025 lane is never opened.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from experiments.conformal import CalibratedIntervals, interval_metrics, rolling_conformalize
from experiments.features import BAData, load_multi_ba
from experiments.overfit import verify_code_checkout, verify_data_snapshot_manifest
from scripts.rebuild_data_snapshot import verify_snapshot
from surge.features import LOAD_V2_CORE, AvailabilityMode, build_evaluation_task
from surge.model_loader import artifact_sha256, load_chronos2
from surge.verification import MATURITY_HOURS, OUTCOME_POLICY_VERSION

RTO_BAS = ("PJM", "CISO", "ERCO", "MISO", "NYIS", "ISNE", "SWPP")
PINNED_CHRONOS2_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
TARGET_COVERAGE = 0.8
DEFAULT_COVERAGE_TOLERANCE = 0.02
_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class BAPredictions:
    origins: np.ndarray
    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray
    truth: np.ndarray


@dataclass(frozen=True)
class ModelProvenance:
    is_local: bool
    revision: str | None
    artifact_sha256: str | None
    artifact_hash_algorithm: str | None


@dataclass(frozen=True)
class CandidateRun:
    pooled: bool
    window: int
    calibrated: CalibratedIntervals


def _model_output(value: Any) -> np.ndarray:
    if hasattr(value, "squeeze"):
        value = value.squeeze(0)
    for method in ("float", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def _validation_origins(data: BAData, *, context: int, horizon: int) -> list[int]:
    origins = [
        origin
        for origin in range(data.train_end, data.val_end - horizon + 1, horizon)
        if origin - context >= 0
    ]
    if not origins:
        raise ValueError(f"no validation origins for {data.ba}")
    origin_times = data.ts_utc[np.asarray(origins)].astype("datetime64[us]")
    if len(origin_times) > 1 and np.any(
        np.diff(origin_times) != np.timedelta64(horizon, "h")
    ):
        raise ValueError(f"{data.ba} validation origins are not {horizon} hours apart")
    if np.any(np.asarray(origins) + horizon > data.val_end):
        raise RuntimeError(f"{data.ba} validation schedule crosses into the held-out lane")
    return origins


def _validate_origin_alignment(
    data: dict[str, BAData],
    bas: list[str],
    *,
    context: int,
    horizon: int,
) -> dict[str, list[int]]:
    schedules = {
        ba: _validation_origins(data[ba], context=context, horizon=horizon) for ba in bas
    }
    reference_ba = bas[0]
    reference = data[reference_ba].ts_utc[np.asarray(schedules[reference_ba])].astype(
        "datetime64[us]"
    )
    mismatches: list[str] = []
    for ba in bas[1:]:
        timestamps = data[ba].ts_utc[np.asarray(schedules[ba])].astype("datetime64[us]")
        if not np.array_equal(timestamps, reference):
            mismatches.append(
                f"{ba}={len(timestamps)} origins "
                f"({timestamps[0] if len(timestamps) else 'none'}.."
                f"{timestamps[-1] if len(timestamps) else 'none'})"
            )
    if mismatches:
        reference_range = f"{reference[0]}..{reference[-1]}"
        raise ValueError(
            f"validation origin alignment is incomplete; {reference_ba}={len(reference)} "
            f"({reference_range}); " + "; ".join(mismatches)
        )
    return schedules


def _predict_ba(
    pipe: Any,
    data: BAData,
    *,
    origins: list[int],
    context: int,
    horizon: int,
    batch_size: int,
) -> BAPredictions:
    lower_rows: list[np.ndarray] = []
    median_rows: list[np.ndarray] = []
    upper_rows: list[np.ndarray] = []
    truth_rows: list[np.ndarray] = []
    for start in range(0, len(origins), batch_size):
        batch_origins = origins[start : start + batch_size]
        tasks = [
            build_evaluation_task(
                data,
                origin=origin,
                context_length=context,
                prediction_length=horizon,
                spec=LOAD_V2_CORE,
            )
            for origin in batch_origins
        ]
        quantiles_list, _means = pipe.predict_quantiles(
            tasks,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=len(tasks),
        )
        quantiles = np.stack([_model_output(value) for value in quantiles_list])
        expected = (len(tasks), horizon, 3)
        if quantiles.shape != expected:
            raise ValueError(f"unexpected quantile shape {quantiles.shape}; expected {expected}")
        if not np.isfinite(quantiles).all() or np.any(np.diff(quantiles, axis=-1) < 0):
            raise ValueError(f"invalid quantiles for {data.ba}")
        lower_rows.extend(quantiles[..., 0])
        median_rows.extend(quantiles[..., 1])
        upper_rows.extend(quantiles[..., 2])
        truth_rows.extend(data.target[origin : origin + horizon] for origin in batch_origins)

    return BAPredictions(
        origins=data.ts_utc[np.asarray(origins)].astype("datetime64[us]"),
        lower=np.asarray(lower_rows),
        median=np.asarray(median_rows),
        upper=np.asarray(upper_rows),
        truth=np.asarray(truth_rows),
    )


def _align_predictions(
    predictions: dict[str, BAPredictions],
    bas: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not bas:
        raise ValueError("at least one BA is required")
    reference = predictions[bas[0]].origins.astype("datetime64[us]")
    if not len(reference):
        raise ValueError("prediction origin schedule is empty")
    horizon: int | None = None
    fields: dict[str, list[np.ndarray]] = {
        "lower": [],
        "median": [],
        "upper": [],
        "truth": [],
    }
    for ba in bas:
        prediction = predictions[ba]
        origins = prediction.origins.astype("datetime64[us]")
        if not np.array_equal(origins, reference):
            raise ValueError(f"{ba} prediction origins do not exactly match the shared schedule")
        for field in fields:
            values = np.asarray(getattr(prediction, field))
            if values.ndim != 2 or values.shape[0] != len(reference):
                raise ValueError(f"{ba} {field} shape {values.shape} is not origin-by-horizon")
            if horizon is None:
                horizon = values.shape[1]
            if values.shape[1] != horizon:
                raise ValueError(f"{ba} {field} horizon does not match the shared schedule")
            fields[field].append(values)
    return (
        reference,
        np.stack(fields["lower"], axis=1),
        np.stack(fields["median"], axis=1),
        np.stack(fields["upper"], axis=1),
        np.stack(fields["truth"], axis=1),
    )


def _rounded(metrics: dict[str, float | int]) -> dict[str, float | int]:
    return {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in metrics.items()
    }


def _metric_pairs_by_ba(
    *,
    bas: list[str],
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
    calibrated: CalibratedIntervals,
    truth: np.ndarray,
    mask: np.ndarray,
) -> dict[str, dict[str, dict[str, float | int]]]:
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for index, ba in enumerate(bas):
        ba_slice = slice(index, index + 1)
        ba_mask = mask[:, ba_slice, :]
        result[ba] = {
            "baseline": interval_metrics(
                lower[:, ba_slice, :],
                median[:, ba_slice, :],
                upper[:, ba_slice, :],
                truth[:, ba_slice, :],
                mask=ba_mask,
            ),
            "calibrated": interval_metrics(
                calibrated.lower[:, ba_slice, :],
                median[:, ba_slice, :],
                calibrated.upper[:, ba_slice, :],
                truth[:, ba_slice, :],
                mask=ba_mask,
            ),
        }
    return result


def _relative_interval_score(
    per_ba: dict[str, dict[str, dict[str, float | int]]],
) -> float:
    ratios: list[float] = []
    for metrics in per_ba.values():
        baseline = float(metrics["baseline"]["interval_score"])
        calibrated = float(metrics["calibrated"]["interval_score"])
        if baseline < 0 or calibrated < 0:
            raise RuntimeError("interval scores must be nonnegative")
        if baseline == 0:
            ratios.append(1.0 if calibrated == 0 else math.inf)
        else:
            ratios.append(calibrated / baseline)
    return float(np.mean(ratios))


def _candidate_summary(
    *,
    bas: list[str],
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
    truth: np.ndarray,
    run: CandidateRun,
    mask: np.ndarray,
    target_coverage: float,
    coverage_tolerance: float,
) -> dict[str, Any]:
    per_ba_raw = _metric_pairs_by_ba(
        bas=bas,
        lower=lower,
        median=median,
        upper=upper,
        calibrated=run.calibrated,
        truth=truth,
        mask=mask,
    )
    relative_score = _relative_interval_score(per_ba_raw)
    coverage_errors = {
        ba: abs(float(metrics["calibrated"]["coverage"]) - target_coverage)
        for ba, metrics in per_ba_raw.items()
    }
    baseline = interval_metrics(lower, median, upper, truth, mask=mask)
    adjusted = interval_metrics(
        run.calibrated.lower,
        median,
        run.calibrated.upper,
        truth,
        mask=mask,
    )
    return {
        "pool": "all-seven-rto-normalized-block" if run.pooled else "per-rto",
        "window_origins": run.window,
        "window_mature_origins": run.window,
        "baseline": _rounded(baseline),
        "calibrated": _rounded(adjusted),
        "macro_relative_interval_score": round(relative_score, 6),
        "macro_interval_score_improvement_pct": round(100.0 * (1.0 - relative_score), 4),
        "interval_score_improvement_pct": round(100.0 * (1.0 - relative_score), 4),
        "max_per_ba_coverage_error": round(max(coverage_errors.values()), 6),
        "coverage_constraint_satisfied": all(
            error <= coverage_tolerance for error in coverage_errors.values()
        ),
        "per_ba": {
            ba: {
                "baseline": _rounded(metrics["baseline"]),
                "calibrated": _rounded(metrics["calibrated"]),
                "coverage_error": round(coverage_errors[ba], 6),
            }
            for ba, metrics in per_ba_raw.items()
        },
    }


def _common_eligible(runs: list[CandidateRun]) -> np.ndarray:
    if not runs:
        raise ValueError("at least one calibration candidate is required")
    common = runs[0].calibrated.eligible.copy()
    for run in runs[1:]:
        if run.calibrated.eligible.shape != common.shape:
            raise ValueError("candidate eligibility shapes do not match")
        common &= run.calibrated.eligible
    return common


def _partition_masks(
    common_eligible: np.ndarray,
    *,
    selection_fraction: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    origin_count = common_eligible.shape[0]
    split_index = math.floor(origin_count * selection_fraction)
    if split_index < 1 or split_index >= origin_count:
        raise ValueError("selection split leaves an empty chronological partition")
    selection = common_eligible.copy()
    selection[split_index:] = False
    outer = common_eligible.copy()
    outer[:split_index] = False
    if not np.any(selection) or not np.any(outer):
        raise ValueError("common eligibility leaves an empty selection or outer partition")
    return selection, outer, split_index


def _eligible_origin_range(
    origins: np.ndarray,
    mask: np.ndarray,
    truth: np.ndarray,
) -> tuple[str, str]:
    scored = mask & np.isfinite(truth)
    origin_selected = np.any(scored, axis=(1, 2))
    indices = np.flatnonzero(origin_selected)
    if not len(indices):
        raise ValueError("partition has no finite scored origins")
    return str(origins[indices[0]]), str(origins[indices[-1]])


def _optional_distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _validate_args(args: argparse.Namespace) -> ModelProvenance:
    if len(args.bas) != len(RTO_BAS) or len(set(args.bas)) != len(RTO_BAS):
        raise ValueError("--bas must contain each of the seven v0.2 RTOs exactly once")
    if set(args.bas) != set(RTO_BAS):
        raise ValueError("--bas must be exactly PJM CISO ERCO MISO NYIS ISNE SWPP")
    if args.horizon != 24:
        raise ValueError("v0.2 calibration selection requires a 24-hour horizon")
    if args.context < 1 or args.batch_size < 1:
        raise ValueError("--context and --batch-size must be positive")
    if args.min_history < 1:
        raise ValueError("--min-history must be positive")
    if not args.windows or len(args.windows) != len(set(args.windows)):
        raise ValueError("--windows must contain unique values")
    if any(window < args.min_history for window in args.windows):
        raise ValueError("every window must be at least --min-history")
    if not math.isclose(args.coverage, TARGET_COVERAGE, abs_tol=1e-12):
        raise ValueError("ledger-compatible p10/p90 scoring requires --coverage 0.8")
    if not 0 <= args.coverage_tolerance < 0.5:
        raise ValueError("--coverage-tolerance must be in [0, 0.5)")
    if not 0 < args.selection_fraction < 1:
        raise ValueError("--selection-fraction must be in (0, 1)")
    if not _SHA40.fullmatch(args.code_revision):
        raise ValueError("--code-revision must be a full 40-character Git commit SHA")
    if not _SHA64.fullmatch(args.data_snapshot_sha256):
        raise ValueError("--data-snapshot-sha256 must be a lowercase SHA-256")

    model_path = Path(args.model).expanduser()
    supplied_artifact_sha = args.model_artifact_sha256 or None
    if supplied_artifact_sha is not None and not _SHA64.fullmatch(supplied_artifact_sha):
        raise ValueError("--model-artifact-sha256 must be a lowercase SHA-256")
    if model_path.exists():
        computed = artifact_sha256(model_path)
        if supplied_artifact_sha is not None and supplied_artifact_sha != computed:
            raise ValueError("--model-artifact-sha256 does not match the local model artifact")
        if args.model_revision:
            raise ValueError("--model-revision does not apply to a local artifact; use its SHA-256")
        return ModelProvenance(
            is_local=True,
            revision=None,
            artifact_sha256=computed,
            artifact_hash_algorithm="sha256-tree-v1",
        )
    if not args.model_revision or not _SHA40.fullmatch(args.model_revision):
        raise ValueError("remote models require a full 40-character immutable revision")
    return ModelProvenance(
        is_local=False,
        revision=args.model_revision,
        artifact_sha256=supplied_artifact_sha,
        artifact_hash_algorithm=None,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="amazon/chronos-2")
    parser.add_argument("--model-revision", default=os.environ.get("SURGE_MODEL_REVISION"))
    parser.add_argument(
        "--model-artifact-sha256",
        default=os.environ.get("SURGE_MODEL_ARTIFACT_SHA256", ""),
    )
    parser.add_argument("--bas", nargs="+", default=list(RTO_BAS))
    parser.add_argument("--context", type=int, default=2_048)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--windows", nargs="+", type=int, default=[28, 42, 56, 84, 112, 168])
    parser.add_argument("--min-history", type=int, default=28)
    parser.add_argument("--coverage", type=float, default=TARGET_COVERAGE)
    parser.add_argument("--coverage-tolerance", type=float, default=DEFAULT_COVERAGE_TOLERANCE)
    parser.add_argument("--selection-fraction", type=float, default=0.5)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--code-revision", default=os.environ.get("SURGE_CODE_REVISION", "unknown"))
    parser.add_argument(
        "--data-snapshot-sha256",
        default=os.environ.get("SURGE_DATA_SNAPSHOT_SHA256", "unknown"),
    )
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.model == "amazon/chronos-2" and not args.model_revision:
        args.model_revision = PINNED_CHRONOS2_REVISION
    try:
        model_provenance = _validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    verify_code_checkout(Path(__file__).resolve().parents[1], args.code_revision)
    data_root_value = os.environ.get("SURGE_DATA_DIR")
    if not data_root_value:
        parser.error("SURGE_DATA_DIR is required to bind the conformal data snapshot")
    try:
        data_root = verify_data_snapshot_manifest(
            Path(data_root_value), args.data_snapshot_sha256
        )
        verify_snapshot(data_root)
    except ValueError as exc:
        parser.error(str(exc))

    availability_mode = AvailabilityMode.RETROSPECTIVE_FINAL
    # Apply the locked-test boundary in the lazy data query itself.  The
    # conformal lane is validation-only, so 2025 target rows must never enter
    # process memory merely because the scheduler later ignores them.
    selection_valid_before = datetime(2025, 1, 1, tzinfo=UTC)
    data = load_multi_ba(
        args.bas,
        with_gen=False,
        availability_mode=availability_mode,
        valid_before=selection_valid_before,
    )
    schedules = _validate_origin_alignment(
        data,
        args.bas,
        context=args.context,
        horizon=args.horizon,
    )

    import torch

    if args.device_map.startswith("cuda") and not torch.cuda.is_available():
        parser.error("--device-map requests CUDA but torch.cuda.is_available() is false")
    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "dtype": torch_dtype,
    }
    if model_provenance.revision is not None:
        model_kwargs["revision"] = model_provenance.revision
    load_started = time.time()
    pipe = load_chronos2(args.model, **model_kwargs)
    load_seconds = time.time() - load_started

    predict_started = time.time()
    by_ba = {
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
    origins, lower, median, upper, truth = _align_predictions(by_ba, args.bas)
    predict_seconds = time.time() - predict_started

    candidate_runs = [
        CandidateRun(
            pooled=pooled,
            window=window,
            calibrated=rolling_conformalize(
                lower,
                median,
                upper,
                truth,
                origin_times_utc=origins,
                outcome_delay_hours=MATURITY_HOURS,
                window=window,
                min_history=args.min_history,
                coverage=args.coverage,
                pooled_series=pooled,
                normalized=True,
            ),
        )
        for pooled in (False, True)
        for window in args.windows
    ]
    common = _common_eligible(candidate_runs)
    selection_mask, outer_mask, split_index = _partition_masks(
        common,
        selection_fraction=args.selection_fraction,
    )

    candidates = [
        _candidate_summary(
            bas=args.bas,
            lower=lower,
            median=median,
            upper=upper,
            truth=truth,
            run=run,
            mask=selection_mask,
            target_coverage=args.coverage,
            coverage_tolerance=args.coverage_tolerance,
        )
        for run in candidate_runs
    ]
    feasible_indices = [
        index
        for index, candidate in enumerate(candidates)
        if candidate["coverage_constraint_satisfied"]
    ]
    selection_indices = feasible_indices or list(range(len(candidates)))
    best_index = min(
        selection_indices,
        key=lambda index: float(candidates[index]["macro_relative_interval_score"]),
    )
    best = candidates[best_index]
    best_run = candidate_runs[best_index]
    outer = _candidate_summary(
        bas=args.bas,
        lower=lower,
        median=median,
        upper=upper,
        truth=truth,
        run=best_run,
        mask=outer_mask,
        target_coverage=args.coverage,
        coverage_tolerance=args.coverage_tolerance,
    )
    selection_start, selection_end = _eligible_origin_range(origins, selection_mask, truth)
    outer_start, outer_end = _eligible_origin_range(origins, outer_mask, truth)

    output = {
        "schema_version": 2,
        "protocol": "validation-only-prequential-inner-select-outer-report",
        "held_out_test_lane": "not-opened",
        "model": args.model,
        "model_revision": model_provenance.revision,
        "model_artifact_sha256": model_provenance.artifact_sha256,
        "model_artifact_hash_algorithm": model_provenance.artifact_hash_algorithm,
        "code_revision": args.code_revision,
        "data_snapshot_sha256": args.data_snapshot_sha256,
        "feature_spec_version": LOAD_V2_CORE.version,
        "feature_spec_sha256": LOAD_V2_CORE.sha256,
        "availability_mode": availability_mode.value,
        "selection_valid_before_utc": selection_valid_before.isoformat(),
        "point_in_time_replay": False,
        "outcome_timing_policy_reference": OUTCOME_POLICY_VERSION,
        "outcome_delay_hours": MATURITY_HOURS,
        "score_availability_rule": "prior forecast last target +72h <= current origin",
        "outcome_revision_availability": (
            "retrospective-final; this is not an exact +72h vintage replay"
        ),
        "bas": args.bas,
        "origin_start_utc": str(origins[0]),
        "origin_end_utc": str(origins[-1]),
        "origin_count": len(origins),
        "origin_count_by_ba": {ba: len(schedules[ba]) for ba in args.bas},
        "horizon": args.horizon,
        "context": args.context,
        "batch_size": args.batch_size,
        "target_coverage": args.coverage,
        "coverage_tolerance": args.coverage_tolerance,
        "min_history_mature_origins": args.min_history,
        "candidate_windows_mature_origins": args.windows,
        "normalized_cqr": True,
        "negative_adjustment_policy": "clamp-to-zero-no-interval-shrink",
        "pooled_rank_policy": "complete-origin-block finite-sample correction",
        "pooled_cross_series_guarantee": (
            "heuristic ablation; simultaneous RTO scores are not claimed independent"
        ),
        "selection": {
            "chronological_fraction": args.selection_fraction,
            "raw_origin_count": split_index,
            "scored_origin_start_utc": selection_start,
            "scored_origin_end_utc": selection_end,
            "rule_predeclared": (
                "minimum equal-BA macro relative interval score among candidates with every "
                "BA within coverage tolerance"
            ),
            "rule_applied": (
                "coverage-constrained"
                if feasible_indices
                else "fallback-minimum-relative-score-no-candidate-met-coverage"
            ),
            "coverage_constraint_satisfied": bool(feasible_indices),
        },
        "outer_validation": {
            "used_for_selection": False,
            "raw_origin_count": len(origins) - split_index,
            "scored_origin_start_utc": outer_start,
            "scored_origin_end_utc": outer_end,
            "result": outer,
        },
        "research_basis": [
            {
                "id": "arXiv:2605.08422v1",
                "title": "Rolling-Origin Conformal Prediction under Local Stationarity and Weak Dependence",
                "url": "https://arxiv.org/abs/2605.08422v1",
            },
            {
                "id": "arXiv:2606.31804v1",
                "title": (
                    "Relational and Sequential Conformal Inference for Energy Time Series over "
                    "Graphs via Foundation Models"
                ),
                "url": "https://arxiv.org/abs/2606.31804v1",
                "relationship": "motivates cross-RTO ablation; Surge does not implement STOIC",
            },
        ],
        "versions": {
            "chronos_forecasting": importlib.metadata.version("chronos-forecasting"),
            "transformers": importlib.metadata.version("transformers"),
            "numpy": np.__version__,
            "polars": importlib.metadata.version("polars"),
            "holidays": importlib.metadata.version("holidays"),
            "pyarrow": importlib.metadata.version("pyarrow"),
            "torch": torch.__version__,
            "peft": _optional_distribution_version("peft"),
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
        "load_seconds": round(load_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
        "candidate_count": len(candidates),
        "feasible_candidate_count": len(feasible_indices),
        "selection_rule": (
            "chronological inner selection by equal-BA relative interval score with per-BA "
            "coverage constraint; outer validation is reporting-only"
        ),
        "best": best,
        "candidates": candidates,
        "best_selection_candidate": best,
        "candidates_selection_only": candidates,
    }
    payload = json.dumps(output, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(f".{args.out.name}.tmp-{os.getpid()}")
        temporary.write_text(payload + "\n", encoding="utf-8")
        os.replace(temporary, args.out)
    print("CALIBRATION_RESULT:", json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
