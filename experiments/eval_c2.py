"""Rolling 24h eval for Chronos-2 with covariates.

Differs from `eval.py` only in how model inputs are shaped: we pass
`{target, past_covariates, future_covariates}` dicts instead of raw arrays.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from experiments.features import BAData
from surge.features import (
    LOAD_V2_CORE,
    POINT_ESTIMATE_KIND,
    POINT_ESTIMATE_LABEL,
    POINT_ESTIMATE_QUANTILE,
    build_evaluation_task,
)

_REQUIRED_QUANTILES = (0.1, 0.5, 0.9)
_SUM_FIELDS = (
    "actual",
    "forecast_error",
    "abs_error",
    "squared_error",
    "pinball_p10",
    "pinball_p50",
    "pinball_p90",
    "pinball_all",
    "pi80_coverage",
    "pi80_width",
    "wis",
)
_AGGREGATE_METRIC_KEYS = (
    "mae",
    "rmse",
    "mase",
    "bias",
    "pinball_p10",
    "pinball_p50",
    "pinball_p90",
    "mean_pi80_width",
    "cov_pi80",
    "wis",
    "wis_scaled",
    "crps_approx",
)
_MICROSECONDS_PER_HOUR = 3_600_000_000


@dataclass(frozen=True)
class SharedOriginSchedule:
    """One target-availability-only origin cohort shared by every RTO."""

    split: str
    origin_keys_utc_us: tuple[int, ...]
    step_hours: int
    requested_origin_count: int
    candidate_grid_count: int
    complete_shared_count: int
    excluded_incomplete_count: int
    excluded_latest_complete_count: int
    anchor_utc: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            np.asarray(self.origin_keys_utc_us, dtype="<i8").tobytes()
        ).hexdigest()

    def indices_for(self, data: BAData) -> list[int]:
        timestamp_keys = data.ts_utc.astype("datetime64[us]").astype(np.int64)
        indices = np.searchsorted(
            timestamp_keys,
            np.asarray(self.origin_keys_utc_us, dtype=np.int64),
        )
        if np.any(indices >= len(timestamp_keys)) or any(
            int(timestamp_keys[index]) != key
            for index, key in zip(indices, self.origin_keys_utc_us, strict=True)
        ):
            raise ValueError(f"{data.ba} does not contain the shared origin schedule")
        return [int(index) for index in indices]

    def as_dict(self) -> dict[str, Any]:
        origins_utc = [
            str(np.datetime64(origin_key, "us"))
            for origin_key in self.origin_keys_utc_us
        ]
        return {
            "split": self.split,
            "shared_across_bas": True,
            "complete_target_origins_only": True,
            "origin_count": len(self.origin_keys_utc_us),
            "requested_origin_count": self.requested_origin_count,
            "step_hours": self.step_hours,
            "candidate_grid_count": self.candidate_grid_count,
            "complete_shared_count": self.complete_shared_count,
            "excluded_incomplete_count": self.excluded_incomplete_count,
            "excluded_latest_complete_count": self.excluded_latest_complete_count,
            "anchor_utc": self.anchor_utc,
            "origin_start_utc": origins_utc[0],
            "origin_end_utc": origins_utc[-1],
            "origins_utc": origins_utc,
            "origin_sha256": self.sha256,
        }


def _split_bounds(
    data: BAData,
    *,
    on: str,
    context: int,
    horizon: int,
) -> tuple[int, int]:
    if on == "train":
        eval_start, eval_end = context, data.train_end
    elif on == "val":
        eval_start, eval_end = data.train_end, data.val_end
    elif on == "test":
        eval_start, eval_end = data.val_end, len(data.target)
    else:
        raise ValueError("on must be 'train', 'val', or 'test'")
    first = max(eval_start, context)
    last = eval_end - horizon
    if last < first:
        raise ValueError(f"{data.ba} has no evaluable {on} origins")
    return first, last


def select_shared_complete_origin_schedule(
    bas: Mapping[str, BAData],
    *,
    on: str,
    context: int,
    horizon: int,
    step: int,
    origin_count: int,
    exclude_latest_complete: int = 0,
) -> SharedOriginSchedule:
    """Select a backwards-anchored complete cohort shared by every RTO.

    Selection observes target availability only. It never reads predictions or
    model errors. ``exclude_latest_complete`` reserves a later cohort, allowing
    checkpoint selection and promotion to use disjoint validation windows.
    """
    if not bas:
        raise ValueError("shared origin selection requires at least one BA")
    for name, value in (
        ("context", context),
        ("horizon", horizon),
        ("step", step),
        ("origin_count", origin_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(exclude_latest_complete, bool)
        or not isinstance(exclude_latest_complete, int)
        or exclude_latest_complete < 0
    ):
        raise ValueError("exclude_latest_complete must be a non-negative integer")

    bounds = {
        ba: _split_bounds(data, on=on, context=context, horizon=horizon)
        for ba, data in bas.items()
    }
    earliest_key = max(
        int(bas[ba].ts_utc[first].astype("datetime64[us]").astype(np.int64))
        for ba, (first, _last) in bounds.items()
    )
    latest_key = min(
        int(bas[ba].ts_utc[last].astype("datetime64[us]").astype(np.int64))
        for ba, (_first, last) in bounds.items()
    )
    if latest_key < earliest_key:
        raise ValueError(f"RTOs have no shared {on} origin range")

    step_us = step * _MICROSECONDS_PER_HOUR
    candidate_keys_desc = list(range(latest_key, earliest_key - 1, -step_us))
    indices_by_ba = {
        ba: {
            int(timestamp): index
            for index, timestamp in enumerate(
                data.ts_utc.astype("datetime64[us]").astype(np.int64)
            )
        }
        for ba, data in bas.items()
    }
    complete_desc: list[int] = []
    for origin_key in candidate_keys_desc:
        complete = True
        for ba, data in bas.items():
            origin = indices_by_ba[ba].get(origin_key)
            first, last = bounds[ba]
            if (
                origin is None
                or origin < first
                or origin > last
                or not np.isfinite(data.target[origin : origin + horizon]).all()
            ):
                complete = False
                break
        if complete:
            complete_desc.append(origin_key)

    stop = exclude_latest_complete + origin_count
    if len(complete_desc) < stop:
        raise ValueError(
            f"shared {on} schedule requires {stop} complete origins but found "
            f"{len(complete_desc)}"
        )
    selected = tuple(
        reversed(
            complete_desc[
                exclude_latest_complete : exclude_latest_complete + origin_count
            ]
        )
    )
    return SharedOriginSchedule(
        split=on,
        origin_keys_utc_us=selected,
        step_hours=step,
        requested_origin_count=origin_count,
        candidate_grid_count=len(candidate_keys_desc),
        complete_shared_count=len(complete_desc),
        excluded_incomplete_count=len(candidate_keys_desc) - len(complete_desc),
        excluded_latest_complete_count=exclude_latest_complete,
        anchor_utc=str(np.datetime64(latest_key, "us")),
    )


def _model_output(value: Any) -> np.ndarray:
    """Convert torch-like output to numpy without making torch a test dependency."""
    if hasattr(value, "squeeze"):
        value = value.squeeze(0)
    for method in ("float", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def _paired_origin_block_mase_ci(
    mase_by_ba_origin: dict[str, dict[int, float]],
    *,
    bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap equal-BA macro MASE by resampling shared origin blocks."""
    if bootstrap < 1:
        raise ValueError("bootstrap samples must be positive")
    if not mase_by_ba_origin:
        raise ValueError("bootstrap requested but there are no evaluated BAs")

    bas = list(mase_by_ba_origin)
    reference_ba = bas[0]
    shared_origins = set(mase_by_ba_origin[reference_ba])
    for ba in bas[1:]:
        candidate_origins = set(mase_by_ba_origin[ba])
        if candidate_origins != shared_origins:
            missing = len(shared_origins - candidate_origins)
            extra = len(candidate_origins - shared_origins)
            raise ValueError(
                "bootstrap requires identical origin schedules across evaluated BAs; "
                f"{ba} differs from {reference_ba} (missing={missing}, extra={extra})"
            )
    if not shared_origins:
        raise ValueError(
            "bootstrap requested but no shared finite origin blocks exist across evaluated BAs"
        )
    finite_origins = sorted(shared_origins)
    if any(
        not np.isfinite(mase_by_ba_origin[ba][origin])
        for origin in finite_origins
        for ba in bas
    ):
        raise ValueError(
            "bootstrap requires every shared origin block to have finite MASE across all BAs"
        )

    # A row is one forecast-origin block across every BA. Drawing row indices
    # once preserves same-origin cross-BA dependence; the two-stage mean makes
    # the equal-BA macro statistic explicit.
    blocks = np.asarray(
        [[mase_by_ba_origin[ba][origin] for ba in bas] for origin in finite_origins],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    boots = np.empty(bootstrap, dtype=np.float64)
    for sample in range(bootstrap):
        indices = rng.integers(0, len(blocks), len(blocks))
        per_ba_mase = blocks[indices].mean(axis=0)
        boots[sample] = per_ba_mase.mean()
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _summarize_metric_sums(
    sums: Mapping[str, float],
    *,
    count: int,
    denom_mae: float,
    quantile_count: int,
) -> dict[str, float]:
    """Convert additive point statistics into one finite metric row."""
    if count < 1:
        raise ValueError("cannot summarize metrics without finite target points")
    mae = sums["abs_error"] / count
    wis = sums["wis"] / count
    return {
        "mae": mae,
        "rmse": math.sqrt(sums["squared_error"] / count),
        "mase": mae / denom_mae,
        # Bias follows the serving/verification convention: forecast - actual.
        "bias": sums["forecast_error"] / count,
        "pinball_p10": sums["pinball_p10"] / count,
        "pinball_p50": sums["pinball_p50"] / count,
        "pinball_p90": sums["pinball_p90"] / count,
        "mean_pi80_width": sums["pi80_width"] / count,
        "cov_pi80": sums["pi80_coverage"] / count,
        "wis": wis,
        "wis_scaled": wis / denom_mae,
        # With a finite quantile grid this is an approximation to the CRPS
        # integral, not exact CRPS. The factor of two is part of the identity.
        "crps_approx": 2.0 * sums["pinball_all"] / (count * quantile_count),
        "mean_actual_mw": sums["actual"] / count,
    }


def _load_weighted_aggregate(
    rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Weight BA-level metrics by mean scored actual load when that is sound."""
    bas = list(rows)
    weights = np.asarray([float(rows[ba]["mean_actual_mw"]) for ba in bas], dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        return None
    normalized = weights / weights.sum()
    result: dict[str, Any] = {
        "aggregation": "mean_actual_mw_weighted_mean_of_ba_metrics",
        "weight_basis": "mean_actual_mw",
        "weights": {ba: float(weight) for ba, weight in zip(bas, normalized, strict=True)},
    }
    for key in _AGGREGATE_METRIC_KEYS:
        values = np.asarray([float(rows[ba][key]) for ba in bas], dtype=np.float64)
        result[key] = float(np.dot(normalized, values))
    return result


def rolling_eval_c2(
    pipe,
    bas: dict[str, BAData],
    *,
    on: str = "val",
    context: int = 2048,
    horizon: int = 24,
    step: int = 24,
    quantile_levels: Iterable[float] = (0.1, 0.5, 0.9),
    batch_size: int = 16,
    bootstrap: int = 0,
    seed: int = 0,
    per_step: bool = False,
    max_origins: int | None = None,
    require_complete_origins: bool = False,
    origin_schedule: SharedOriginSchedule | None = None,
    emit_origin_metrics: bool = False,
) -> dict:
    if on not in {"train", "val", "test"}:
        raise ValueError("on must be 'train', 'val', or 'test'")
    for name, value in (
        ("context", context),
        ("horizon", horizon),
        ("step", step),
        ("batch_size", batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("bootstrap must be a non-negative integer")
    if max_origins is not None and (
        isinstance(max_origins, bool)
        or not isinstance(max_origins, int)
        or max_origins < 1
    ):
        raise ValueError("max_origins must be a positive integer when provided")
    if not isinstance(require_complete_origins, bool):
        raise ValueError("require_complete_origins must be a boolean")
    if not isinstance(emit_origin_metrics, bool):
        raise ValueError("emit_origin_metrics must be a boolean")
    if origin_schedule is not None:
        if max_origins is not None:
            raise ValueError("max_origins cannot be combined with origin_schedule")
        if require_complete_origins is not True:
            raise ValueError("shared origin schedules require complete target origins")
        if origin_schedule.split != on or origin_schedule.step_hours != step:
            raise ValueError("shared origin schedule disagrees with evaluation split or step")
    q_levels = list(quantile_levels)
    if (
        not q_levels
        or any(
            isinstance(level, bool)
            or not isinstance(level, (float, int))
            or not 0 < level < 1
            for level in q_levels
        )
        or any(left >= right for left, right in pairwise(q_levels))
    ):
        raise ValueError("quantile_levels must be unique, increasing values inside (0, 1)")
    missing_quantiles = [level for level in _REQUIRED_QUANTILES if level not in q_levels]
    if missing_quantiles:
        raise ValueError(
            "quantile_levels must contain 0.1, 0.5, and 0.9 for protocol metrics; "
            f"missing {missing_quantiles}"
        )
    p10_index = q_levels.index(0.1)
    p50_index = q_levels.index(POINT_ESTIMATE_QUANTILE)
    p90_index = q_levels.index(0.9)
    quantile_array = np.asarray(q_levels, dtype=np.float64)[None, None, :]
    per_ba: dict[str, dict[str, Any]] = {}
    mase_by_ba_origin: dict[str, dict[int, float]] = {}

    for ba, bd in bas.items():
        try:
            eval_start, last_origin = _split_bounds(
                bd,
                on=on,
                context=context,
                horizon=horizon,
            )
        except ValueError:
            if bootstrap > 0:
                raise ValueError(
                    f"{ba} bootstrap requires at least one evaluable origin"
                ) from None
            if origin_schedule is not None:
                raise
            continue
        if origin_schedule is not None:
            origins = origin_schedule.indices_for(bd)
            if any(origin < eval_start or origin > last_origin for origin in origins):
                raise ValueError(f"{ba} shared origin schedule escapes the {on} split")
            candidate_origin_count = origin_schedule.candidate_grid_count
            incomplete_target_origins = origin_schedule.excluded_incomplete_count
        else:
            candidate_origins = list(range(eval_start, last_origin + 1, step))
            incomplete_target_origins = sum(
                not np.isfinite(bd.target[origin : origin + horizon]).all()
                for origin in candidate_origins
            )
            origins = candidate_origins
            if require_complete_origins:
                # This filter depends only on realized-target availability and
                # is applied before the trailing limit, never on performance.
                origins = [
                    origin
                    for origin in candidate_origins
                    if np.isfinite(bd.target[origin : origin + horizon]).all()
                ]
            if max_origins is not None:
                origins = origins[-max_origins:]
            candidate_origin_count = len(candidate_origins)
        if not origins:
            if bootstrap > 0:
                raise ValueError(f"{ba} bootstrap requires at least one evaluable origin")
            continue
        if not np.isfinite(bd.denom_mae) or bd.denom_mae <= 0:
            raise ValueError(
                f"{ba} has no finite per-window MASE: denom_mae must be finite and positive"
            )

        # Batch origins
        per_horizon_sums = {
            field: np.zeros(horizon, dtype=np.float64) for field in _SUM_FIELDS
        }
        per_horizon_counts = np.zeros(horizon, dtype=np.int64)
        origin_mase: dict[int, float] = {}
        for i in range(0, len(origins), batch_size):
            batch_origins = origins[i : i + batch_size]
            tasks = []
            truth_rows = []
            for o in batch_origins:
                tasks.append(
                    build_evaluation_task(
                        bd,
                        origin=o,
                        context_length=context,
                        prediction_length=horizon,
                        spec=LOAD_V2_CORE,
                    )
                )
                truth_rows.append(bd.target[o : o + horizon].astype(np.float32))
            truths = np.stack(truth_rows)  # (B, H)

            quants_list, _means_list = pipe.predict_quantiles(
                tasks,
                prediction_length=horizon,
                quantile_levels=q_levels,
                batch_size=len(tasks),
            )
            quants = np.stack([_model_output(quantile) for quantile in quants_list])
            expected_shape = (len(tasks), horizon, len(q_levels))
            if quants.shape != expected_shape:
                raise ValueError(
                    f"unexpected quantile output shape {quants.shape}; expected {expected_shape}"
                )
            if not np.isfinite(quants).all():
                raise ValueError("model returned non-finite quantiles")
            if np.any(quants < 0):
                raise ValueError("model returned negative load quantiles")
            if np.any(np.diff(quants, axis=-1) < 0):
                raise ValueError("model returned crossing quantiles")
            p50 = quants[..., p50_index]

            # The public point forecast is p50, so all point metrics and MASE
            # must score p50 rather than Chronos' separately returned mean.
            valid = np.isfinite(truths)
            forecast_error = p50 - truths
            abs_error = np.abs(forecast_error)
            quantile_errors = truths[..., None] - quants
            pinball = np.where(
                quantile_errors >= 0,
                quantile_array * quantile_errors,
                (1.0 - quantile_array) * -quantile_errors,
            )
            lower = quants[..., p10_index]
            upper = quants[..., p90_index]
            pi80_width = upper - lower
            pi80_coverage = (truths >= lower) & (truths <= upper)
            below = np.where(truths < lower, 10.0 * (lower - truths), 0.0)
            above = np.where(truths > upper, 10.0 * (truths - upper), 0.0)
            interval_score = pi80_width + below + above
            wis = (0.5 * abs_error + 0.1 * interval_score) / 1.5

            batch_values = {
                "actual": truths,
                "forecast_error": forecast_error,
                "abs_error": abs_error,
                "squared_error": forecast_error * forecast_error,
                "pinball_p10": pinball[..., p10_index],
                "pinball_p50": pinball[..., p50_index],
                "pinball_p90": pinball[..., p90_index],
                "pinball_all": pinball.sum(axis=-1),
                "pi80_coverage": pi80_coverage,
                "pi80_width": pi80_width,
                "wis": wis,
            }
            for field, values in batch_values.items():
                per_horizon_sums[field] += np.where(valid, values, 0.0).sum(
                    axis=0, dtype=np.float64
                )
            per_horizon_counts += valid.sum(axis=0)

            # per-window abs err for bootstrap
            valid_counts = valid.sum(axis=1)
            if bootstrap > 0 and np.any(valid_counts != horizon):
                incomplete = [
                    str(bd.ts_utc[origin])
                    for origin, count in zip(batch_origins, valid_counts, strict=True)
                    if count != horizon
                ]
                raise ValueError(
                    f"{ba} bootstrap requires all {horizon} target hours finite at every "
                    f"origin; incomplete origins={incomplete}"
                )
            window_abs = np.divide(
                np.where(valid, abs_error, 0).sum(axis=1, dtype=np.float64),
                valid_counts,
                out=np.full(len(tasks), np.nan, dtype=np.float64),
                where=valid_counts > 0,
            )
            scaled_windows = window_abs / bd.denom_mae
            for origin, scaled_window in zip(batch_origins, scaled_windows, strict=True):
                origin_timestamp = bd.ts_utc[origin].astype("datetime64[us]")
                if np.isnat(origin_timestamp):
                    raise ValueError(f"{ba} has a NaT forecast origin at index {origin}")
                origin_key = int(origin_timestamp.astype(np.int64))
                if origin_key in origin_mase:
                    raise ValueError(f"{ba} has duplicate forecast origin {origin_timestamp}")
                origin_mase[origin_key] = float(scaled_window)

        point_count = int(per_horizon_counts.sum())
        if point_count == 0:
            raise ValueError(f"{ba} has zero finite target points across evaluated {on} origins")
        if not any(np.isfinite(value) for value in origin_mase.values()):
            raise ValueError(f"{ba} has zero finite per-window MASE values")
        mase_by_ba_origin[ba] = origin_mase
        total_sums = {
            field: float(values.sum()) for field, values in per_horizon_sums.items()
        }
        summary = _summarize_metric_sums(
            total_sums,
            count=point_count,
            denom_mae=bd.denom_mae,
            quantile_count=len(q_levels),
        )
        per_ba[ba] = {
            **summary,
            "n_windows": len(origins),
            "n_points": point_count,
            "candidate_origins_before_limit": candidate_origin_count,
            "incomplete_target_origins_before_limit": incomplete_target_origins,
            "complete_target_origins_only": require_complete_origins,
            "point_estimate_kind": POINT_ESTIMATE_KIND,
            "point_estimate_quantile": POINT_ESTIMATE_LABEL,
            "origin_start_utc": str(bd.ts_utc[origins[0]]),
            "origin_end_utc": str(bd.ts_utc[origins[-1]]),
            "origin_step_hours": step,
            "origin_sha256": hashlib.sha256(
                np.asarray(sorted(origin_mase), dtype="<i8").tobytes()
            ).hexdigest(),
        }
        if emit_origin_metrics:
            per_ba[ba]["origin_mase"] = [
                {
                    "origin_utc": str(np.datetime64(origin_key, "us")),
                    "mase": origin_mase[origin_key],
                }
                for origin_key in sorted(origin_mase)
            ]
        if per_step:
            missing_horizons = (np.flatnonzero(per_horizon_counts == 0) + 1).tolist()
            if missing_horizons:
                raise ValueError(
                    f"{ba} has zero finite target points at horizons {missing_horizons}"
                )
            per_ba[ba]["per_horizon"] = [
                {
                    "horizon": horizon_index + 1,
                    "n_points": int(per_horizon_counts[horizon_index]),
                    **_summarize_metric_sums(
                        {
                            field: float(values[horizon_index])
                            for field, values in per_horizon_sums.items()
                        },
                        count=int(per_horizon_counts[horizon_index]),
                        denom_mae=bd.denom_mae,
                        quantile_count=len(q_levels),
                    ),
                }
                for horizon_index in range(horizon)
            ]

    if not per_ba:
        raise ValueError(f"no evaluable {on} origins")

    # The top-level metrics are always equal-BA macro averages. Point count or
    # load never changes a BA's weight here; the optional load-weighted view is
    # emitted under its own explicitly named object below.
    macro: dict[str, Any] = {
        key: float(np.mean([float(value[key]) for value in per_ba.values()]))
        for key in _AGGREGATE_METRIC_KEYS
    }
    macro["aggregation"] = "equal_ba_macro"
    macro["per_ba"] = per_ba
    macro["n_bas"] = len(per_ba)
    macro["point_estimate_kind"] = POINT_ESTIMATE_KIND
    macro["point_estimate_quantile"] = POINT_ESTIMATE_LABEL
    macro["point_estimate_quantile_value"] = POINT_ESTIMATE_QUANTILE
    macro["split"] = on
    macro["horizon"] = horizon
    macro["origin_step_hours"] = step
    macro["max_origins"] = max_origins
    macro["complete_target_origins_only"] = require_complete_origins
    macro["shared_origin_schedule"] = origin_schedule is not None
    macro["origin_metrics_emitted"] = emit_origin_metrics
    if origin_schedule is not None:
        macro["origin_schedule"] = origin_schedule.as_dict()
    macro["crps_approximation"] = "2x_mean_pinball"
    macro["crps_approx_quantile_levels"] = q_levels
    mase_values = {ba: float(value["mase"]) for ba, value in per_ba.items()}
    macro["mase_std"] = float(np.std(list(mase_values.values())))
    macro["mase_min"] = min(mase_values.values())
    macro["mase_max"] = max(mase_values.values())
    macro["worst_mase_ba"] = max(mase_values, key=mase_values.__getitem__)

    if bootstrap > 0:
        low, high = _paired_origin_block_mase_ci(
            mase_by_ba_origin,
            bootstrap=bootstrap,
            seed=seed,
        )
        macro["mase_ci_low"] = low
        macro["mase_ci_high"] = high

    load_weighted = _load_weighted_aggregate(per_ba)
    if load_weighted is not None:
        load_weighted["n_bas"] = len(per_ba)
        load_weighted["n_points"] = sum(int(row["n_points"]) for row in per_ba.values())

    if per_step:
        per_horizon_macro: list[dict[str, Any]] = []
        per_horizon_weighted: list[dict[str, Any]] = []
        weighted_horizons_complete = load_weighted is not None
        for horizon_index in range(horizon):
            horizon_rows = {
                ba: row["per_horizon"][horizon_index] for ba, row in per_ba.items()
            }
            macro_row: dict[str, Any] = {
                key: float(
                    np.mean([float(row[key]) for row in horizon_rows.values()])
                )
                for key in _AGGREGATE_METRIC_KEYS
            }
            macro_row.update(
                {
                    "horizon": horizon_index + 1,
                    "n_bas": len(horizon_rows),
                    "n_points": sum(int(row["n_points"]) for row in horizon_rows.values()),
                    "aggregation": "equal_ba_macro",
                }
            )
            per_horizon_macro.append(macro_row)

            weighted_row = _load_weighted_aggregate(horizon_rows)
            if weighted_row is None:
                weighted_horizons_complete = False
            elif weighted_horizons_complete:
                weighted_row.update(
                    {
                        "horizon": horizon_index + 1,
                        "n_bas": len(horizon_rows),
                        "n_points": sum(
                            int(row["n_points"]) for row in horizon_rows.values()
                        ),
                    }
                )
                per_horizon_weighted.append(weighted_row)

        macro["per_horizon"] = per_horizon_macro
        # Backwards-compatible projection for existing plotting code.
        macro["per_step_mase"] = [row["mase"] for row in per_horizon_macro]
        if load_weighted is not None and weighted_horizons_complete:
            load_weighted["per_horizon"] = per_horizon_weighted

    if load_weighted is not None:
        macro["load_weighted"] = load_weighted

    return macro
