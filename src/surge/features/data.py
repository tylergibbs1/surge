"""Point-in-time data loading and live feature snapshot construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from surge import store
from surge.features.calendar import (
    as_utc,
    calendar_covariates,
    datetime_to_numpy,
    next_full_utc_hour,
    numpy_to_datetime,
)
from surge.features.spec import LOAD_V2_CORE, AvailabilityMode, FeatureSpec, validate_task

_ONE_HOUR = np.timedelta64(1, "h")
MAX_LIVE_SOURCE_LAG_HOURS = 12
_REQUIRED_LIVE_SOURCES = ("load_hourly", "weather_hourly")


@dataclass
class BAData:
    """Aligned hourly history used by training and rolling evaluation."""

    ba: str
    ts_utc: np.ndarray
    target: np.ndarray
    covariates: dict[str, np.ndarray]
    future_keys: list[str]
    train_end: int
    val_end: int
    denom_mae: float
    feature_spec: FeatureSpec = LOAD_V2_CORE
    availability_mode: AvailabilityMode = AvailabilityMode.RETROSPECTIVE_FINAL
    warnings: tuple[str, ...] = ()
    provenance: dict[str, dict[str, Any]] = field(default_factory=dict)

    def slice(self, start: int, end: int) -> dict[str, Any]:
        return {
            "target": self.target[start:end],
            "past_covariates": {key: value[start:end] for key, value in self.covariates.items()},
        }

    def future_dict(self, start: int, end: int) -> dict[str, np.ndarray]:
        return {key: self.covariates[key][start:end] for key in self.future_keys}

    @property
    def feature_spec_version(self) -> str:
        return self.feature_spec.version


@dataclass(frozen=True)
class ForecastFeatureBundle:
    """Frozen model inputs and timing metadata for one forecast issuance."""

    ba: str
    issued_at_utc: datetime
    cutoff_utc: datetime
    availability_mode: AvailabilityMode
    feature_spec: FeatureSpec
    context_ts_utc: np.ndarray
    model_future_ts_utc: np.ndarray
    served_ts_utc: np.ndarray
    task: dict[str, Any]
    discard_prefix: int
    mase_scale_24: float
    feature_snapshot_sha256: str
    provenance: dict[str, dict[str, Any]]
    warnings: tuple[str, ...]

    @property
    def prediction_length(self) -> int:
        return len(self.model_future_ts_utc)

    @property
    def horizon(self) -> int:
        return len(self.served_ts_utc)

    @property
    def forecast_start_utc(self) -> datetime:
        return numpy_to_datetime(self.served_ts_utc[0])


def _frame_provenance(
    name: str,
    frame: pl.DataFrame,
    *,
    availability_col: str = "as_of",
    value_column: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"dataset": name, "rows": frame.height}
    if frame.is_empty():
        return result
    if "ts_utc" in frame.columns:
        result["valid_start_utc"] = frame["ts_utc"].min()
        result["valid_end_utc"] = frame["ts_utc"].max()
        if value_column is not None and value_column in frame.columns:
            observed = frame.filter(pl.col(value_column).is_not_null())
            result["observed_end_utc"] = (
                observed["ts_utc"].max() if not observed.is_empty() else None
            )
    if availability_col in frame.columns:
        result["max_available_at_utc"] = frame[availability_col].max()
    if "source" in frame.columns:
        result["sources"] = sorted(str(value) for value in frame["source"].drop_nulls().unique())
    return result


def _collect_dataset(
    name: str,
    *,
    ba: str,
    dedupe_on: list[str],
    mode: AvailabilityMode,
    cutoff: datetime | None,
    valid_before: datetime | None,
) -> pl.DataFrame:
    if mode is AvailabilityMode.EXACT_VINTAGE:
        if cutoff is None:
            raise ValueError("exact_vintage reads require a cutoff")
        lazy = store.scan_as_of(name, cutoff=cutoff, dedupe_on=dedupe_on)
    else:
        if cutoff is not None:
            raise ValueError("retrospective_final reads do not accept an availability cutoff")
        lazy = store.scan(name, dedupe_on=dedupe_on)

    names = lazy.collect_schema().names()
    if not names:
        return pl.DataFrame()
    required = {"ts_utc", "ba"}
    missing = required - set(names)
    if missing:
        raise ValueError(f"dataset {name!r} missing columns: {sorted(missing)}")
    lazy = lazy.filter(pl.col("ba") == ba)
    if valid_before is not None:
        lazy = lazy.filter(pl.col("ts_utc") < valid_before)
    if cutoff is not None:
        # Availability and valid time are separate. Observations about a future
        # valid time cannot enter context even if a malformed source published
        # them before cutoff.
        lazy = lazy.filter(pl.col("ts_utc") <= cutoff)
    return lazy.sort("ts_utc").collect()


def _regularize_hourly(
    joined: pl.DataFrame,
    *,
    value_columns: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    raw_source_ts = joined["ts_utc"].to_numpy().astype("datetime64[us]")
    source_ts = raw_source_ts.astype("datetime64[h]")
    if len(source_ts) == 0:
        raise ValueError("cannot regularize an empty series")
    if not np.all(raw_source_ts == source_ts.astype("datetime64[us]")):
        raise ValueError("feature timestamps must be aligned to whole UTC hours")
    if len(np.unique(source_ts)) != len(source_ts):
        raise ValueError("duplicate hourly timestamps remain after point-in-time deduplication")

    grid = np.arange(source_ts[0], source_ts[-1] + _ONE_HOUR, _ONE_HOUR).astype("datetime64[us]")
    offsets = ((source_ts - source_ts[0]) / _ONE_HOUR).astype(np.int64)
    values: dict[str, np.ndarray] = {}
    warnings: list[str] = []
    missing_hours = len(grid) - len(source_ts)
    if missing_hours:
        warnings.append(f"inserted {missing_hours} missing hourly rows")

    for column in value_columns:
        aligned = np.full(len(grid), np.nan, dtype=np.float64)
        if column in joined.columns:
            raw = joined[column].cast(pl.Float64, strict=False).to_numpy()
            aligned[offsets] = raw
        # Chronos-2 represents historical missingness explicitly. Preserve NaN
        # instead of manufacturing arbitrarily stale observations via an
        # unbounded forward fill. Live issuance separately enforces each
        # source's own valid-time watermark below.
        values[column] = aligned
        count = int(np.count_nonzero(~np.isfinite(aligned)))
        if count:
            warnings.append(f"preserved {count} missing {column} values")
    return grid, values, warnings


def seasonal_naive_scale(target: np.ndarray, *, period: int = 24) -> float:
    target = np.asarray(target, dtype=np.float64)
    if len(target) <= period:
        return float("nan")
    differences = np.abs(target[period:] - target[:-period])
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return float("nan")
    scale = float(differences.mean())
    return scale if scale > 0 else float("nan")


def load_ba_data(
    ba: str,
    *,
    availability_mode: AvailabilityMode | str = AvailabilityMode.RETROSPECTIVE_FINAL,
    cutoff: datetime | None = None,
    valid_before: datetime | None = None,
    include_generation: bool = False,
    spec: FeatureSpec = LOAD_V2_CORE,
) -> BAData:
    """Load aligned history under an explicit observation-vintage policy.

    ``valid_before`` is an exclusive valid-time boundary applied in the lazy
    query, before rows are materialized. Model-selection code uses it to keep
    the locked test partition out of process memory entirely.
    """
    mode = AvailabilityMode(availability_mode)
    normalized_cutoff = as_utc(cutoff, field="cutoff") if cutoff is not None else None
    normalized_valid_before = (
        as_utc(valid_before, field="valid_before") if valid_before is not None else None
    )
    warnings: list[str] = []
    if mode is AvailabilityMode.RETROSPECTIVE_FINAL:
        warnings.append(
            "retrospective_final may use revisions published after historical forecast times"
        )

    load = _collect_dataset(
        "load_hourly",
        ba=ba,
        dedupe_on=["ts_utc", "ba"],
        mode=mode,
        cutoff=normalized_cutoff,
        valid_before=normalized_valid_before,
    )
    if load.is_empty():
        raise ValueError(f"no load history for {ba} in {mode.value} mode")
    required_load = {"ts_utc", "load_mw"}
    if missing := required_load - set(load.columns):
        raise ValueError(f"load_hourly missing columns: {sorted(missing)}")
    load = load.with_columns(
        pl.when(pl.col("load_mw").is_between(0, 200_000, closed="right"))
        .then(pl.col("load_mw"))
        .otherwise(None)
        .alias("load_mw")
    )

    weather = _collect_dataset(
        "weather_hourly",
        ba=ba,
        dedupe_on=["ts_utc", "ba"],
        mode=mode,
        cutoff=normalized_cutoff,
        valid_before=normalized_valid_before,
    )
    if weather.is_empty() or "temp_c" not in weather.columns:
        raise ValueError(f"no observed temperature history for {ba} in {mode.value} mode")

    provenance = {
        "load_hourly": _frame_provenance("load_hourly", load, value_column="load_mw"),
        "weather_hourly": _frame_provenance("weather_hourly", weather, value_column="temp_c"),
    }
    joined = load.select("ts_utc", "load_mw").join(
        weather.select("ts_utc", "temp_c"), on="ts_utc", how="left"
    )
    value_columns = ["load_mw", "temp_c"]

    if include_generation:
        generation = _collect_dataset(
            "gen_by_fuel",
            ba=ba,
            dedupe_on=["ts_utc", "ba", "fuel"],
            mode=mode,
            cutoff=normalized_cutoff,
            valid_before=normalized_valid_before,
        )
        if generation.is_empty() or not {"fuel", "gen_mw"}.issubset(generation.columns):
            warnings.append("generation requested but no wind/solar observations were available")
        else:
            generation = generation.filter(pl.col("fuel").is_in(["WND", "SUN"]))
            provenance["gen_by_fuel"] = _frame_provenance(
                "gen_by_fuel", generation, value_column="gen_mw"
            )
            if not generation.is_empty():
                wide = generation.select("ts_utc", "fuel", "gen_mw").pivot(
                    values="gen_mw", index="ts_utc", on="fuel", aggregate_function="mean"
                )
                wide = wide.rename(
                    {
                        key: value
                        for key, value in {"WND": "wind_mw", "SUN": "solar_mw"}.items()
                        if key in wide.columns
                    }
                )
                generation_columns = [
                    column for column in ("wind_mw", "solar_mw") if column in wide.columns
                ]
                joined = joined.join(wide, on="ts_utc", how="left")
                value_columns.extend(generation_columns)

    ts_utc, aligned, alignment_warnings = _regularize_hourly(
        joined.sort("ts_utc"), value_columns=value_columns
    )
    warnings.extend(alignment_warnings)

    usable = np.ones(len(ts_utc), dtype=bool)
    for column in value_columns:
        usable &= np.isfinite(aligned[column])
    usable_indices = np.flatnonzero(usable)
    if not len(usable_indices):
        raise ValueError(f"no complete feature history for {ba}")
    first = int(usable_indices[0])
    if first:
        warnings.append(f"trimmed {first} leading rows instead of backward-filling them")
    ts_utc = ts_utc[first:]
    aligned = {key: value[first:] for key, value in aligned.items()}

    target = aligned.pop("load_mw").astype(np.float32)
    covariates = {
        "temp_c": aligned.pop("temp_c").astype(np.float32),
        **calendar_covariates(ts_utc),
        **{key: value.astype(np.float32) for key, value in aligned.items()},
    }
    years = ts_utc.astype("datetime64[Y]").astype(np.int64) + 1970
    train_end = int(np.searchsorted(years, 2024, side="left"))
    val_end = int(np.searchsorted(years, 2025, side="left"))
    scale_source = target[:train_end]
    if len(scale_source) <= 24:
        scale_source = target[:val_end]
    if len(scale_source) <= 24:
        scale_source = target
    denom = seasonal_naive_scale(scale_source)
    if not np.isfinite(denom):
        warnings.append("24-hour seasonal-naive MASE scale is unavailable")

    return BAData(
        ba=ba,
        ts_utc=ts_utc,
        target=target,
        covariates=covariates,
        future_keys=list(spec.future_covariates),
        train_end=train_end,
        val_end=val_end,
        denom_mae=denom,
        feature_spec=spec,
        availability_mode=mode,
        warnings=tuple(dict.fromkeys(warnings)),
        provenance=provenance,
    )


def load_multi_ba_data(
    bas: list[str],
    *,
    availability_mode: AvailabilityMode | str = AvailabilityMode.RETROSPECTIVE_FINAL,
    cutoff: datetime | None = None,
    valid_before: datetime | None = None,
    include_generation: bool = False,
    spec: FeatureSpec = LOAD_V2_CORE,
) -> dict[str, BAData]:
    return {
        ba: load_ba_data(
            ba,
            availability_mode=availability_mode,
            cutoff=cutoff,
            valid_before=valid_before,
            include_generation=include_generation,
            spec=spec,
        )
        for ba in bas
    }


def _snapshot_hash(
    context_ts: np.ndarray,
    model_future_ts: np.ndarray,
    task: dict[str, Any],
    *,
    ba: str,
    cutoff: datetime,
    availability_mode: AvailabilityMode,
    provenance: dict[str, dict[str, Any]],
    spec: FeatureSpec,
) -> str:
    digest = hashlib.sha256()
    digest.update(spec.sha256.encode())
    metadata = json.dumps(
        {
            "ba": ba,
            "cutoff_utc": cutoff.isoformat(),
            "availability_mode": availability_mode.value,
            "provenance": provenance,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: value.isoformat() if isinstance(value, datetime) else str(value),
    )
    digest.update(metadata.encode())
    digest.update(context_ts.astype("datetime64[us]").astype(np.int64).tobytes())
    digest.update(model_future_ts.astype("datetime64[us]").astype(np.int64).tobytes())
    digest.update(np.asarray(task["target"], dtype=np.float32).tobytes())
    for section in ("past_covariates", "future_covariates"):
        for key in sorted(task[section]):
            digest.update(section.encode())
            digest.update(key.encode())
            digest.update(np.asarray(task[section][key], dtype=np.float32).tobytes())
    return digest.hexdigest()


def build_live_bundle_from_data(
    data: BAData,
    *,
    issued_at_utc: datetime,
    cutoff_utc: datetime,
    horizon: int,
    context_length: int,
    spec: FeatureSpec = LOAD_V2_CORE,
) -> ForecastFeatureBundle:
    """Freeze a no-peeking live task and compensate for source publication lag."""
    issued = as_utc(issued_at_utc, field="issued_at_utc")
    cutoff = as_utc(cutoff_utc, field="cutoff_utc")
    if cutoff > issued:
        raise ValueError("cutoff_utc cannot be after issued_at_utc")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if context_length < 1:
        raise ValueError("context_length must be positive")

    cutoff_np = datetime_to_numpy(cutoff)
    eligible = np.flatnonzero(data.ts_utc.astype("datetime64[us]") <= cutoff_np)
    if len(eligible) < context_length:
        raise ValueError(f"not enough history for {data.ba} (need {context_length}h by cutoff)")
    end = int(eligible[-1]) + 1
    start = end - context_length
    context_ts = data.ts_utc[start:end].astype("datetime64[us]")
    if len(context_ts) > 1 and not np.all(np.diff(context_ts) == _ONE_HOUR):
        raise ValueError("model context must be contiguous and hourly")

    context_end_utc = numpy_to_datetime(context_ts[-1])
    max_lag_seconds = MAX_LIVE_SOURCE_LAG_HOURS * 60 * 60
    context_lag_seconds = (issued - context_end_utc).total_seconds()
    if context_lag_seconds > max_lag_seconds:
        raise ValueError(
            f"{data.ba} context is stale by {context_lag_seconds / 3600:.1f}h; "
            f"maximum is {MAX_LIVE_SOURCE_LAG_HOURS}h"
        )
    for source in _REQUIRED_LIVE_SOURCES:
        source_provenance = data.provenance.get(source)
        if not source_provenance:
            raise ValueError(f"missing required {source} provenance for live issuance")
        watermark = source_provenance.get("observed_end_utc")
        if not isinstance(watermark, datetime):
            raise ValueError(f"missing observed valid-time watermark for {source}")
        watermark = as_utc(watermark, field=f"{source}.observed_end_utc")
        if watermark > cutoff:
            raise ValueError(f"{source} watermark is after the frozen feature cutoff")
        lag_seconds = (issued - watermark).total_seconds()
        if lag_seconds > max_lag_seconds:
            raise ValueError(
                f"{source} is stale by {lag_seconds / 3600:.1f}h; "
                f"maximum is {MAX_LIVE_SOURCE_LAG_HOURS}h"
            )

    past = {
        key: values[start:end].astype(np.float32)
        for key, values in data.covariates.items()
        if key in spec.allowed_past_covariates
    }
    context_end = context_ts[-1].astype("datetime64[h]")
    natural_start = context_end + _ONE_HOUR
    requested_start = datetime_to_numpy(next_full_utc_hour(issued)).astype("datetime64[h]")
    if natural_start > requested_start:
        # This should be impossible when valid times are filtered at a cutoff
        # no later than issuance. Keep it a hard failure rather than silently
        # shifting the public forecast window.
        raise ValueError("context extends beyond the first valid forecast hour")
    discard_prefix = int((requested_start - natural_start) / _ONE_HOUR)
    prediction_length = discard_prefix + horizon
    model_future_ts = np.arange(
        natural_start,
        natural_start + prediction_length * _ONE_HOUR,
        _ONE_HOUR,
    ).astype("datetime64[us]")
    served_ts = model_future_ts[discard_prefix:]
    future = calendar_covariates(model_future_ts)
    task: dict[str, Any] = {
        "target": data.target[start:end].astype(np.float32),
        "past_covariates": past,
        "future_covariates": future,
    }
    validate_task(task, prediction_length=prediction_length, spec=spec)

    warnings = list(data.warnings)
    if discard_prefix:
        warnings.append(
            f"source data lagged forecast start by {discard_prefix}h; gap predictions discarded"
        )
    if data.availability_mode is AvailabilityMode.RETROSPECTIVE_FINAL:
        warnings.append("retrospective_final is for historical experiments, not live claims")
    scale = seasonal_naive_scale(task["target"])
    if not np.isfinite(scale):
        warnings.append("24-hour seasonal-naive MASE scale is unavailable")

    return ForecastFeatureBundle(
        ba=data.ba,
        issued_at_utc=issued,
        cutoff_utc=cutoff,
        availability_mode=data.availability_mode,
        feature_spec=spec,
        context_ts_utc=context_ts,
        model_future_ts_utc=model_future_ts,
        served_ts_utc=served_ts,
        task=task,
        discard_prefix=discard_prefix,
        mase_scale_24=scale,
        feature_snapshot_sha256=_snapshot_hash(
            context_ts,
            model_future_ts,
            task,
            ba=data.ba,
            cutoff=cutoff,
            availability_mode=data.availability_mode,
            provenance=data.provenance,
            spec=spec,
        ),
        provenance=data.provenance,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def build_live_bundle(
    ba: str,
    *,
    issued_at_utc: datetime,
    cutoff_utc: datetime,
    horizon: int,
    context_length: int,
    availability_mode: AvailabilityMode | str = AvailabilityMode.EXACT_VINTAGE,
    include_generation: bool = False,
    spec: FeatureSpec = LOAD_V2_CORE,
) -> ForecastFeatureBundle:
    """Read then freeze one feature bundle under the selected vintage policy."""
    mode = AvailabilityMode(availability_mode)
    cutoff = as_utc(cutoff_utc, field="cutoff_utc")
    data = load_ba_data(
        ba,
        availability_mode=mode,
        cutoff=cutoff if mode is AvailabilityMode.EXACT_VINTAGE else None,
        include_generation=include_generation,
        spec=spec,
    )
    return build_live_bundle_from_data(
        data,
        issued_at_utc=issued_at_utc,
        cutoff_utc=cutoff,
        horizon=horizon,
        context_length=context_length,
        spec=spec,
    )
