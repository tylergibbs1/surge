"""Pure, point-in-time-safe forecasting logic for the FastAPI app."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from surge import calibration as calibration_module
from surge import store
from surge.features import (
    DEFAULT_QUANTILE_LEVELS,
    LOAD_V2_CORE,
    POINT_ESTIMATE_KIND,
    POINT_ESTIMATE_LABEL,
    AvailabilityMode,
    build_live_bundle,
    numpy_to_datetime,
)
from surge.model_loader import artifact_sha256
from surge.verification import MATURITY_HOURS

# The old Surge v3 checkpoint used a legacy feature/training contract and must
# not be presented as a v0.2 no-peeking model. Default to a pinned upstream
# Chronos-2 artifact; a v0.2 fine-tune can be selected explicitly by path and
# immutable revision.
DEFAULT_CHRONOS2_MODEL = "amazon/chronos-2"
PINNED_CHRONOS2_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
UNKNOWN_MODEL_REVISION = "unknown"


def _resolve_model_revision(model_path: str, configured_revision: str | None) -> str:
    if configured_revision:
        return configured_revision
    if model_path == DEFAULT_CHRONOS2_MODEL:
        return PINNED_CHRONOS2_REVISION
    return UNKNOWN_MODEL_REVISION


MODEL_PATH = os.environ.get("SURGE_MODEL_PATH", DEFAULT_CHRONOS2_MODEL)
MODEL_REVISION = _resolve_model_revision(
    MODEL_PATH,
    os.environ.get("SURGE_MODEL_REVISION"),
)
CODE_REVISION = os.environ.get("SURGE_CODE_REVISION", "unknown")
MODEL_NAME = os.environ.get(
    "SURGE_MODEL_NAME",
    MODEL_PATH.rstrip("/").rsplit("/", 1)[-1],
)
MODEL_FEATURE_SPEC_SHA256 = os.environ.get("SURGE_MODEL_FEATURE_SPEC_SHA256")
MODEL_ARTIFACT_SHA256 = os.environ.get("SURGE_MODEL_ARTIFACT_SHA256")
CONTEXT_LENGTH = 2048


def model_load_revision() -> str | None:
    """Return a real Hub revision for loading, never the provenance sentinel."""
    if MODEL_REVISION == UNKNOWN_MODEL_REVISION:
        return None
    if Path(MODEL_PATH).expanduser().exists():
        return None
    return MODEL_REVISION


def _model_attestation() -> tuple[bool, str | None]:
    pinned_upstream = (
        MODEL_PATH == DEFAULT_CHRONOS2_MODEL
        and MODEL_REVISION == PINNED_CHRONOS2_REVISION
    )
    if MODEL_PATH == DEFAULT_CHRONOS2_MODEL:
        return pinned_upstream, None

    try:
        computed_sha256 = artifact_sha256(MODEL_PATH)
    except (OSError, RuntimeError, ValueError):
        return False, None

    custom_attested = (
        MODEL_REVISION != UNKNOWN_MODEL_REVISION
        and LOAD_V2_CORE.sha256 == MODEL_FEATURE_SPEC_SHA256
        and computed_sha256 == MODEL_ARTIFACT_SHA256
    )
    return custom_attested, computed_sha256


def model_release_safe() -> bool:
    """Whether the configured artifact has an immutable no-peeking identity."""
    release_safe, _ = _model_attestation()
    return release_safe


def model_artifact_identity() -> str | None:
    """Return the computed local artifact identity used in provenance."""
    _, computed_sha256 = _model_attestation()
    return computed_sha256


def data_end_utc() -> datetime | None:
    """Compatibility accessor for the load-source watermark."""
    return data_watermarks_utc()["load_hourly"]


def data_watermarks_utc() -> dict[str, datetime | None]:
    """Return last non-null valid times for every required live source."""
    watermarks: dict[str, datetime | None] = {}
    for name, value_column in (
        ("load_hourly", "load_mw"),
        ("weather_hourly", "temp_c"),
    ):
        try:
            lazy = store.scan(name)
            names = lazy.collect_schema().names()
            if not {"ts_utc", value_column}.issubset(names):
                watermarks[name] = None
                continue
            df = (
                lazy.filter(pl.col(value_column).is_not_null())
                .select(pl.col("ts_utc").max().alias("m"))
                .collect()
            )
            watermarks[name] = None if df.is_empty() else df["m"][0]
        except Exception:
            watermarks[name] = None
    return watermarks


def _as_numpy(value: Any) -> np.ndarray:
    """Convert torch-like model output without importing torch in this module."""
    for method in ("float", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _squeezed_output(value: Any, *, expected_ndim: int) -> np.ndarray:
    result = _as_numpy(value)
    if result.ndim == expected_ndim + 1 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != expected_ndim:
        raise ValueError(f"unexpected model output shape {result.shape}")
    return result


def forecast_ba(
    pipe: Any,
    ba: str,
    horizon: int = 24,
    *,
    issued_at_utc: datetime | None = None,
    feature_cutoff_utc: datetime | None = None,
    cutoff_utc: datetime | None = None,
    availability_mode: AvailabilityMode | str = AvailabilityMode.EXACT_VINTAGE,
) -> dict[str, Any]:
    """Produce one probabilistic forecast from a frozen point-in-time snapshot."""
    if horizon < 1 or horizon > 168:
        raise ValueError("horizon must be in 1..168")
    if feature_cutoff_utc is not None and cutoff_utc is not None:
        raise ValueError("pass only feature_cutoff_utc; cutoff_utc is a compatibility alias")

    # Freeze wall-clock values before the first datastore read. A multi-read
    # feature build therefore has one coherent availability boundary.
    issued = issued_at_utc or datetime.now(tz=UTC)
    feature_cutoff = feature_cutoff_utc or cutoff_utc or issued
    mode = AvailabilityMode(availability_mode)
    bundle = build_live_bundle(
        ba,
        issued_at_utc=issued,
        cutoff_utc=feature_cutoff,
        horizon=horizon,
        context_length=CONTEXT_LENGTH,
        availability_mode=mode,
        include_generation=False,
        spec=LOAD_V2_CORE,
    )

    quantiles_list, means_list = pipe.predict_quantiles(
        [bundle.task],
        prediction_length=bundle.prediction_length,
        quantile_levels=list(DEFAULT_QUANTILE_LEVELS),
        batch_size=1,
    )
    quantiles = _squeezed_output(quantiles_list[0], expected_ndim=2)
    means = _squeezed_output(means_list[0], expected_ndim=1)
    if quantiles.shape != (bundle.prediction_length, len(DEFAULT_QUANTILE_LEVELS)):
        raise ValueError(f"unexpected quantile output shape {quantiles.shape}")
    if means.shape != (bundle.prediction_length,):
        raise ValueError(f"unexpected mean output shape {means.shape}")

    start = bundle.discard_prefix
    stop = start + horizon
    quantiles = quantiles[start:stop]
    means = means[start:stop]

    # The model's raw 80% band has measured ~0.73-0.76 coverage. Calibrate it
    # against this BA's own settled history before publishing. Fails open: with
    # too little matured history the model interval is served unchanged and the
    # issuance says so.
    calibrated_low, calibrated_high, calibration = calibration_module.calibrate(
        quantiles[:, 0],
        quantiles[:, 2],
        history=calibration_module.matured_history(
            ba,
            now_utc=bundle.issued_at_utc,
            horizon=horizon,
            outcome_delay_hours=MATURITY_HOURS,
        ),
        issued_at_utc=numpy_to_datetime(bundle.served_ts_utc[0]),
    )

    points: list[dict[str, Any]] = []
    for index, timestamp in enumerate(bundle.served_ts_utc):
        valid_at = numpy_to_datetime(timestamp)
        if valid_at <= bundle.issued_at_utc:
            raise ValueError("served forecast timestamps must be strictly after issuance")
        p50 = float(quantiles[index, 1])
        points.append(
            {
                "ts_utc": valid_at,
                "valid_at_utc": valid_at,
                "horizon_step": index + 1,
                "lead_minutes_from_issue": int(
                    (valid_at - bundle.issued_at_utc).total_seconds() // 60
                ),
                "mean_mw": float(means[index]),
                "p10_mw": float(calibrated_low[index]),
                "p50_mw": p50,
                "median_mw": p50,
                "p90_mw": float(calibrated_high[index]),
                "uncalibrated_p10_mw": float(quantiles[index, 0]),
                "uncalibrated_p90_mw": float(quantiles[index, 2]),
                # LOAD_V2_CORE has no future temperature input. Keep nullable
                # compatibility fields explicit so clients cannot mistake an
                # observed or persisted value for a weather forecast.
                "temp_c": None,
                "future_temp_c": None,
                "future_temp_vintage_id": None,
            }
        )

    release_safe, computed_artifact_sha256 = _model_attestation()
    provenance: dict[str, Any] = {
        **bundle.provenance,
        "model_revision": MODEL_REVISION,
        "model_artifact_sha256": computed_artifact_sha256,
        "code_revision": CODE_REVISION,
        "feature_snapshot_sha256": bundle.feature_snapshot_sha256,
        "model_feature_spec_sha256": MODEL_FEATURE_SPEC_SHA256,
        "model_release_safe": release_safe,
        **calibration.as_provenance(),
    }
    return {
        "points": points,
        "issued_at_utc": bundle.issued_at_utc,
        "feature_cutoff_utc": bundle.cutoff_utc,
        "cutoff_utc": bundle.cutoff_utc,
        "forecast_start_utc": bundle.forecast_start_utc,
        "context_start_utc": numpy_to_datetime(bundle.context_ts_utc[0]),
        "context_end_utc": numpy_to_datetime(bundle.context_ts_utc[-1]),
        "feature_spec_version": bundle.feature_spec.version,
        "feature_spec_sha256": bundle.feature_spec.sha256,
        "feature_snapshot_sha256": bundle.feature_snapshot_sha256,
        "availability_mode": bundle.availability_mode.value,
        "point_estimate_kind": POINT_ESTIMATE_KIND,
        "point_estimate_quantile": POINT_ESTIMATE_LABEL,
        "mase_scale_24": bundle.mase_scale_24,
        "mase_scale": bundle.mase_scale_24,
        "source_lag_hours": bundle.discard_prefix,
        "model_prediction_length": bundle.prediction_length,
        "warnings": list(bundle.warnings),
        "provenance": provenance,
    }
