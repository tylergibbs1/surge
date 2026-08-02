"""Canonical column conventions for every Surge dataset.

All timestamps UTC. Energy in MW / MWh. Price in $/MWh.
Every table carries `source` and `as_of` for lineage.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

LOAD = {
    "ts_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "ba": pl.Utf8,
    "load_mw": pl.Float64,
    "source": pl.Utf8,
    "as_of": pl.Datetime(time_unit="us", time_zone="UTC"),
}

LMP = {
    "ts_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "iso": pl.Utf8,
    "node": pl.Utf8,
    "market": pl.Utf8,  # "DA" or "RT"
    "lmp_usd_per_mwh": pl.Float64,
    "energy_usd_per_mwh": pl.Float64,
    "congestion_usd_per_mwh": pl.Float64,
    "losses_usd_per_mwh": pl.Float64,
    "source": pl.Utf8,
    "as_of": pl.Datetime(time_unit="us", time_zone="UTC"),
}

GEN_BY_FUEL = {
    "ts_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "ba": pl.Utf8,
    "fuel": pl.Utf8,
    "gen_mw": pl.Float64,
    "source": pl.Utf8,
    "as_of": pl.Datetime(time_unit="us", time_zone="UTC"),
}

FORECAST_ISSUANCE = {
    "schema_version": pl.Int16,
    "issuance_id": pl.Utf8,
    "run_id": pl.Utf8,
    "mode": pl.Utf8,
    "ba": pl.Utf8,
    "target_name": pl.Utf8,
    "units": pl.Utf8,
    "scheduled_for_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "feature_cutoff_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "issued_at_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "first_valid_at_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "last_valid_at_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "context_start_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "context_end_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "horizon_hours": pl.Int16,
    "frequency_minutes": pl.Int16,
    "model_name": pl.Utf8,
    "model_revision": pl.Utf8,
    "model_artifact_sha256": pl.Utf8,
    "code_revision": pl.Utf8,
    "feature_spec_version": pl.Utf8,
    "feature_spec_sha256": pl.Utf8,
    "feature_snapshot_sha256": pl.Utf8,
    "availability_mode": pl.Utf8,
    "point_estimate_kind": pl.Utf8,
    "mase_scale_24": pl.Float64,
    "points_sha256": pl.Utf8,
    "point_count": pl.Int16,
    "warnings_json": pl.Utf8,
}

FORECAST_POINT = {
    "schema_version": pl.Int16,
    "issuance_id": pl.Utf8,
    "valid_at_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "horizon_step": pl.Int16,
    "lead_minutes_from_issue": pl.Int32,
    "mean_mw": pl.Float64,
    "p10_mw": pl.Float64,
    "p50_mw": pl.Float64,
    "p90_mw": pl.Float64,
    "future_temp_c": pl.Float64,
    "future_temp_vintage_id": pl.Utf8,
}

FORECAST_RUN = {
    "schema_version": pl.Int16,
    "run_id": pl.Utf8,
    "mode": pl.Utf8,
    "scheduled_for_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "published_at_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "required_bas_json": pl.Utf8,
    "issuance_ids_json": pl.Utf8,
    "feature_snapshots_json": pl.Utf8,
    "points_sha256s_json": pl.Utf8,
    "target_name": pl.Utf8,
    "units": pl.Utf8,
    "horizon_hours": pl.Int16,
    "frequency_minutes": pl.Int16,
    "quantiles_json": pl.Utf8,
    "model_name": pl.Utf8,
    "model_revision": pl.Utf8,
    "model_artifact_sha256": pl.Utf8,
    "code_revision": pl.Utf8,
    "feature_spec_version": pl.Utf8,
    "feature_spec_sha256": pl.Utf8,
    "availability_mode": pl.Utf8,
    "point_estimate_kind": pl.Utf8,
    "run_content_sha256": pl.Utf8,
}

FORECAST_VERIFICATION = {
    "schema_version": pl.Int16,
    "verification_id": pl.Utf8,
    "issuance_id": pl.Utf8,
    "ba": pl.Utf8,
    "valid_at_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "verified_at_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "outcome_cutoff_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "outcome_policy_version": pl.Utf8,
    "outcome_observation_id": pl.Utf8,
    "outcome_as_of_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "actual_mw": pl.Float64,
    "point_estimate_mw": pl.Float64,
    "point_estimate_kind": pl.Utf8,
    "error_mw": pl.Float64,
    "abs_error_mw": pl.Float64,
    "squared_error_mw2": pl.Float64,
    "pinball_p10": pl.Float64,
    "pinball_p50": pl.Float64,
    "pinball_p90": pl.Float64,
    "inside_p10_p90": pl.Boolean,
    "interval_score_80": pl.Float64,
    "wis": pl.Float64,
    "metric_version": pl.Utf8,
}


def enforce(df: pl.DataFrame, schema: Mapping[str, Any]) -> pl.DataFrame:
    """Reorder and cast a frame to match a canonical schema. Missing cols raise."""
    missing = set(schema) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df.select([pl.col(c).cast(t) for c, t in schema.items()])
