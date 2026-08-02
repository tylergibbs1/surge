"""Point-in-time outcome verification and forecast scoring.

Forecasts are immutable. Verification rows are separate immutable artifacts
that pin the exact observation revision selected by a deterministic settlement
policy, so later EIA revisions cannot rewrite an existing score.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict

from surge import ledger, schemas, store

MATURITY_HOURS = 72
METRIC_VERSION = "surge-probabilistic-v1"


def outcome_policy_version(maturity_hours: int) -> str:
    """Bind a settlement policy identity to its observation maturity."""
    if isinstance(maturity_hours, bool) or not isinstance(maturity_hours, int):
        raise ValueError("maturity_hours must be an integer")
    if maturity_hours < 0:
        raise ValueError("maturity_hours must be nonnegative")
    return f"eia-latest-at-plus{maturity_hours}h-v1"


OUTCOME_POLICY_VERSION = outcome_policy_version(MATURITY_HOURS)


class ScoreSummary(BaseModel):
    """Derived metrics for one issuance or an explicitly selected window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    basis: str
    n_points: int
    mae_mw: float
    rmse_mw: float
    bias_mw: float
    wape_pct: float
    mase_24: float
    wis_mw: float
    pi80_coverage_pct: float
    pi80_calibration_error_pct: float
    scored_from_utc: datetime
    scored_through_utc: datetime


def pinball_loss(actual: float, quantile: float, tau: float) -> float:
    error = actual - quantile
    return tau * error if error >= 0 else (1.0 - tau) * -error


def interval_score_80(actual: float, lower: float, upper: float) -> float:
    """Central 80% interval score (alpha=0.2; lower is better)."""
    alpha = 0.2
    below = (2.0 / alpha) * (lower - actual) if actual < lower else 0.0
    above = (2.0 / alpha) * (actual - upper) if actual > upper else 0.0
    return (upper - lower) + below + above


def weighted_interval_score(
    actual: float,
    lower: float,
    median: float,
    upper: float,
) -> float:
    """WIS for p10/p50/p90, following the standard one-interval weighting."""
    alpha = 0.2
    return (
        0.5 * abs(actual - median)
        + (alpha / 2.0) * interval_score_80(actual, lower, upper)
    ) / 1.5


def _observation_id(row: dict[str, Any]) -> str:
    identity = {
        "source": row.get("source", ""),
        "ba": row["ba"],
        "valid_at_utc": row["ts_utc"].astimezone(UTC).isoformat(),
        "load_mw": float(row["load_mw"]),
        "as_of": row["as_of"].astimezone(UTC).isoformat(),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _verification_id(
    issuance_id: str,
    valid_at_utc: datetime,
    policy_version: str,
) -> str:
    identity = f"verification|{issuance_id}|{valid_at_utc.isoformat()}|{policy_version}"
    return str(uuid.uuid5(ledger.LEDGER_NAMESPACE, identity))


def _select_outcome(
    *,
    ba: str,
    valid_at_utc: datetime,
    outcome_cutoff_utc: datetime,
) -> dict[str, Any] | None:
    lf = store.scan_as_of(
        "load_hourly",
        cutoff=outcome_cutoff_utc,
        dedupe_on=["ts_utc", "ba"],
    )
    if not lf.collect_schema().names():
        return None
    df = (
        lf.filter(pl.col("ba") == ba)
        .filter(pl.col("ts_utc") == valid_at_utc)
        .filter(pl.col("load_mw").is_not_null())
        .collect()
    )
    if df.height != 1:
        return None
    return df.row(0, named=True)


def _verification_row(
    *,
    record: ledger.ForecastRecord,
    point: ledger.ForecastPointRecord,
    outcome: dict[str, Any],
    verified_at_utc: datetime,
    outcome_cutoff_utc: datetime,
    policy_version: str,
) -> dict[str, Any]:
    actual = float(outcome["load_mw"])
    point_estimate = (
        point.p50_mw
        if record.point_estimate_kind == ledger.PointEstimateKind.MEDIAN
        else point.mean_mw
    )
    error = point_estimate - actual
    inside = point.p10_mw <= actual <= point.p90_mw
    interval_score = interval_score_80(actual, point.p10_mw, point.p90_mw)
    return {
        "schema_version": ledger.SCHEMA_VERSION,
        "verification_id": _verification_id(
            record.issuance_id, point.valid_at_utc, policy_version
        ),
        "issuance_id": record.issuance_id,
        "ba": record.ba,
        "valid_at_utc": point.valid_at_utc,
        "verified_at_utc": verified_at_utc,
        "outcome_cutoff_utc": outcome_cutoff_utc,
        "outcome_policy_version": policy_version,
        "outcome_observation_id": _observation_id(outcome),
        "outcome_as_of_utc": outcome["as_of"],
        "actual_mw": actual,
        "point_estimate_mw": point_estimate,
        "point_estimate_kind": record.point_estimate_kind.value,
        "error_mw": error,
        "abs_error_mw": abs(error),
        "squared_error_mw2": error * error,
        "pinball_p10": pinball_loss(actual, point.p10_mw, 0.1),
        "pinball_p50": pinball_loss(actual, point.p50_mw, 0.5),
        "pinball_p90": pinball_loss(actual, point.p90_mw, 0.9),
        "inside_p10_p90": inside,
        "interval_score_80": interval_score,
        "wis": weighted_interval_score(
            actual, point.p10_mw, point.p50_mw, point.p90_mw
        ),
        "metric_version": METRIC_VERSION,
    }


def verify_forecast(
    issuance_id: str,
    *,
    verified_at_utc: datetime | None = None,
    maturity_hours: int = MATURITY_HOURS,
    policy_version: str | None = None,
    require_complete: bool = True,
) -> int:
    """Append pinned outcome rows for every mature point in an issuance."""
    now = (verified_at_utc or datetime.now(tz=UTC)).astimezone(UTC)
    derived_policy_version = outcome_policy_version(maturity_hours)
    if policy_version is not None and policy_version != derived_policy_version:
        raise ValueError(
            "policy_version does not match maturity_hours: "
            f"expected {derived_policy_version}"
        )
    policy_version = derived_policy_version
    record = ledger.get_forecast(issuance_id)
    final_maturity = record.last_valid_at_utc + timedelta(hours=maturity_hours)
    if require_complete and now < final_maturity:
        raise ValueError(
            f"issuance is not fully mature until {final_maturity.isoformat()}"
        )

    rows: list[dict[str, Any]] = []
    for point in record.points:
        outcome_cutoff = point.valid_at_utc + timedelta(hours=maturity_hours)
        if now < outcome_cutoff:
            continue
        outcome = _select_outcome(
            ba=record.ba,
            valid_at_utc=point.valid_at_utc,
            outcome_cutoff_utc=outcome_cutoff,
        )
        if outcome is None:
            if require_complete:
                raise RuntimeError(
                    f"no eligible {record.ba} outcome for {point.valid_at_utc.isoformat()}"
                )
            continue
        rows.append(
            _verification_row(
                record=record,
                point=point,
                outcome=outcome,
                verified_at_utc=now,
                outcome_cutoff_utc=outcome_cutoff,
                policy_version=policy_version,
            )
        )

    committed = 0
    for row in rows:
        partitions = {
            "ledger_mode": record.mode.value,
            "valid_date": row["valid_at_utc"].date().isoformat(),
        }
        # A later verifier run must preserve the first pinned settlement row,
        # including its original verification timestamp.
        if store.immutable_path(
            "forecast_verifications",
            row["verification_id"],
            partitions=partitions,
        ).exists():
            continue
        frame = schemas.enforce(
            pl.DataFrame([row]),
            schemas.FORECAST_VERIFICATION,
        )
        store.write_immutable(
            "forecast_verifications",
            row["verification_id"],
            frame,
            partitions=partitions,
        )
        committed += 1
    return committed


def verification_rows(
    issuance_id: str,
    *,
    policy_version: str | None = None,
) -> pl.DataFrame:
    lf = store.scan("forecast_verifications")
    if not lf.collect_schema().names():
        return schemas.enforce(
            pl.DataFrame({column: [] for column in schemas.FORECAST_VERIFICATION}),
            schemas.FORECAST_VERIFICATION,
        )
    lf = lf.filter(pl.col("issuance_id") == issuance_id)
    if policy_version is not None:
        lf = lf.filter(pl.col("outcome_policy_version") == policy_version)
    return (
        lf
        .sort("valid_at_utc")
        .collect()
        .select(list(schemas.FORECAST_VERIFICATION))
    )


def score_forecast(
    issuance_id: str,
    *,
    policy_version: str = OUTCOME_POLICY_VERSION,
) -> ScoreSummary:
    record = ledger.get_forecast(issuance_id)
    df = verification_rows(issuance_id, policy_version=policy_version)
    if df.is_empty():
        raise ValueError(
            f"issuance {issuance_id} has no verified points for {policy_version}"
        )

    n_points = df.height
    actual_abs_sum = _as_float(df.get_column("actual_mw").abs().sum())
    mae = _as_float(df.get_column("abs_error_mw").mean())
    rmse = math.sqrt(_as_float(df.get_column("squared_error_mw2").mean()))
    bias = _as_float(df.get_column("error_mw").mean())
    coverage_pct = _as_float(df.get_column("inside_p10_p90").mean()) * 100.0
    wape = (
        _as_float(df.get_column("abs_error_mw").sum()) / actual_abs_sum * 100.0
        if actual_abs_sum > 0
        else math.nan
    )
    scored_from = df.get_column("valid_at_utc").min()
    scored_through = df.get_column("valid_at_utc").max()
    if not isinstance(scored_from, datetime) or not isinstance(scored_through, datetime):
        raise RuntimeError("verification timestamps are malformed")
    return ScoreSummary(
        basis=record.mode.value,
        n_points=n_points,
        mae_mw=mae,
        rmse_mw=rmse,
        bias_mw=bias,
        wape_pct=wape,
        mase_24=mae / record.mase_scale_24,
        wis_mw=_as_float(df.get_column("wis").mean()),
        pi80_coverage_pct=coverage_pct,
        pi80_calibration_error_pct=abs(coverage_pct - 80.0),
        scored_from_utc=scored_from,
        scored_through_utc=scored_through,
    )


def _as_float(value: Any) -> float:
    if value is None:
        raise RuntimeError("required score aggregate is null")
    return float(value)
