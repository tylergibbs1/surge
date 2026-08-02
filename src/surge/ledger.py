"""Immutable forecast issuance and atomic seven-RTO publication ledger.

Each BA issuance is durable and directly addressable for audit. Live collection
readers, however, only expose issuances referenced by an immutable complete-run
marker. The marker is written after all seven required RTOs have passed the
cross-issuance provenance and shape checks, so a partial run is never public.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from surge import bas, schemas, store

SCHEMA_VERSION = 2
RUN_SCHEMA_VERSION = 1
LEDGER_NAMESPACE = uuid.UUID("97f8345d-308d-4fc1-969a-e1f8f443b8f4")
REQUIRED_RTO_BAS = tuple(bas.rto_codes())
FORECAST_QUANTILES = (0.1, 0.5, 0.9)


class ForecastMode(StrEnum):
    LIVE = "live"
    BACKTEST = "backtest"
    REPLAY = "replay"
    MIGRATED_UNVERIFIED = "migrated_unverified"


class AvailabilityMode(StrEnum):
    EXACT_VINTAGE = "exact_vintage"
    INFERRED_LAG = "inferred_lag"
    RETROSPECTIVE_FINAL = "retrospective_final"


class PointEstimateKind(StrEnum):
    MEDIAN = "median"
    MEAN = "mean"


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("ledger timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_run_id(mode: ForecastMode, scheduled_for_utc: datetime) -> str:
    scheduled = _as_utc(scheduled_for_utc).isoformat()
    return str(uuid.uuid5(LEDGER_NAMESPACE, f"run|{mode.value}|{scheduled}"))


def deterministic_issuance_id(
    *,
    mode: ForecastMode,
    scheduled_for_utc: datetime,
    ba: str,
    target_name: str,
    model_revision: str,
    feature_spec_version: str,
) -> str:
    identity = "|".join(
        (
            "issuance",
            mode.value,
            _as_utc(scheduled_for_utc).isoformat(),
            ba.upper(),
            target_name,
            model_revision,
            feature_spec_version,
        )
    )
    return str(uuid.uuid5(LEDGER_NAMESPACE, identity))


class ForecastRunRecord(BaseModel):
    """Immutable marker proving that one live seven-RTO run is complete."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)

    schema_version: int = RUN_SCHEMA_VERSION
    run_id: str
    mode: ForecastMode
    scheduled_for_utc: datetime
    published_at_utc: datetime
    required_bas: tuple[str, ...]
    issuance_ids: dict[str, str]
    feature_snapshot_sha256s: dict[str, str]
    points_sha256s: dict[str, str]
    target_name: str
    units: str
    horizon_hours: int = Field(gt=0)
    frequency_minutes: int = Field(gt=0)
    quantiles: tuple[float, ...]
    model_name: str
    model_revision: str
    model_artifact_sha256: str | None = None
    code_revision: str
    feature_spec_version: str
    feature_spec_sha256: str
    availability_mode: AvailabilityMode
    point_estimate_kind: PointEstimateKind
    run_content_sha256: str = ""

    @field_validator("scheduled_for_utc", "published_at_utc")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_run(self) -> ForecastRunRecord:
        if self.schema_version != RUN_SCHEMA_VERSION:
            raise ValueError(f"run schema_version must be {RUN_SCHEMA_VERSION}")
        if self.mode != ForecastMode.LIVE:
            raise ValueError("complete-run publication is only defined for live forecasts")
        if self.required_bas != REQUIRED_RTO_BAS:
            raise ValueError("complete-run marker must contain the canonical seven RTOs")
        required = set(REQUIRED_RTO_BAS)
        if set(self.issuance_ids) != required:
            raise ValueError("complete-run marker has an invalid issuance set")
        if len(set(self.issuance_ids.values())) != len(REQUIRED_RTO_BAS):
            raise ValueError("complete-run marker contains duplicate issuance IDs")
        if set(self.feature_snapshot_sha256s) != required:
            raise ValueError("complete-run marker has an invalid feature snapshot set")
        if set(self.points_sha256s) != required:
            raise ValueError("complete-run marker has an invalid point digest set")
        if self.quantiles != FORECAST_QUANTILES:
            raise ValueError("complete-run marker has an unexpected quantile contract")
        expected_run_id = deterministic_run_id(self.mode, self.scheduled_for_utc)
        if self.run_id != expected_run_id:
            raise ValueError("complete-run marker has an invalid deterministic run ID")

        content = self.model_dump(mode="json", exclude={"run_content_sha256"})
        expected_sha = _canonical_sha256(content)
        if self.run_content_sha256 and self.run_content_sha256 != expected_sha:
            raise ValueError("run_content_sha256 does not match complete-run marker")
        object.__setattr__(self, "run_content_sha256", expected_sha)
        return self


class ForecastPointRecord(BaseModel):
    """One immutable probabilistic forecast point."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    valid_at_utc: datetime
    mean_mw: float
    p10_mw: float
    p50_mw: float
    p90_mw: float
    future_temp_c: float | None = None
    future_temp_vintage_id: str | None = None

    @field_validator("valid_at_utc")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_values(self) -> ForecastPointRecord:
        values = (self.mean_mw, self.p10_mw, self.p50_mw, self.p90_mw)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("forecast values must be finite")
        if any(v < 0 for v in values):
            raise ValueError("load forecasts must be nonnegative")
        if not self.p10_mw <= self.p50_mw <= self.p90_mw:
            raise ValueError("forecast quantiles must satisfy p10 <= p50 <= p90")
        if self.future_temp_c is not None and not math.isfinite(self.future_temp_c):
            raise ValueError("future_temp_c must be finite when present")
        if self.future_temp_c is not None and not self.future_temp_vintage_id:
            raise ValueError("future temperature requires a forecast vintage ID")
        return self


class ForecastRecord(BaseModel):
    """A complete BA forecast ready to be durably staged in the ledger."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)

    schema_version: int = SCHEMA_VERSION
    issuance_id: str = ""
    run_id: str = ""
    mode: ForecastMode = ForecastMode.LIVE
    ba: str
    target_name: str = "load_mw"
    units: str = "MW"
    scheduled_for_utc: datetime
    feature_cutoff_utc: datetime
    issued_at_utc: datetime
    context_start_utc: datetime
    context_end_utc: datetime
    frequency_minutes: int = 60
    model_name: str
    model_revision: str
    model_artifact_sha256: str | None = None
    code_revision: str
    feature_spec_version: str
    feature_spec_sha256: str
    feature_snapshot_sha256: str
    availability_mode: AvailabilityMode
    point_estimate_kind: PointEstimateKind = PointEstimateKind.MEDIAN
    mase_scale_24: float = Field(gt=0)
    warnings: tuple[str, ...] = ()
    points_sha256: str = ""
    points: tuple[ForecastPointRecord, ...]

    @field_validator(
        "scheduled_for_utc",
        "feature_cutoff_utc",
        "issued_at_utc",
        "context_start_utc",
        "context_end_utc",
    )
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @field_validator("ba")
    @classmethod
    def normalize_ba(cls, value: str) -> str:
        value = value.upper().strip()
        if not value or len(value) > 8:
            raise ValueError("invalid BA code")
        return value

    @model_validator(mode="after")
    def validate_record(self) -> ForecastRecord:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if not self.points:
            raise ValueError("a forecast issuance must contain at least one point")
        if self.frequency_minutes != 60:
            raise ValueError("v0.2 ledger forecasts must use an hourly frequency")
        if self.context_start_utc > self.context_end_utc:
            raise ValueError("context_start_utc must not follow context_end_utc")
        if self.feature_cutoff_utc > self.issued_at_utc:
            raise ValueError("feature cutoff cannot be after issuance")
        if not math.isfinite(self.mase_scale_24):
            raise ValueError("mase_scale_24 must be finite")

        expected_delta = timedelta(minutes=self.frequency_minutes)
        for previous, current in zip(self.points, self.points[1:], strict=False):
            if current.valid_at_utc - previous.valid_at_utc != expected_delta:
                raise ValueError("forecast valid times must be contiguous")
        first_valid = self.points[0].valid_at_utc
        if self.context_end_utc >= first_valid:
            raise ValueError("forecast context must end before the first valid time")
        if self.mode == ForecastMode.LIVE and self.issued_at_utc >= first_valid:
            raise ValueError("live forecasts must be committed before their first valid time")

        points_payload = [p.model_dump(mode="json") for p in self.points]
        points_sha = _canonical_sha256(points_payload)
        if self.points_sha256 and self.points_sha256 != points_sha:
            raise ValueError("points_sha256 does not match forecast points")
        object.__setattr__(self, "points_sha256", points_sha)

        expected_run_id = deterministic_run_id(self.mode, self.scheduled_for_utc)
        if self.run_id and self.run_id != expected_run_id:
            raise ValueError("run_id does not match its deterministic identity")
        object.__setattr__(self, "run_id", expected_run_id)

        expected_issuance_id = deterministic_issuance_id(
            mode=self.mode,
            scheduled_for_utc=self.scheduled_for_utc,
            ba=self.ba,
            target_name=self.target_name,
            model_revision=self.model_revision,
            feature_spec_version=self.feature_spec_version,
        )
        if self.issuance_id and self.issuance_id != expected_issuance_id:
            raise ValueError("issuance_id does not match its deterministic identity")
        object.__setattr__(self, "issuance_id", expected_issuance_id)
        return self

    @property
    def first_valid_at_utc(self) -> datetime:
        return self.points[0].valid_at_utc

    @property
    def last_valid_at_utc(self) -> datetime:
        return self.points[-1].valid_at_utc

    @property
    def horizon_hours(self) -> int:
        return len(self.points)


def _points_frame(record: ForecastRecord) -> pl.DataFrame:
    rows = []
    for index, point in enumerate(record.points, start=1):
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "issuance_id": record.issuance_id,
                "valid_at_utc": point.valid_at_utc,
                "horizon_step": index,
                "lead_minutes_from_issue": int(
                    (point.valid_at_utc - record.issued_at_utc).total_seconds() // 60
                ),
                "mean_mw": point.mean_mw,
                "p10_mw": point.p10_mw,
                "p50_mw": point.p50_mw,
                "p90_mw": point.p90_mw,
                "future_temp_c": point.future_temp_c,
                "future_temp_vintage_id": point.future_temp_vintage_id or "",
            }
        )
    return schemas.enforce(pl.DataFrame(rows), schemas.FORECAST_POINT)


def _issuance_frame(record: ForecastRecord) -> pl.DataFrame:
    row = {
        "schema_version": SCHEMA_VERSION,
        "issuance_id": record.issuance_id,
        "run_id": record.run_id,
        "mode": record.mode.value,
        "ba": record.ba,
        "target_name": record.target_name,
        "units": record.units,
        "scheduled_for_utc": record.scheduled_for_utc,
        "feature_cutoff_utc": record.feature_cutoff_utc,
        "issued_at_utc": record.issued_at_utc,
        "first_valid_at_utc": record.first_valid_at_utc,
        "last_valid_at_utc": record.last_valid_at_utc,
        "context_start_utc": record.context_start_utc,
        "context_end_utc": record.context_end_utc,
        "horizon_hours": record.horizon_hours,
        "frequency_minutes": record.frequency_minutes,
        "model_name": record.model_name,
        "model_revision": record.model_revision,
        "model_artifact_sha256": record.model_artifact_sha256 or "",
        "code_revision": record.code_revision,
        "feature_spec_version": record.feature_spec_version,
        "feature_spec_sha256": record.feature_spec_sha256,
        "feature_snapshot_sha256": record.feature_snapshot_sha256,
        "availability_mode": record.availability_mode.value,
        "point_estimate_kind": record.point_estimate_kind.value,
        "mase_scale_24": record.mase_scale_24,
        "points_sha256": record.points_sha256,
        "point_count": record.horizon_hours,
        "warnings_json": json.dumps(list(record.warnings), separators=(",", ":")),
    }
    return schemas.enforce(pl.DataFrame([row]), schemas.FORECAST_ISSUANCE)


def _run_frame(record: ForecastRunRecord) -> pl.DataFrame:
    row = {
        "schema_version": record.schema_version,
        "run_id": record.run_id,
        "mode": record.mode.value,
        "scheduled_for_utc": record.scheduled_for_utc,
        "published_at_utc": record.published_at_utc,
        "required_bas_json": json.dumps(list(record.required_bas), separators=(",", ":")),
        "issuance_ids_json": json.dumps(
            record.issuance_ids, sort_keys=True, separators=(",", ":")
        ),
        "feature_snapshots_json": json.dumps(
            record.feature_snapshot_sha256s, sort_keys=True, separators=(",", ":")
        ),
        "points_sha256s_json": json.dumps(
            record.points_sha256s, sort_keys=True, separators=(",", ":")
        ),
        "target_name": record.target_name,
        "units": record.units,
        "horizon_hours": record.horizon_hours,
        "frequency_minutes": record.frequency_minutes,
        "quantiles_json": json.dumps(list(record.quantiles), separators=(",", ":")),
        "model_name": record.model_name,
        "model_revision": record.model_revision,
        "model_artifact_sha256": record.model_artifact_sha256 or "",
        "code_revision": record.code_revision,
        "feature_spec_version": record.feature_spec_version,
        "feature_spec_sha256": record.feature_spec_sha256,
        "availability_mode": record.availability_mode.value,
        "point_estimate_kind": record.point_estimate_kind.value,
        "run_content_sha256": record.run_content_sha256,
    }
    return schemas.enforce(pl.DataFrame([row]), schemas.FORECAST_RUN)


def _run_from_header(header: dict[str, Any]) -> ForecastRunRecord:
    return ForecastRunRecord(
        schema_version=header["schema_version"],
        run_id=header["run_id"],
        mode=ForecastMode(header["mode"]),
        scheduled_for_utc=header["scheduled_for_utc"],
        published_at_utc=header["published_at_utc"],
        required_bas=tuple(json.loads(header["required_bas_json"])),
        issuance_ids=json.loads(header["issuance_ids_json"]),
        feature_snapshot_sha256s=json.loads(header["feature_snapshots_json"]),
        points_sha256s=json.loads(header["points_sha256s_json"]),
        target_name=header["target_name"],
        units=header["units"],
        horizon_hours=header["horizon_hours"],
        frequency_minutes=header["frequency_minutes"],
        quantiles=tuple(json.loads(header["quantiles_json"])),
        model_name=header["model_name"],
        model_revision=header["model_revision"],
        model_artifact_sha256=header["model_artifact_sha256"] or None,
        code_revision=header["code_revision"],
        feature_spec_version=header["feature_spec_version"],
        feature_spec_sha256=header["feature_spec_sha256"],
        availability_mode=AvailabilityMode(header["availability_mode"]),
        point_estimate_kind=PointEstimateKind(header["point_estimate_kind"]),
        run_content_sha256=header["run_content_sha256"],
    )


def get_run(run_id: str) -> ForecastRunRecord:
    """Return a complete-run marker, or raise when the run is not public."""
    lf = store.scan("forecast_runs")
    if not lf.collect_schema().names():
        raise KeyError(run_id)
    df = lf.filter(pl.col("run_id") == run_id).collect()
    if df.height != 1:
        raise KeyError(run_id)
    return _run_from_header(df.row(0, named=True))


def _raw_headers_for_run(run_id: str) -> list[dict[str, Any]]:
    lf = store.scan("forecast_issuances")
    if not lf.collect_schema().names():
        return []
    return (
        lf.filter(pl.col("run_id") == run_id)
        .sort("ba")
        .collect()
        .to_dicts()
    )


def _required_records_for_run(run_id: str) -> list[ForecastRecord]:
    records: list[ForecastRecord] = []
    seen_bas: set[str] = set()
    for header in _raw_headers_for_run(run_id):
        ba = str(header["ba"])
        if ba not in REQUIRED_RTO_BAS:
            continue
        if ba in seen_bas:
            raise store.ImmutableConflictError(
                f"run {run_id} contains multiple immutable issuances for {ba}"
            )
        seen_bas.add(ba)
        try:
            records.append(get_forecast(str(header["issuance_id"])))
        except Exception as exc:
            raise store.ImmutableConflictError(
                f"run {run_id} contains an invalid issuance for {ba}"
            ) from exc
    return records


_UNIFORM_RUN_FIELDS = (
    "mode",
    "scheduled_for_utc",
    "target_name",
    "units",
    "horizon_hours",
    "frequency_minutes",
    "first_valid_at_utc",
    "last_valid_at_utc",
    "model_name",
    "model_revision",
    "model_artifact_sha256",
    "code_revision",
    "feature_spec_version",
    "feature_spec_sha256",
    "availability_mode",
    "point_estimate_kind",
)


def _assert_compatible_run_records(records: list[ForecastRecord]) -> None:
    if not records:
        return
    baseline = records[0]
    if baseline.mode != ForecastMode.LIVE:
        raise store.ImmutableConflictError("complete-run publication requires live issuances")
    for record in records:
        if record.ba not in REQUIRED_RTO_BAS:
            raise store.ImmutableConflictError(f"unexpected BA {record.ba} in RTO run")
        if record.run_id != baseline.run_id:
            raise store.ImmutableConflictError("issuances do not share one deterministic run ID")
        for field_name in _UNIFORM_RUN_FIELDS:
            if getattr(record, field_name) != getattr(baseline, field_name):
                raise store.ImmutableConflictError(
                    f"run {baseline.run_id} has mixed {field_name} provenance"
                )


def _build_run_record(records: list[ForecastRecord]) -> ForecastRunRecord:
    _assert_compatible_run_records(records)
    by_ba = {record.ba: record for record in records}
    if set(by_ba) != set(REQUIRED_RTO_BAS) or len(records) != len(REQUIRED_RTO_BAS):
        raise ValueError("a complete run requires exactly the canonical seven RTO issuances")
    ordered = [by_ba[ba] for ba in REQUIRED_RTO_BAS]
    baseline = ordered[0]
    return ForecastRunRecord(
        run_id=baseline.run_id,
        mode=baseline.mode,
        scheduled_for_utc=baseline.scheduled_for_utc,
        published_at_utc=max(record.issued_at_utc for record in ordered),
        required_bas=REQUIRED_RTO_BAS,
        issuance_ids={record.ba: record.issuance_id for record in ordered},
        feature_snapshot_sha256s={
            record.ba: record.feature_snapshot_sha256 for record in ordered
        },
        points_sha256s={record.ba: record.points_sha256 for record in ordered},
        target_name=baseline.target_name,
        units=baseline.units,
        horizon_hours=baseline.horizon_hours,
        frequency_minutes=baseline.frequency_minutes,
        quantiles=FORECAST_QUANTILES,
        model_name=baseline.model_name,
        model_revision=baseline.model_revision,
        model_artifact_sha256=baseline.model_artifact_sha256,
        code_revision=baseline.code_revision,
        feature_spec_version=baseline.feature_spec_version,
        feature_spec_sha256=baseline.feature_spec_sha256,
        availability_mode=baseline.availability_mode,
        point_estimate_kind=baseline.point_estimate_kind,
    )


def _run_lock_path(run_id: str) -> Path:
    try:
        normalized = str(uuid.UUID(run_id))
    except ValueError as exc:
        raise ValueError("run_id must be a UUID") from exc
    if run_id != normalized:
        raise ValueError("run_id must use canonical UUID form")
    root = store.dataset_path("_ledger_locks") / "forecast_runs"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{normalized}.lock"


@contextmanager
def _run_commit_lock(run_id: str) -> Iterator[None]:
    """Serialize staging/finalization so concurrent seventh writes cannot miss."""
    with _run_lock_path(run_id).open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_forecast_files(record: ForecastRecord) -> None:
    partitions = {
        "ledger_mode": record.mode.value,
        "issued_date": record.issued_at_utc.date().isoformat(),
    }
    store.write_immutable(
        "forecast_points",
        record.issuance_id,
        _points_frame(record),
        partitions=partitions,
    )
    store.write_immutable(
        "forecast_issuances",
        record.issuance_id,
        _issuance_frame(record),
        partitions=partitions,
    )


def _publish_run_if_complete_locked(run_id: str) -> ForecastRunRecord | None:
    records = _required_records_for_run(run_id)
    _assert_compatible_run_records(records)
    if {record.ba for record in records} != set(REQUIRED_RTO_BAS):
        return None
    marker = _build_run_record(records)
    partitions = {
        "ledger_mode": marker.mode.value,
        "scheduled_date": marker.scheduled_for_utc.date().isoformat(),
    }
    store.write_immutable(
        "forecast_runs",
        marker.run_id,
        _run_frame(marker),
        partitions=partitions,
    )
    return marker


def publish_run_if_complete(run_id: str) -> ForecastRunRecord | None:
    """Atomically publish ``run_id`` when all seven compatible issuances exist.

    Returning ``None`` means the run is still incomplete. Exact calls after a
    marker exists are idempotent; any content disagreement is a hard immutable
    conflict.
    """
    with _run_commit_lock(run_id):
        return _publish_run_if_complete_locked(run_id)


def commit_forecast(record: ForecastRecord) -> ForecastRecord:
    """Durably stage one issuance and publish its live RTO run when complete."""
    if record.mode != ForecastMode.LIVE or record.ba not in REQUIRED_RTO_BAS:
        _write_forecast_files(record)
        return record

    with _run_commit_lock(record.run_id):
        existing = _required_records_for_run(record.run_id)
        same_ba = [item for item in existing if item.ba == record.ba]
        if same_ba and same_ba[0] != record:
            raise store.ImmutableConflictError(
                f"run {record.run_id} already has a different issuance for {record.ba}"
            )
        candidates = existing if same_ba else [*existing, record]
        _assert_compatible_run_records(candidates)

        try:
            published = get_run(record.run_id)
        except KeyError:
            published = None
        if published is not None:
            expected_id = published.issuance_ids.get(record.ba)
            if expected_id != record.issuance_id:
                raise store.ImmutableConflictError(
                    f"published run {record.run_id} rejects a different {record.ba} issuance"
                )

        _write_forecast_files(record)
        _publish_run_if_complete_locked(record.run_id)
    return record


def _committed_header(issuance_id: str) -> dict[str, Any]:
    lf = store.scan("forecast_issuances")
    if not lf.collect_schema().names():
        raise KeyError(issuance_id)
    df = lf.filter(pl.col("issuance_id") == issuance_id).collect()
    if df.height != 1:
        raise KeyError(issuance_id)
    return df.row(0, named=True)


def points_for(issuance_id: str) -> list[ForecastPointRecord]:
    """Read points only when their issuance commit marker exists."""
    _committed_header(issuance_id)
    lf = store.scan("forecast_points")
    if not lf.collect_schema().names():
        raise KeyError(issuance_id)
    df = (
        lf.filter(pl.col("issuance_id") == issuance_id)
        .sort("horizon_step")
        .collect()
    )
    if df.is_empty():
        raise RuntimeError(f"committed issuance {issuance_id} has no points")
    return [
        ForecastPointRecord(
            valid_at_utc=row["valid_at_utc"],
            mean_mw=row["mean_mw"],
            p10_mw=row["p10_mw"],
            p50_mw=row["p50_mw"],
            p90_mw=row["p90_mw"],
            future_temp_c=row["future_temp_c"],
            future_temp_vintage_id=row["future_temp_vintage_id"] or None,
        )
        for row in df.iter_rows(named=True)
    ]


def get_forecast(issuance_id: str) -> ForecastRecord:
    header = _committed_header(issuance_id)
    points = tuple(points_for(issuance_id))
    return ForecastRecord(
        schema_version=header["schema_version"],
        issuance_id=header["issuance_id"],
        run_id=header["run_id"],
        mode=ForecastMode(header["mode"]),
        ba=header["ba"],
        target_name=header["target_name"],
        units=header["units"],
        scheduled_for_utc=header["scheduled_for_utc"],
        feature_cutoff_utc=header["feature_cutoff_utc"],
        issued_at_utc=header["issued_at_utc"],
        context_start_utc=header["context_start_utc"],
        context_end_utc=header["context_end_utc"],
        frequency_minutes=header["frequency_minutes"],
        model_name=header["model_name"],
        model_revision=header["model_revision"],
        model_artifact_sha256=header["model_artifact_sha256"] or None,
        code_revision=header["code_revision"],
        feature_spec_version=header["feature_spec_version"],
        feature_spec_sha256=header["feature_spec_sha256"],
        feature_snapshot_sha256=header["feature_snapshot_sha256"],
        availability_mode=AvailabilityMode(header["availability_mode"]),
        point_estimate_kind=PointEstimateKind(header["point_estimate_kind"]),
        mase_scale_24=header["mase_scale_24"],
        warnings=tuple(json.loads(header["warnings_json"])),
        points_sha256=header["points_sha256"],
        points=points,
    )


def list_runs(
    *,
    mode: ForecastMode = ForecastMode.LIVE,
    limit: int = 100,
) -> list[ForecastRunRecord]:
    """List immutable complete-run markers, newest scheduled run first."""
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be in 1..1000")
    lf = store.scan("forecast_runs")
    if not lf.collect_schema().names():
        return []
    ids = (
        lf.filter(pl.col("mode") == mode.value)
        .sort(["scheduled_for_utc", "published_at_utc"], descending=True)
        .select("run_id")
        .limit(limit)
        .collect()
        .get_column("run_id")
        .to_list()
    )
    return [get_run(run_id) for run_id in ids]


def latest_complete_run(
    *, mode: ForecastMode = ForecastMode.LIVE
) -> ForecastRunRecord:
    runs = list_runs(mode=mode, limit=1)
    if not runs:
        raise KeyError(mode.value)
    return runs[0]


def forecasts_for_run(run_id: str) -> list[ForecastRecord]:
    """Return all seven records referenced by a complete-run marker."""
    marker = get_run(run_id)
    records = [get_forecast(marker.issuance_ids[ba]) for ba in marker.required_bas]
    rebuilt = _build_run_record(records)
    if rebuilt != marker:
        raise store.ImmutableConflictError(
            f"complete-run marker {run_id} disagrees with its immutable issuances"
        )
    return records


def _published_live_run_ids() -> set[str]:
    lf = store.scan("forecast_runs")
    if not lf.collect_schema().names():
        return set()
    return set(
        lf.filter(pl.col("mode") == ForecastMode.LIVE.value)
        .select("run_id")
        .collect()
        .get_column("run_id")
        .to_list()
    )


def list_forecasts(
    *,
    ba: str | None = None,
    mode: ForecastMode | None = ForecastMode.LIVE,
    limit: int = 100,
    include_unpublished: bool = False,
) -> list[ForecastRecord]:
    """List forecasts, hiding partial live runs unless explicitly requested."""
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be in 1..1000")
    lf = store.scan("forecast_issuances")
    if not lf.collect_schema().names():
        return []
    if ba:
        lf = lf.filter(pl.col("ba") == ba.upper())
    if mode:
        lf = lf.filter(pl.col("mode") == mode.value)
    if not include_unpublished and mode in (None, ForecastMode.LIVE):
        published = list(_published_live_run_ids())
        if mode == ForecastMode.LIVE and not published:
            return []
        lf = lf.filter(
            (pl.col("mode") != ForecastMode.LIVE.value)
            | pl.col("run_id").is_in(published)
        )
    ids = (
        lf.sort("issued_at_utc", descending=True)
        .select("issuance_id")
        .limit(limit)
        .collect()
        .get_column("issuance_id")
        .to_list()
    )
    return [get_forecast(issuance_id) for issuance_id in ids]


def latest_forecast(
    ba: str,
    *,
    mode: ForecastMode = ForecastMode.LIVE,
    include_unpublished: bool = False,
) -> ForecastRecord:
    records = list_forecasts(
        ba=ba,
        mode=mode,
        limit=1,
        include_unpublished=include_unpublished,
    )
    if not records:
        raise KeyError(ba.upper())
    return records[0]


def audit_forecast(issuance_id: str) -> list[str]:
    """Return integrity violations for one committed issuance."""
    errors: list[str] = []
    try:
        header = _committed_header(issuance_id)
        points = points_for(issuance_id)
    except Exception as exc:
        return [str(exc)]
    digest = _canonical_sha256([p.model_dump(mode="json") for p in points])
    if digest != header["points_sha256"]:
        errors.append("points digest mismatch")
    if len(points) != header["point_count"]:
        errors.append("point count mismatch")
    if points[0].valid_at_utc != header["first_valid_at_utc"]:
        errors.append("first valid timestamp mismatch")
    if points[-1].valid_at_utc != header["last_valid_at_utc"]:
        errors.append("last valid timestamp mismatch")
    return errors
