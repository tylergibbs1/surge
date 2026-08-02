"""Request/response schemas for the forecast API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from surge import bas as _bas

# Everything the model can forecast — i.e. every BA with a demand series.
# Sourced from the central registry so adding a BA in one place propagates
# here automatically.
SUPPORTED_BAS: tuple[str, ...] = tuple(_bas.demand_codes())


class ForecastPoint(BaseModel):
    ts_utc: datetime
    mean_mw: float = Field(..., description="Predictive mean; the public point estimate is p50")
    median_mw: float = Field(..., description="Point forecast (median)")
    p10_mw: float = Field(
        ...,
        description="10th percentile — lower end of the published 80% PI, "
        "after conformal calibration when it applied",
    )
    p90_mw: float = Field(
        ...,
        description="90th percentile — upper end of the published 80% PI, "
        "after conformal calibration when it applied",
    )
    uncalibrated_p10_mw: float | None = Field(
        None,
        description="The model's own p10 before calibration. Published so a "
        "widened or tightened interval can be audited against its source.",
    )
    uncalibrated_p90_mw: float | None = Field(
        None, description="The model's own p90 before calibration."
    )
    temp_c: float | None = Field(
        None,
        description="Forecast-vintage temperature at the BA station (°C), when used. "
        "Null under the calendar-only load-v2-core feature specification.",
    )


class ForecastQuality(BaseModel):
    status: Literal["fresh", "delayed", "stale", "unavailable"]
    reasons: list[str] = Field(default_factory=list)


class ForecastResponse(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    issuance_id: str
    run_id: str
    ba: str
    model: str
    model_revision: str
    model_artifact_sha256: str | None = None
    as_of_utc: datetime = Field(..., description="Timestamp the forecast was produced")
    generated_at_utc: datetime
    issued_at_utc: datetime
    data_cutoff_utc: datetime = Field(
        ..., description="Latest observed target hour included in model context"
    )
    feature_cutoff_utc: datetime = Field(
        ..., description="Maximum feature availability timestamp allowed for this issuance"
    )
    context_start_utc: datetime
    context_end_utc: datetime
    horizon: int
    units: str = "MW"
    feature_spec_version: str
    feature_spec_sha256: str
    feature_snapshot_sha256: str
    availability_mode: str
    point_estimate_kind: Literal["median", "mean"] = "median"
    mase_scale_24: float
    code_revision: str
    committed: bool = Field(
        ...,
        description="True only when this issuance is durably present in the ledger",
    )
    run_published: bool = Field(
        ...,
        description="True only when the issuance belongs to a complete published run",
    )
    published_at_utc: datetime | None = Field(
        None,
        description="Complete-run publication timestamp; null for ephemeral or staged issuances",
    )
    warnings: list[str] = Field(default_factory=list)
    quality: ForecastQuality
    points: list[ForecastPoint]


class BAMeta(BaseModel):
    code: str
    name: str
    interconnect: str
    utc_offset: int
    station: str | None
    has_demand: bool
    is_rto: bool
    centroid: tuple[float, float] = Field(..., description="(longitude, latitude)")
    peak_mw: int | None


class BAListResponse(BaseModel):
    bas: list[str]
    count: int
    # Full registry payload. Clients that just want codes can read `bas`;
    # richer clients (e.g. the map UI) use `metadata` to draw labels,
    # colour-scale by peak demand, and place centroids without a second round trip.
    metadata: list[BAMeta] = Field(default_factory=list)


class HealthResponse(BaseModel):
    checked_at_utc: datetime
    status: Literal["ok", "degraded", "error", "loading"]
    model_loaded: bool
    model_name: str | None
    model_revision: str | None = None
    data_end_utc: datetime | None
    data_age_hours: float | None = None
    source_watermarks_utc: dict[str, datetime | None] = Field(default_factory=dict)
    source_age_hours: dict[str, float | None] = Field(default_factory=dict)
    latest_issuance_utc: datetime | None = None
    reasons: list[str] = Field(default_factory=list)


class LiveResponse(BaseModel):
    status: Literal["ok"] = "ok"
    checked_at_utc: datetime


class LedgerForecastSummary(BaseModel):
    issuance_id: str
    run_id: str
    mode: str
    ba: str
    issued_at_utc: datetime
    data_cutoff_utc: datetime
    first_valid_at_utc: datetime
    last_valid_at_utc: datetime
    model: str
    model_revision: str
    model_artifact_sha256: str | None = None
    feature_spec_version: str
    availability_mode: str
    point_estimate_kind: str
    horizon: int
    warnings: list[str] = Field(default_factory=list)


class LedgerForecastListResponse(BaseModel):
    forecasts: list[LedgerForecastSummary]
    count: int


class LedgerRunResponse(BaseModel):
    schema_version: int
    run_id: str
    mode: str
    scheduled_for_utc: datetime
    published_at_utc: datetime
    required_bas: list[str]
    issuance_ids: dict[str, str]
    feature_snapshot_sha256s: dict[str, str]
    points_sha256s: dict[str, str]
    target_name: str
    units: str
    horizon_hours: int
    frequency_minutes: int
    quantiles: list[float]
    model_name: str
    model_revision: str
    model_artifact_sha256: str | None = None
    code_revision: str
    feature_spec_version: str
    feature_spec_sha256: str
    availability_mode: str
    point_estimate_kind: str
    run_content_sha256: str


class LedgerBakeResponse(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    run: LedgerRunResponse
    forecasts: list[ForecastResponse]
    committed_regions: int
    run_published: Literal[True] = True


class ScoreSummaryResponse(BaseModel):
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


class LedgerScoreboardRow(BaseModel):
    ba: str
    forecast: LedgerForecastSummary | None
    score: ScoreSummaryResponse | None
    state: Literal["fresh", "delayed", "stale", "unavailable"]
    reasons: list[str] = Field(default_factory=list)


class LedgerScoreboardResponse(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    generated_at_utc: datetime
    expected_regions: int
    available_regions: int
    state: Literal["fresh", "delayed", "stale", "unavailable"]
    regions: list[LedgerScoreboardRow]


class CurrentLoadPoint(BaseModel):
    ts_utc: datetime
    total_mw: float = Field(
        ..., description="Sum of load_mw across every reporting BA at this hour"
    )
    ba_count: int = Field(..., description="How many BAs contributed (some BAs lag in publishing)")


class CurrentLoadResponse(BaseModel):
    as_of_utc: datetime
    latest_ts_utc: datetime
    latest_total_mw: float
    hours: int
    points: list[CurrentLoadPoint]


class ActualPoint(BaseModel):
    ts_utc: datetime
    load_mw: float


class ActualsResponse(BaseModel):
    ba: str
    as_of_utc: datetime
    hours: int
    units: str = "MW"
    points: list[ActualPoint]
