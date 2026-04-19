"""Request/response schemas for the forecast API."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from surge import bas as _bas

# Everything the model can forecast — i.e. every BA with a demand series.
# Sourced from the central registry so adding a BA in one place propagates
# here automatically.
SUPPORTED_BAS: tuple[str, ...] = tuple(_bas.demand_codes())


class ForecastPoint(BaseModel):
    ts_utc: datetime
    median_mw: float = Field(..., description="Point forecast (median)")
    p10_mw: float = Field(..., description="10th percentile — lower end of 80% PI")
    p90_mw: float = Field(..., description="90th percentile — upper end of 80% PI")
    temp_c: float | None = Field(
        None,
        description="Future-covariate temperature at BA centroid station (°C). "
        "Assumes perfect forecast — see README Limitations.",
    )


class ForecastResponse(BaseModel):
    ba: str
    model: str
    as_of_utc: datetime = Field(..., description="Timestamp the forecast was produced")
    context_start_utc: datetime
    context_end_utc: datetime
    horizon: int
    units: str = "MW"
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
    status: str
    model_loaded: bool
    model_name: str | None
    data_end_utc: datetime | None


class CurrentLoadPoint(BaseModel):
    ts_utc: datetime
    total_mw: float = Field(..., description="Sum of load_mw across every reporting BA at this hour")
    ba_count: int = Field(..., description="How many BAs contributed (some BAs lag in publishing)")


class CurrentLoadResponse(BaseModel):
    as_of_utc: datetime
    latest_ts_utc: datetime
    latest_total_mw: float
    hours: int
    points: list[CurrentLoadPoint]
