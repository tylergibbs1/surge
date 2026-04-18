"""Request/response schemas for the forecast API."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

SUPPORTED_BAS = ("PJM", "CISO", "ERCO", "MISO", "NYIS", "ISNE", "SWPP")


class ForecastPoint(BaseModel):
    ts_utc: datetime
    median_mw: float = Field(..., description="Point forecast (median)")
    p10_mw: float = Field(..., description="10th percentile — lower end of 80% PI")
    p90_mw: float = Field(..., description="90th percentile — upper end of 80% PI")


class ForecastResponse(BaseModel):
    ba: str
    model: str
    as_of_utc: datetime = Field(..., description="Timestamp the forecast was produced")
    context_start_utc: datetime
    context_end_utc: datetime
    horizon: int
    units: str = "MW"
    points: list[ForecastPoint]


class BAListResponse(BaseModel):
    bas: list[str]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None
    data_end_utc: datetime | None
