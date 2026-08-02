"""Deterministic calendar features for hourly UTC load models."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import holidays
import numpy as np

from surge.features.spec import CALENDAR_COVARIATES

US_HOLIDAYS = holidays.UnitedStates()


def as_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def next_full_utc_hour(value: datetime) -> datetime:
    """Return the first whole UTC hour strictly after ``value``."""
    value = as_utc(value, field="value")
    return value.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)


def datetime_to_numpy(value: datetime) -> np.datetime64:
    value = as_utc(value, field="value")
    return np.datetime64(value.replace(tzinfo=None), "us")


def numpy_to_datetime(value: np.datetime64) -> datetime:
    if np.isnat(value):
        raise ValueError("timestamp cannot be NaT")
    micros = int(value.astype("datetime64[us]").astype(np.int64))
    return datetime.fromtimestamp(micros / 1_000_000, tz=UTC)


def calendar_covariates(ts_utc: np.ndarray) -> dict[str, np.ndarray]:
    """Build the future-safe feature set declared by ``LOAD_V2_CORE``."""
    ts = np.asarray(ts_utc).astype("datetime64[h]")
    if ts.ndim != 1:
        raise ValueError("ts_utc must be one-dimensional")
    if np.isnat(ts).any():
        raise ValueError("ts_utc cannot contain NaT")

    hour = (ts - ts.astype("datetime64[D]")).astype(np.int64) % 24
    days = ts.astype("datetime64[D]")
    days_since_epoch = days.astype(np.int64)
    dow = ((days_since_epoch + 3) % 7).astype(np.float32)
    weekend = (dow >= 5).astype(np.float32)
    holiday = np.array(
        [
            1.0
            if date.fromisoformat(np.datetime_as_string(day, unit="D")) in US_HOLIDAYS
            else 0.0
            for day in days
        ],
        dtype=np.float32,
    )

    two_pi = 2 * np.pi
    result = {
        "hour_sin": np.sin(two_pi * hour / 24).astype(np.float32),
        "hour_cos": np.cos(two_pi * hour / 24).astype(np.float32),
        "dow_sin": np.sin(two_pi * dow / 7).astype(np.float32),
        "dow_cos": np.cos(two_pi * dow / 7).astype(np.float32),
        "is_weekend": weekend,
        "is_holiday": holiday,
    }
    if tuple(result) != CALENDAR_COVARIATES:  # pragma: no cover - guarded by constants
        raise RuntimeError("calendar feature implementation and feature spec diverged")
    return result
