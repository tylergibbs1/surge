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


def local_calendar_covariates(
    ts_utc: np.ndarray, *, timezone: str
) -> dict[str, np.ndarray]:
    """Calendar features on the BA's local wall clock, including DST.

    ``calendar_covariates`` derives hour, weekday, weekend and holiday from the
    UTC stamp. Cyclic hour and weekday survive that: for a single series the
    offset is constant and learnable. The binary flags do not. A US holiday and
    a weekend are local-calendar spans, so a UTC flag is misaligned by the
    offset at both ends -- and the misalignment lands on the evening peak, the
    hours that matter most. Measured over 2024, the UTC weekend flag is wrong
    for 772 CISO hours, 525 of them between 17:00 and 21:00 local.
    """
    from zoneinfo import ZoneInfo

    zone = ZoneInfo(timezone)
    ts = np.asarray(ts_utc).astype("datetime64[h]")
    if ts.ndim != 1:
        raise ValueError("ts_utc must be one-dimensional")
    if np.isnat(ts).any():
        raise ValueError("ts_utc cannot contain NaT")

    local = [
        datetime.fromisoformat(str(stamp)).replace(tzinfo=UTC).astimezone(zone)
        for stamp in ts
    ]
    hour = np.array([value.hour for value in local], dtype=np.int64)
    dow = np.array([value.weekday() for value in local], dtype=np.float32)
    weekend = (dow >= 5).astype(np.float32)
    holiday = np.array(
        [1.0 if value.date() in US_HOLIDAYS else 0.0 for value in local],
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
