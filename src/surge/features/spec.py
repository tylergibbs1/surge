"""Versioned feature contracts shared by training, evaluation, and serving."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np


class AvailabilityMode(StrEnum):
    """How observation revisions are selected for a feature snapshot."""

    EXACT_VINTAGE = "exact_vintage"
    RETROSPECTIVE_FINAL = "retrospective_final"


class FeatureContractError(ValueError):
    """A model task does not conform to its declared feature contract."""


CALENDAR_COVARIATES = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_holiday",
)
OBSERVED_COVARIATES = ("temp_c", "wind_mw", "solar_mw")


@dataclass(frozen=True)
class FeatureSpec:
    """Names and availability rules for one immutable model feature set."""

    version: str
    target: str
    required_past_covariates: tuple[str, ...]
    optional_past_covariates: tuple[str, ...]
    future_covariates: tuple[str, ...]
    forbidden_future_covariates: tuple[str, ...]
    frequency_minutes: int = 60

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            {
                "version": self.version,
                "target": self.target,
                "required_past_covariates": self.required_past_covariates,
                "optional_past_covariates": self.optional_past_covariates,
                "future_covariates": self.future_covariates,
                "forbidden_future_covariates": self.forbidden_future_covariates,
                "frequency_minutes": self.frequency_minutes,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def allowed_past_covariates(self) -> frozenset[str]:
        return frozenset((*self.required_past_covariates, *self.optional_past_covariates))


# Core v2 deliberately has no observed or assumed-perfect future inputs. A
# weather-forecast-enabled feature set must be introduced as a different spec,
# with explicit forecast vintages, rather than weakening this contract.
LOAD_V2_CORE = FeatureSpec(
    version="load-v2-core",
    target="load_mw",
    required_past_covariates=("temp_c", *CALENDAR_COVARIATES),
    optional_past_covariates=("wind_mw", "solar_mw"),
    future_covariates=CALENDAR_COVARIATES,
    forbidden_future_covariates=OBSERVED_COVARIATES,
)

DEFAULT_QUANTILE_LEVELS = (0.1, 0.5, 0.9)
POINT_ESTIMATE_QUANTILE = 0.5
POINT_ESTIMATE_KIND = "median"
POINT_ESTIMATE_LABEL = "p50"


def _numeric_array(value: Any, *, field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FeatureContractError(f"{field} must be numeric") from exc
    if array.ndim != 1:
        raise FeatureContractError(f"{field} must be one-dimensional")
    if np.isinf(array).any():
        raise FeatureContractError(f"{field} must not contain infinite values")
    return array


def _array_length(value: Any, *, field: str) -> int:
    return int(_numeric_array(value, field=field).shape[0])


def validate_task(
    task: Mapping[str, Any],
    *,
    prediction_length: int,
    spec: FeatureSpec = LOAD_V2_CORE,
    allow_empty_future: bool = False,
) -> None:
    """Validate a Chronos task before it reaches training or inference.

    Training uses empty future arrays as key declarations, which is the only
    exception to the normal ``prediction_length`` requirement.
    """
    if prediction_length < 1:
        raise FeatureContractError("prediction_length must be positive")

    expected_sections = {"target", "past_covariates", "future_covariates"}
    missing_sections = expected_sections - set(task)
    if missing_sections:
        raise FeatureContractError(f"task missing sections: {sorted(missing_sections)}")

    target = _numeric_array(task["target"], field="target")
    target_length = int(target.shape[0])
    if target_length < 1:
        raise FeatureContractError("target must not be empty")
    if not np.isfinite(target).any():
        raise FeatureContractError("target must contain at least one observed value")

    past = task["past_covariates"]
    future = task["future_covariates"]
    if not isinstance(past, Mapping) or not isinstance(future, Mapping):
        raise FeatureContractError("past_covariates and future_covariates must be mappings")

    past_keys = set(past)
    missing_past = set(spec.required_past_covariates) - past_keys
    extra_past = past_keys - spec.allowed_past_covariates
    if missing_past:
        raise FeatureContractError(f"missing required past covariates: {sorted(missing_past)}")
    if extra_past:
        raise FeatureContractError(f"unknown past covariates: {sorted(extra_past)}")

    forbidden = set(future) & set(spec.forbidden_future_covariates)
    if forbidden:
        raise FeatureContractError(
            f"observed covariates cannot be used as future inputs: {sorted(forbidden)}"
        )
    expected_future = set(spec.future_covariates)
    if set(future) != expected_future:
        missing = expected_future - set(future)
        extra = set(future) - expected_future
        raise FeatureContractError(
            f"future covariate keys do not match {spec.version}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    for key, value in past.items():
        field = f"past_covariates[{key!r}]"
        array = _numeric_array(value, field=field)
        length = int(array.shape[0])
        if length != target_length:
            raise FeatureContractError(
                f"past_covariates[{key!r}] has length {length}, expected {target_length}"
            )
    for key, value in future.items():
        field = f"future_covariates[{key!r}]"
        array = _numeric_array(value, field=field)
        length = int(array.shape[0])
        if allow_empty_future and length == 0:
            continue
        if length != prediction_length:
            raise FeatureContractError(
                f"future_covariates[{key!r}] has length {length}, expected {prediction_length}"
            )
        if not np.isfinite(array).all():
            raise FeatureContractError(f"{field} must be fully known and finite")
