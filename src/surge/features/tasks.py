"""Safe Chronos task construction from versioned Surge feature data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from surge.features.spec import LOAD_V2_CORE, FeatureSpec, validate_task

if TYPE_CHECKING:
    from surge.features.data import BAData


def build_evaluation_task(
    data: BAData,
    *,
    origin: int,
    context_length: int,
    prediction_length: int,
    spec: FeatureSpec = LOAD_V2_CORE,
) -> dict[str, Any]:
    """Build one no-peeking rolling-evaluation task.

    Observed arrays after ``origin`` are intentionally unreachable here: only
    the spec's deterministic calendar keys can enter ``future_covariates``.
    """
    if context_length < 1:
        raise ValueError("context_length must be positive")
    if origin - context_length < 0 or origin + prediction_length > len(data.target):
        raise ValueError("evaluation window is outside the available series")

    past = {
        key: values[origin - context_length : origin].astype(np.float32)
        for key, values in data.covariates.items()
        if key in spec.allowed_past_covariates
    }
    future = {
        key: data.covariates[key][origin : origin + prediction_length].astype(np.float32)
        for key in spec.future_covariates
    }
    task = {
        "target": data.target[origin - context_length : origin].astype(np.float32),
        "past_covariates": past,
        "future_covariates": future,
    }
    validate_task(task, prediction_length=prediction_length, spec=spec)
    return task


def build_training_task(
    data: BAData,
    *,
    start: int,
    end: int,
    prediction_length: int,
    spec: FeatureSpec = LOAD_V2_CORE,
) -> dict[str, Any]:
    """Build a Chronos training task with future-safe key declarations."""
    if start < 0 or end > len(data.target) or start >= end:
        raise ValueError("invalid training slice")
    past = {
        key: values[start:end].astype(np.float32)
        for key, values in data.covariates.items()
        if key in spec.allowed_past_covariates
    }
    future = {key: np.array([], dtype=np.float32) for key in spec.future_covariates}
    task = {
        "target": data.target[start:end].astype(np.float32),
        "past_covariates": past,
        "future_covariates": future,
    }
    validate_task(
        task,
        prediction_length=prediction_length,
        spec=spec,
        allow_empty_future=True,
    )
    return task


def build_rolling_validation_tasks(
    data: BAData,
    *,
    context_length: int,
    prediction_length: int,
    step: int,
    max_origins: int,
    spec: FeatureSpec = LOAD_V2_CORE,
) -> list[dict[str, Any]]:
    """Build trailing-label validation tasks wholly inside the validation split.

    Chronos-2 scores the final ``prediction_length`` values of each validation
    input and uses the preceding values only as context. Supplying one task per
    rolling origin makes checkpoint selection reflect the same broad validation
    tail used by Surge's post-fit overfitting audit instead of a single final day.
    """
    for name, value in (
        ("context_length", context_length),
        ("prediction_length", prediction_length),
        ("step", step),
        ("max_origins", max_origins),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    candidate_origins = [
        origin
        for origin in range(
            data.train_end,
            data.val_end - prediction_length + 1,
            step,
        )
        if origin - context_length >= 0
    ]
    # Missing realized load cannot form a valid checkpoint-selection label.
    # Filter solely on target availability, before taking the trailing cohort,
    # so the rule is independent of model errors and agrees with the post-fit
    # diagnostic evaluator's frozen complete-window policy.
    origins = [
        origin
        for origin in candidate_origins
        if np.isfinite(data.target[origin : origin + prediction_length]).all()
    ][-max_origins:]
    if not origins:
        raise ValueError(
            f"{data.ba} has no complete rolling validation target windows"
        )
    return [
        build_training_task(
            data,
            start=origin - context_length,
            end=origin + prediction_length,
            prediction_length=prediction_length,
            spec=spec,
        )
        for origin in origins
    ]
