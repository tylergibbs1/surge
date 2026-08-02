"""Frozen gradient-boosting baseline for day-ahead load.

A public scoreboard needs a strong open baseline that is not a foundation
model, or a reader cannot tell whether the foundation model is doing anything.
This is that baseline. It is also, empirically, half of Surge's best forecast:
blended with Chronos-2 it beats either model alone on every RTO.

Information set is identical to the served forecast. For origin ``o`` the model
may read load and weather up to ``o-1`` and nothing after. Horizon step ``h`` in
1..24 targets hour ``o+h-1``. No future weather of any kind.

Three invariants are enforced rather than documented, because each cost a
materially wrong answer while this was being built:

1. **Origins must share an hour of day across train and evaluation.** Striding
   from a fixed array index put training cutoffs at 08:00 UTC and evaluation
   cutoffs at 00:00 UTC. The model learned one diurnal position and was scored
   at another: a flat +19% bias on the predicted ratio, and a 12.3% MAPE that
   looked like "gradient boosting does not work here". It was worth 9.1 points.
2. **The target is a ratio, not a level.** Trees cannot extrapolate. US load has
   grown since 2019, so a level-target model predicts the training range and
   fails hardest exactly where growth is fastest.
3. **The anchor must be positive and finite**, since every feature is scaled by
   it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HORIZON = 24
BASELINE_VERSION = "surge-gbm-baseline-v1"

FEATURE_NAMES = (
    "horizon_step",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "load_t_minus_48_over_anchor",
    "load_t_minus_168_over_anchor",
    "load_t_minus_169_over_anchor",
    "last_over_anchor",
    "mean24_over_anchor",
    "mean168_over_anchor",
    "std24_over_mean24",
    "ramp_over_mean24",
    "last_minus_mean24_over_mean24",
    "temp_at_cutoff",
    "temp_mean24_at_cutoff",
    "temp_t_minus_24",
    "temp_t_minus_168",
    "temp_anomaly_at_cutoff",
)

#: Deliberately untuned. The point of a baseline is that it is reproducible and
#: hard to accuse of being over-fitted, not that it is maximal.
DEFAULT_PARAMS = {
    "objective": "l1",
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "verbose": -1,
}


@dataclass(frozen=True)
class DesignMatrix:
    features: np.ndarray
    target_ratio: np.ndarray
    anchor: np.ndarray

    @property
    def level(self) -> np.ndarray:
        """The actual load these rows describe."""
        return self.target_ratio * self.anchor


def aligned_training_origins(
    *, evaluation_origins: list[int], earliest: int, before: int, horizon: int = HORIZON
) -> list[int]:
    """Training origins on the same hour of day as the evaluation origins.

    Invariant 1. Any stride that does not satisfy this trains and scores the
    model at different diurnal positions.
    """
    if not evaluation_origins:
        raise ValueError("at least one evaluation origin is required")
    anchor_hour = evaluation_origins[0] % horizon
    start = earliest + ((anchor_hour - earliest) % horizon)
    origins = list(range(start, before - horizon, horizon))
    if not origins:
        raise ValueError("no training origins available before the evaluation split")
    if (origins[0] - evaluation_origins[0]) % horizon != 0:
        raise RuntimeError("training and evaluation origins are not hour-aligned")
    return origins


def build_design(
    target: np.ndarray,
    temp: np.ndarray,
    ts_utc: np.ndarray,
    origins: list[int],
    *,
    horizon: int = HORIZON,
) -> DesignMatrix:
    """Feature rows for every (origin, horizon step), cutoff at ``origin - 1``."""
    hours = ts_utc.astype("datetime64[h]").astype(np.int64)
    hour_of_day = hours % 24
    days = ts_utc.astype("datetime64[D]").astype(np.int64)
    dow = (days + 3) % 7
    month = ts_utc.astype("datetime64[M]").astype(np.int64) % 12 + 1

    features: list[list[float]] = []
    ratios: list[float] = []
    anchors: list[float] = []
    for origin in origins:
        cutoff = origin - 1
        if cutoff - 169 < 0:
            continue
        recent = target[cutoff - 23 : cutoff + 1]
        week = target[cutoff - 167 : cutoff + 1]
        last = target[cutoff]
        mean24 = np.nanmean(recent)
        if not np.isfinite(last) or not np.isfinite(mean24) or mean24 <= 0:
            continue
        mean168 = np.nanmean(week)
        std24 = np.nanstd(recent)
        ramp = last - target[cutoff - 1]
        temp_now = temp[cutoff]
        temp_mean24 = np.nanmean(temp[cutoff - 23 : cutoff + 1])
        for step in range(1, horizon + 1):
            index = origin + step - 1
            if index >= len(target) or index - 169 < 0:
                continue
            anchor = target[index - 24]  # invariant 3
            if not np.isfinite(anchor) or anchor <= 0:
                continue
            features.append(
                [
                    step,
                    hour_of_day[index],
                    dow[index],
                    month[index],
                    1.0 if dow[index] >= 5 else 0.0,
                    np.sin(2 * np.pi * hour_of_day[index] / 24),
                    np.cos(2 * np.pi * hour_of_day[index] / 24),
                    target[index - 48] / anchor,
                    target[index - 168] / anchor,
                    target[index - 169] / anchor,
                    last / anchor,
                    mean24 / anchor,
                    mean168 / anchor,
                    std24 / mean24,
                    ramp / mean24,
                    (last - mean24) / mean24,
                    temp_now,
                    temp_mean24,
                    temp[index - 24],
                    temp[index - 168],
                    temp_now - temp_mean24,
                ]
            )
            ratios.append(target[index] / anchor)  # invariant 2
            anchors.append(anchor)

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.size and matrix.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("feature matrix and FEATURE_NAMES disagree")
    return DesignMatrix(
        features=matrix,
        target_ratio=np.asarray(ratios, dtype=np.float64),
        anchor=np.asarray(anchors, dtype=np.float64),
    )


def fit(design: DesignMatrix, *, params: dict | None = None):
    """Train on rows with a finite label.

    Feature NaNs are kept: LightGBM handles them natively, and dropping a whole
    row for one missing temperature reading discards usable load history.
    """
    import lightgbm as lgb

    usable = np.isfinite(design.target_ratio)
    if not usable.any():
        raise ValueError("no finite training labels")
    model = lgb.LGBMRegressor(**(params or DEFAULT_PARAMS))
    model.fit(design.features[usable], design.target_ratio[usable])
    return model


def predict_levels(model, design: DesignMatrix) -> np.ndarray:
    """Predicted load levels for the design's rows."""
    if not len(design.anchor):
        return np.asarray([], dtype=np.float64)
    return np.asarray(model.predict(design.features), dtype=np.float64) * design.anchor
