"""The GBM baseline's invariants, each of which cost a wrong answer once."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.gbm_baseline import (
    FEATURE_NAMES,
    HORIZON,
    aligned_training_origins,
    build_design,
)


def _series(hours: int = 24 * 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = np.datetime64("2020-01-01T00", "h") + np.arange(hours)
    index = np.arange(hours)
    # Daily and weekly shape plus a rising trend, so a level-target model would
    # have to extrapolate.
    load = (
        1000.0
        + 200 * np.sin(2 * np.pi * (index % 24) / 24)
        + 50 * np.sin(2 * np.pi * (index % 168) / 168)
        + 0.05 * index
    )
    temp = 15.0 + 10 * np.sin(2 * np.pi * (index % 8760) / 8760)
    return load, temp, ts


def test_training_origins_share_the_evaluation_hour_of_day() -> None:
    """Invariant 1. Misalignment was worth 9.1 points of MAPE."""
    evaluation = list(range(5000, 5240, HORIZON))
    training = aligned_training_origins(
        evaluation_origins=evaluation, earliest=2048, before=5000
    )
    assert training
    for origin in training:
        assert (origin - evaluation[0]) % HORIZON == 0


def test_misaligned_origins_are_impossible_to_request() -> None:
    evaluation = [5001]
    training = aligned_training_origins(
        evaluation_origins=evaluation, earliest=2048, before=5000
    )
    assert all(origin % HORIZON == 5001 % HORIZON for origin in training)


def test_no_training_origins_is_an_error_not_an_empty_fit() -> None:
    with pytest.raises(ValueError, match="no training origins"):
        aligned_training_origins(
            evaluation_origins=[3000], earliest=2048, before=2060
        )


def test_the_target_is_a_ratio_to_same_hour_yesterday() -> None:
    """Invariant 2. A level target cannot extrapolate a growing series."""
    load, temp, ts = _series()
    design = build_design(load, temp, ts, [3000, 3024])
    assert design.target_ratio.size
    # A trending, strongly diurnal series still has ratios near one.
    assert 0.5 < float(np.median(design.target_ratio)) < 1.5
    # And the level round-trips.
    expected = np.asarray([load[3000 + step - 1] for step in range(1, HORIZON + 1)])
    assert np.allclose(design.level[:HORIZON], expected)


def test_a_nonpositive_anchor_drops_the_row() -> None:
    """Invariant 3. Every feature is scaled by the anchor."""
    load, temp, ts = _series()
    poisoned = load.copy()
    poisoned[3000 - 24] = 0.0
    design = build_design(poisoned, temp, ts, [3000])
    assert design.features.shape[0] == HORIZON - 1


def test_features_match_their_declared_names() -> None:
    load, temp, ts = _series()
    design = build_design(load, temp, ts, [3000])
    assert design.features.shape[1] == len(FEATURE_NAMES)


def test_no_feature_reads_beyond_the_cutoff() -> None:
    """The information set must match the served forecast exactly.

    Corrupting everything at or after the origin must not change any feature;
    only the labels may move.
    """
    load, temp, ts = _series()
    origin = 3000
    clean = build_design(load, temp, ts, [origin])

    tampered_load = load.copy()
    tampered_temp = temp.copy()
    tampered_load[origin:] += 5000.0
    tampered_temp[origin:] += 50.0
    tampered = build_design(tampered_load, tampered_temp, ts, [origin])

    assert np.allclose(clean.features, tampered.features, equal_nan=True)
    assert not np.allclose(clean.level, tampered.level)
