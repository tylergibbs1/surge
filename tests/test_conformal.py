from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from experiments.conformal import (
    CalibratedIntervals,
    cqr_scores,
    finite_sample_quantile,
    interval_metrics,
    rolling_conformalize,
)
from experiments.run_conformal_c2 import (
    PINNED_CHRONOS2_REVISION,
    RTO_BAS,
    BAPredictions,
    CandidateRun,
    _align_predictions,
    _candidate_summary,
    _common_eligible,
    _partition_masks,
    _relative_interval_score,
    _validate_args,
)
from surge.verification import interval_score_80, weighted_interval_score


def _daily_origins(count: int) -> np.ndarray:
    return np.datetime64("2024-01-01T00:00:00") + np.arange(count) * np.timedelta64(24, "h")


def _calibrate(
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
    truth: np.ndarray,
) -> CalibratedIntervals:
    return rolling_conformalize(
        lower,
        median,
        upper,
        truth,
        origin_times_utc=_daily_origins(lower.shape[0]),
        outcome_delay_hours=0,
        window=3,
        min_history=1,
    )


def _valid_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "bas": list(RTO_BAS),
        "horizon": 24,
        "context": 2_048,
        "batch_size": 16,
        "min_history": 28,
        "windows": [28, 42],
        "coverage": 0.8,
        "coverage_tolerance": 0.02,
        "selection_fraction": 0.5,
        "code_revision": "a" * 40,
        "data_snapshot_sha256": "b" * 64,
        "model": "amazon/chronos-2",
        "model_revision": PINNED_CHRONOS2_REVISION,
        "model_artifact_sha256": "",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_finite_sample_quantile_uses_conservative_rank() -> None:
    scores = np.arange(1, 10, dtype=np.float64)

    assert finite_sample_quantile(scores, coverage=0.8) == 8.0


def test_pooled_quantile_applies_correction_at_origin_block_level() -> None:
    scores = np.arange(1, 19, dtype=np.float64)

    assert finite_sample_quantile(scores, coverage=0.8, calibration_units=9) == 16.0


def test_rolling_calibration_never_reads_current_or_future_truth() -> None:
    shape = (6, 1, 1)
    lower = np.zeros(shape)
    median = np.ones(shape)
    upper = np.full(shape, 2.0)
    truth = np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0]).reshape(shape)

    first = rolling_conformalize(
        lower,
        median,
        upper,
        truth,
        origin_times_utc=_daily_origins(6),
        outcome_delay_hours=0,
        window=3,
        min_history=3,
    )
    changed_future = truth.copy()
    changed_future[3:] = 1_000.0
    second = rolling_conformalize(
        lower,
        median,
        upper,
        changed_future,
        origin_times_utc=_daily_origins(6),
        outcome_delay_hours=0,
        window=3,
        min_history=3,
    )

    assert first.eligible[3, 0, 0]
    assert first.lower[3, 0, 0] == second.lower[3, 0, 0]
    assert first.upper[3, 0, 0] == second.upper[3, 0, 0]
    assert second.upper[4, 0, 0] > first.upper[4, 0, 0]


def test_complete_forecast_must_mature_before_scores_are_available() -> None:
    shape = (6, 1, 2)
    lower = np.zeros(shape)
    median = np.ones(shape)
    upper = np.full(shape, 2.0)
    truth = np.full(shape, 5.0)

    result = rolling_conformalize(
        lower,
        median,
        upper,
        truth,
        origin_times_utc=_daily_origins(6),
        outcome_delay_hours=72,
        window=2,
        min_history=1,
    )

    # Origin 0's last target matures at +73h, after origin 3 (+72h).
    assert not result.eligible[3].any()
    assert result.eligible[4].all()


def test_pooled_min_history_counts_complete_origins_not_score_cells() -> None:
    shape = (5, 2, 1)
    lower = np.zeros(shape)
    median = np.ones(shape)
    upper = np.full(shape, 2.0)
    truth = np.ones(shape)
    truth[1, 1, 0] = np.nan

    pooled = rolling_conformalize(
        lower,
        median,
        upper,
        truth,
        origin_times_utc=_daily_origins(5),
        outcome_delay_hours=0,
        window=2,
        min_history=2,
        pooled_series=True,
    )

    # At origin 2 there are three finite cells but only one complete origin.
    assert not pooled.eligible[2].any()
    # Origins 2 and 3 are complete and mature by origin 4.
    assert pooled.eligible[4].all()


def test_normalized_cqr_rescales_adjustment_by_current_width() -> None:
    lower = np.array([[[0.0]], [[0.0]]])
    median = np.array([[[1.0]], [[2.0]]])
    upper = np.array([[[2.0]], [[4.0]]])
    truth = np.array([[[5.0]], [[2.0]]])

    result = _calibrate(lower, median, upper, truth)

    assert result.adjustment[1, 0, 0] == pytest.approx(1.5)
    assert result.lower[1, 0, 0] == pytest.approx(-6.0)
    assert result.upper[1, 0, 0] == pytest.approx(10.0)


def test_negative_cqr_quantile_never_shrinks_served_interval() -> None:
    lower = np.array([[[0.0]], [[0.0]]])
    median = np.array([[[5.0]], [[5.0]]])
    upper = np.array([[[10.0]], [[10.0]]])
    truth = np.array([[[5.0]], [[5.0]]])

    result = _calibrate(lower, median, upper, truth)

    assert result.adjustment[1, 0, 0] == 0.0
    assert result.lower[1, 0, 0] == lower[1, 0, 0]
    assert result.upper[1, 0, 0] == upper[1, 0, 0]


def test_interval_metrics_calls_same_definitions_as_ledger() -> None:
    lower = np.array([0.0, 4.0, 8.0])
    median = np.array([1.0, 5.0, 9.0])
    upper = np.array([2.0, 6.0, 10.0])
    truth = np.array([3.0, 5.5, 7.0])

    metrics = interval_metrics(lower, median, upper, truth)
    expected_interval = np.mean(
        [
            interval_score_80(actual, low, high)
            for actual, low, high in zip(truth, lower, upper, strict=True)
        ]
    )
    expected_wis = np.mean(
        [
            weighted_interval_score(actual, low, mid, high)
            for actual, low, mid, high in zip(truth, lower, median, upper, strict=True)
        ]
    )

    assert metrics["interval_score"] == pytest.approx(expected_interval)
    assert metrics["wis"] == pytest.approx(expected_wis)


def test_interval_metrics_rejects_broadcasting_mask_and_crossing_quantiles() -> None:
    with pytest.raises(ValueError, match="mask shape"):
        interval_metrics(
            np.zeros((2, 1)),
            np.ones((2, 1)),
            np.full((2, 1), 2.0),
            np.ones((2, 1)),
            mask=np.ones(2, dtype=bool),
        )
    with pytest.raises(ValueError, match="lower <= median <= upper"):
        interval_metrics(
            np.array([2.0]),
            np.array([1.0]),
            np.array([3.0]),
            np.array([1.0]),
        )


def test_cqr_scores_can_be_negative_for_truth_inside_interval() -> None:
    scores = cqr_scores(
        np.array([0.0]),
        np.array([10.0]),
        np.array([4.0]),
        normalized=False,
    )

    assert scores[0] == -4.0


def test_common_mask_and_chronological_partition_are_predeclared() -> None:
    shape = (6, 1, 1)
    first_eligible = np.ones(shape, dtype=bool)
    first_eligible[:2] = False
    second_eligible = np.ones(shape, dtype=bool)
    second_eligible[3, 0, 0] = False
    empty = np.full(shape, np.nan)
    first = CandidateRun(
        pooled=False,
        window=2,
        calibrated=CalibratedIntervals(empty, empty, empty, first_eligible),
    )
    second = CandidateRun(
        pooled=True,
        window=2,
        calibrated=CalibratedIntervals(empty, empty, empty, second_eligible),
    )

    common = _common_eligible([first, second])
    selection, outer, split = _partition_masks(common, selection_fraction=0.5)

    assert split == 3
    assert selection[:, 0, 0].tolist() == [False, False, True, False, False, False]
    assert outer[:, 0, 0].tolist() == [False, False, False, False, True, True]
    assert not np.any(selection & outer)


def test_relative_selection_score_equal_weights_bas() -> None:
    per_ba = {
        "large": {
            "baseline": {"interval_score": 1_000.0},
            "calibrated": {"interval_score": 900.0},
        },
        "small": {
            "baseline": {"interval_score": 10.0},
            "calibrated": {"interval_score": 20.0},
        },
    }

    assert _relative_interval_score(per_ba) == pytest.approx((0.9 + 2.0) / 2.0)


def test_candidate_coverage_constraint_is_enforced_for_every_ba() -> None:
    shape = (1, 2, 5)
    lower = np.zeros(shape)
    median = np.ones(shape)
    upper = np.full(shape, 2.0)
    truth = np.array([[[1.0, 1.0, 1.0, 1.0, 3.0], [3.0, 3.0, 3.0, 3.0, 3.0]]])
    eligible = np.ones(shape, dtype=bool)
    calibrated = CalibratedIntervals(lower.copy(), upper.copy(), np.zeros(shape), eligible)

    summary = _candidate_summary(
        bas=["large", "small"],
        lower=lower,
        median=median,
        upper=upper,
        truth=truth,
        run=CandidateRun(pooled=False, window=28, calibrated=calibrated),
        mask=eligible,
        target_coverage=0.8,
        coverage_tolerance=0.02,
    )

    assert summary["per_ba"]["large"]["calibrated"]["coverage"] == 0.8
    assert summary["per_ba"]["small"]["calibrated"]["coverage"] == 0.0
    assert not summary["coverage_constraint_satisfied"]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"bas": [*RTO_BAS[:-1], "PJM"]}, "each of the seven"),
        ({"batch_size": 0}, "positive"),
        ({"coverage": 0.9}, "requires --coverage 0.8"),
        ({"windows": [28, 28]}, "unique"),
        ({"code_revision": "unknown"}, "40-character"),
        ({"data_snapshot_sha256": "unknown"}, "SHA-256"),
    ],
)
def test_cli_provenance_and_bounds_fail_before_inference(
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_args(_valid_args(**override))


def test_local_model_artifact_is_hashed_and_supplied_hash_is_verified(tmp_path: Path) -> None:
    artifact = tmp_path / "adapter"
    artifact.mkdir()
    (artifact / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    provenance = _validate_args(
        _valid_args(model=str(artifact), model_revision=None, model_artifact_sha256="")
    )

    assert provenance.is_local
    assert provenance.artifact_hash_algorithm == "sha256-tree-v1"
    assert provenance.artifact_sha256 is not None
    with pytest.raises(ValueError, match="does not match"):
        _validate_args(
            _valid_args(
                model=str(artifact),
                model_revision=None,
                model_artifact_sha256="0" * 64,
            )
        )


def test_alignment_rejects_any_partial_origin_overlap() -> None:
    first_origins = _daily_origins(2)
    second_origins = first_origins + np.timedelta64(1, "h")
    values = np.ones((2, 1))
    predictions = {
        "PJM": BAPredictions(first_origins, values, values, values, values),
        "CISO": BAPredictions(second_origins, values, values, values, values),
    }

    with pytest.raises(ValueError, match="exactly match"):
        _align_predictions(predictions, ["PJM", "CISO"])
