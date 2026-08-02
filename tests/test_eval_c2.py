"""Origin-block confidence interval and finite-data evaluator tests."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.eval_c2 import (
    _paired_origin_block_mase_ci,
    rolling_eval_c2,
    select_shared_complete_origin_schedule,
)
from surge.features import BAData, calendar_covariates


def _data(
    ba: str,
    *,
    start: str = "2024-01-01T00:00",
    target: np.ndarray | None = None,
    denom_mae: float = 1.0,
) -> BAData:
    if target is None:
        target = np.zeros(3, dtype=np.float32)
    timestamps = np.arange(
        np.datetime64(start, "h"),
        np.datetime64(start, "h") + len(target) * np.timedelta64(1, "h"),
        np.timedelta64(1, "h"),
    ).astype("datetime64[us]")
    calendar = calendar_covariates(timestamps)
    return BAData(
        ba=ba,
        ts_utc=timestamps,
        target=target,
        covariates={"temp_c": np.zeros(len(target), dtype=np.float32), **calendar},
        future_keys=list(calendar),
        train_end=1,
        val_end=len(target),
        denom_mae=denom_mae,
    )


class PatternPipeline:
    """Return one predetermined p50 forecast pattern per BA evaluation call."""

    def __init__(self, patterns: list[list[float]]) -> None:
        self._patterns = iter(patterns)

    def predict_quantiles(self, tasks, **kwargs):
        errors = next(self._patterns)
        assert len(errors) == len(tasks)
        horizon = kwargs["prediction_length"]
        quantiles = []
        means = []
        for error in errors:
            median = np.full(horizon, error, dtype=np.float32)
            row = np.stack(
                (np.maximum(median - 0.25, 0.0), median, median + 0.25),
                axis=-1,
            )
            quantiles.append(row[None, ...])
            means.append(median[None, ...])
        return quantiles, means


class FixedQuantilePipeline:
    def __init__(self, batches: list[np.ndarray]) -> None:
        self._batches = iter(batches)

    def predict_quantiles(self, tasks, **kwargs):
        batch = next(self._batches)
        expected_shape = (len(tasks), kwargs["prediction_length"], 3)
        assert batch.shape == expected_shape
        quantiles = [row[None, ...] for row in batch]
        means = [row[None, ..., 1] for row in batch]
        return quantiles, means


def test_protocol_metrics_and_per_horizon_rows_are_exact() -> None:
    data = _data(
        "PJM",
        target=np.asarray([5.0, 10.0, 20.0], dtype=np.float32),
        denom_mae=2.0,
    )
    quantiles = np.asarray(
        [
            [8.0, 12.0, 16.0],
            [12.0, 14.0, 16.0],
        ],
        dtype=np.float32,
    )

    metrics = rolling_eval_c2(
        FixedQuantilePipeline([quantiles[None, ...]]),
        {"PJM": data},
        context=1,
        horizon=2,
        step=2,
        batch_size=1,
        per_step=True,
    )

    ba = metrics["per_ba"]["PJM"]
    assert ba["bias"] == pytest.approx(-2.0)  # forecast - actual: mean([2, -6])
    assert ba["mae"] == pytest.approx(4.0)
    assert ba["rmse"] == pytest.approx(np.sqrt(20.0))
    assert ba["mase"] == pytest.approx(2.0)
    assert ba["pinball_p10"] == pytest.approx(0.5)
    assert ba["pinball_p50"] == pytest.approx(2.0)
    assert ba["pinball_p90"] == pytest.approx(2.1)
    assert ba["mean_pi80_width"] == pytest.approx(6.0)
    assert ba["cov_pi80"] == pytest.approx(0.5)
    assert ba["wis"] == pytest.approx((1.2 + 7.4 / 1.5) / 2.0)
    assert ba["crps_approx"] == pytest.approx(
        2.0 * np.mean([ba["pinball_p10"], ba["pinball_p50"], ba["pinball_p90"]])
    )
    assert "crps" not in ba

    per_horizon = ba["per_horizon"]
    assert [row["horizon"] for row in per_horizon] == [1, 2]
    assert [row["n_points"] for row in per_horizon] == [1, 1]
    assert per_horizon[0]["bias"] == pytest.approx(2.0)
    assert per_horizon[0]["mase"] == pytest.approx(1.0)
    assert per_horizon[0]["cov_pi80"] == pytest.approx(1.0)
    assert per_horizon[1]["bias"] == pytest.approx(-6.0)
    assert per_horizon[1]["mase"] == pytest.approx(3.0)
    assert per_horizon[1]["cov_pi80"] == pytest.approx(0.0)

    assert metrics["aggregation"] == "equal_ba_macro"
    assert [row["horizon"] for row in metrics["per_horizon"]] == [1, 2]
    assert all(row["aggregation"] == "equal_ba_macro" for row in metrics["per_horizon"])
    for macro_row, ba_row in zip(metrics["per_horizon"], per_horizon, strict=True):
        assert macro_row["mase"] == pytest.approx(ba_row["mase"])
        assert macro_row["bias"] == pytest.approx(ba_row["bias"])
        assert macro_row["wis"] == pytest.approx(ba_row["wis"])
    assert metrics["per_step_mase"] == pytest.approx([1.0, 3.0])
    assert metrics["crps_approximation"] == "2x_mean_pinball"
    assert metrics["crps_approx_quantile_levels"] == [0.1, 0.5, 0.9]


def test_load_weighted_metrics_are_separate_from_equal_ba_macro() -> None:
    metrics = rolling_eval_c2(
        PatternPipeline([[11.0, 11.0], [33.0, 33.0]]),
        {
            "PJM": _data(
                "PJM", target=np.asarray([10.0, 10.0, 10.0], dtype=np.float32)
            ),
            "CISO": _data(
                "CISO", target=np.asarray([30.0, 30.0, 30.0], dtype=np.float32)
            ),
        },
        context=1,
        horizon=1,
        step=1,
        batch_size=2,
    )

    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["aggregation"] == "equal_ba_macro"
    weighted = metrics["load_weighted"]
    assert weighted["mae"] == pytest.approx(2.5)
    assert weighted["aggregation"] == "mean_actual_mw_weighted_mean_of_ba_metrics"
    assert weighted["weights"] == pytest.approx({"PJM": 0.25, "CISO": 0.75})


def test_bootstrap_preserves_same_origin_cross_ba_dependence() -> None:
    # Every paired origin has macro MASE 1: (0 + 2) / 2 and (2 + 0) / 2.
    # Sampling the four BA-window values independently would create a
    # non-degenerate interval, so [1, 1] proves that cross-BA rows stay paired.
    metrics = rolling_eval_c2(
        PatternPipeline([[0.0, 2.0], [2.0, 0.0]]),
        {"PJM": _data("PJM"), "CISO": _data("CISO")},
        context=1,
        horizon=1,
        step=1,
        batch_size=2,
        bootstrap=500,
        seed=42,
    )

    assert metrics["mase"] == pytest.approx(1.0)
    assert metrics["mase_ci_low"] == pytest.approx(1.0)
    assert metrics["mase_ci_high"] == pytest.approx(1.0)


def test_bootstrap_uses_identical_schedule_and_equal_ba_macro() -> None:
    low, high = _paired_origin_block_mase_ci(
        {
            "PJM": {1: 0.0, 2: 0.0},
            "CISO": {1: 10.0, 2: 10.0},
        },
        bootstrap=100,
        seed=11,
    )

    # Each BA gets half the macro weight:
    # mean([mean(0, 0), mean(10, 10)]) == 5.
    assert low == pytest.approx(5.0)
    assert high == pytest.approx(5.0)


def test_bootstrap_rejects_ba_only_origins_instead_of_intersecting() -> None:
    with pytest.raises(ValueError, match="identical origin schedules"):
        _paired_origin_block_mase_ci(
            {
                "PJM": {1: 0.0, 2: 0.0, 3: 1_000.0},
                "CISO": {1: 10.0, 2: 10.0},
            },
            bootstrap=100,
            seed=11,
        )


def test_bootstrap_rejects_disjoint_ba_origin_schedules() -> None:
    with pytest.raises(ValueError, match="identical origin schedules"):
        rolling_eval_c2(
            PatternPipeline([[0.0, 1.0], [1.0, 0.0]]),
            {
                "PJM": _data("PJM", start="2024-01-01T00:00"),
                "CISO": _data("CISO", start="2024-01-02T00:00"),
            },
            context=1,
            horizon=1,
            step=1,
            batch_size=2,
            bootstrap=20,
            seed=7,
        )


def test_bootstrap_rejects_requested_ba_without_origins() -> None:
    with pytest.raises(ValueError, match="CISO bootstrap requires at least one evaluable origin"):
        rolling_eval_c2(
            PatternPipeline([[0.0, 0.0]]),
            {
                "PJM": _data("PJM"),
                "CISO": _data("CISO", target=np.asarray([0.0], dtype=np.float32)),
            },
            context=1,
            horizon=1,
            step=1,
            batch_size=2,
            bootstrap=20,
            seed=7,
        )


def test_bootstrap_rejects_partial_target_windows() -> None:
    target = np.asarray([1.0, 1.0, np.nan], dtype=np.float32)

    with pytest.raises(ValueError, match="all 2 target hours finite"):
        rolling_eval_c2(
            PatternPipeline([[1.0]]),
            {"PJM": _data("PJM", target=target)},
            context=1,
            horizon=2,
            step=2,
            batch_size=1,
            bootstrap=20,
            seed=7,
        )


def test_complete_origin_policy_filters_before_trailing_limit() -> None:
    target = np.zeros(7, dtype=np.float32)
    target[3] = np.nan
    data = _data("PJM", target=target)

    metrics = rolling_eval_c2(
        PatternPipeline([[0.0, 0.0]]),
        {"PJM": data},
        context=1,
        horizon=2,
        step=2,
        batch_size=2,
        max_origins=2,
        require_complete_origins=True,
    )

    row = metrics["per_ba"]["PJM"]
    assert metrics["complete_target_origins_only"] is True
    assert row["n_windows"] == 2
    assert row["n_points"] == 4
    assert row["candidate_origins_before_limit"] == 3
    assert row["incomplete_target_origins_before_limit"] == 1
    assert row["origin_start_utc"] == str(data.ts_utc[1])
    assert row["origin_end_utc"] == str(data.ts_utc[5])


def test_shared_schedule_is_disjoint_across_checkpoint_and_promotion_cohorts() -> None:
    pjm_target = np.zeros(10, dtype=np.float32)
    ciso_target = np.zeros(10, dtype=np.float32)
    pjm_target[5] = np.nan
    ciso_target[6] = np.nan
    bas = {
        "PJM": _data("PJM", target=pjm_target),
        "CISO": _data("CISO", target=ciso_target),
    }

    promotion = select_shared_complete_origin_schedule(
        bas,
        on="val",
        context=2,
        horizon=1,
        step=1,
        origin_count=2,
    )
    checkpoint = select_shared_complete_origin_schedule(
        bas,
        on="val",
        context=2,
        horizon=1,
        step=1,
        origin_count=2,
        exclude_latest_complete=2,
    )

    assert set(promotion.origin_keys_utc_us).isdisjoint(
        checkpoint.origin_keys_utc_us
    )
    assert [str(np.datetime64(key, "us")) for key in checkpoint.origin_keys_utc_us] == [
        str(bas["PJM"].ts_utc[4]),
        str(bas["PJM"].ts_utc[7]),
    ]

    metrics = rolling_eval_c2(
        PatternPipeline([[0.0, 0.0], [0.0, 0.0]]),
        bas,
        context=2,
        horizon=1,
        step=1,
        batch_size=2,
        require_complete_origins=True,
        origin_schedule=checkpoint,
        emit_origin_metrics=True,
    )

    assert metrics["shared_origin_schedule"] is True
    assert metrics["origin_schedule"]["excluded_latest_complete_count"] == 2
    assert {
        row["origin_sha256"] for row in metrics["per_ba"].values()
    } == {checkpoint.sha256}


def test_evaluator_rejects_negative_load_quantiles() -> None:
    quantiles = np.asarray([[[-0.1, 0.0, 1.0]]], dtype=np.float32)

    with pytest.raises(ValueError, match="negative load quantiles"):
        rolling_eval_c2(
            FixedQuantilePipeline([quantiles]),
            {"PJM": _data("PJM", target=np.asarray([1.0, 1.0], dtype=np.float32))},
            context=1,
            horizon=1,
            step=1,
            batch_size=1,
        )


def test_evaluator_rejects_ba_with_zero_finite_target_points() -> None:
    target = np.asarray([0.0, np.nan, np.nan], dtype=np.float32)

    with pytest.raises(ValueError, match="PJM has zero finite target points"):
        rolling_eval_c2(
            PatternPipeline([[0.0]]),
            {"PJM": _data("PJM", target=target)},
            context=1,
            horizon=2,
            step=2,
            batch_size=1,
        )


@pytest.mark.parametrize("denom_mae", [0.0, np.nan, np.inf])
def test_evaluator_rejects_ba_without_finite_window_mase(denom_mae: float) -> None:
    with pytest.raises(ValueError, match="PJM has no finite per-window MASE"):
        rolling_eval_c2(
            PatternPipeline([]),
            {"PJM": _data("PJM", denom_mae=denom_mae)},
            context=1,
            horizon=1,
            step=1,
        )
