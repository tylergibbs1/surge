"""No-peeking feature-contract tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from surge import store
from surge.features import (
    CALENDAR_COVARIATES,
    LOAD_V2_CORE,
    OBSERVED_COVARIATES,
    AvailabilityMode,
    BAData,
    FeatureContractError,
    build_evaluation_task,
    build_live_bundle_from_data,
    build_rolling_validation_tasks,
    build_training_task,
    calendar_covariates,
    load_ba_data,
    validate_task,
)


def _history(
    *, availability_mode: AvailabilityMode = AvailabilityMode.RETROSPECTIVE_FINAL
) -> BAData:
    length = 96
    start = np.datetime64("2025-12-29T07:00", "h")
    timestamps: np.ndarray = np.arange(
        start,
        start + length * np.timedelta64(1, "h"),
        np.timedelta64(1, "h"),
    )
    calendar = calendar_covariates(timestamps)
    covariates = {
        "temp_c": np.linspace(-5, 12, length, dtype=np.float32),
        "wind_mw": np.linspace(100, 200, length, dtype=np.float32),
        "solar_mw": np.linspace(0, 80, length, dtype=np.float32),
        **calendar,
    }
    return BAData(
        ba="TEST",
        ts_utc=timestamps.astype("datetime64[us]"),
        target=np.arange(length, dtype=np.float32) + 1_000,
        covariates=covariates,
        future_keys=list(CALENDAR_COVARIATES),
        train_end=48,
        val_end=72,
        denom_mae=24.0,
        availability_mode=availability_mode,
        provenance={
            "load_hourly": {
                "rows": length,
                "observed_end_utc": datetime(2026, 1, 2, 6, tzinfo=UTC),
            },
            "weather_hourly": {
                "rows": length,
                "observed_end_utc": datetime(2026, 1, 2, 6, tzinfo=UTC),
            },
        },
    )


def _assert_tasks_equal(left: dict, right: dict) -> None:
    np.testing.assert_array_equal(left["target"], right["target"])
    assert left["past_covariates"].keys() == right["past_covariates"].keys()
    assert left["future_covariates"].keys() == right["future_covariates"].keys()
    for section in ("past_covariates", "future_covariates"):
        for key in left[section]:
            np.testing.assert_array_equal(left[section][key], right[section][key])


def test_load_v2_core_has_calendar_only_future_covariates() -> None:
    assert LOAD_V2_CORE.future_covariates == CALENDAR_COVARIATES
    assert not set(LOAD_V2_CORE.future_covariates) & set(OBSERVED_COVARIATES)


@pytest.mark.parametrize("observed_key", OBSERVED_COVARIATES)
def test_contract_rejects_observed_future_covariates(observed_key: str) -> None:
    data = _history()
    task = build_evaluation_task(data, origin=72, context_length=48, prediction_length=24)
    task["future_covariates"][observed_key] = np.zeros(24, dtype=np.float32)
    with pytest.raises(FeatureContractError, match="observed covariates"):
        validate_task(task, prediction_length=24)


@pytest.mark.parametrize("observed_key", OBSERVED_COVARIATES)
def test_mutating_future_observations_cannot_change_model_task(observed_key: str) -> None:
    origin = 72
    baseline_data = _history()
    baseline = build_evaluation_task(
        baseline_data, origin=origin, context_length=48, prediction_length=24
    )

    mutated_data = _history()
    mutated_data.covariates[observed_key][origin:] = 1_000_000
    mutated = build_evaluation_task(
        mutated_data, origin=origin, context_length=48, prediction_length=24
    )

    _assert_tasks_equal(baseline, mutated)
    assert set(mutated["future_covariates"]) == set(CALENDAR_COVARIATES)


def test_training_declares_only_calendar_future_keys() -> None:
    task = build_training_task(_history(), start=0, end=48, prediction_length=24)
    assert tuple(task["future_covariates"]) == CALENDAR_COVARIATES
    assert all(len(values) == 0 for values in task["future_covariates"].values())
    assert set(OBSERVED_COVARIATES).issubset(task["past_covariates"])


def test_checkpoint_validation_uses_multiple_validation_only_label_windows() -> None:
    data = _history()
    tasks = build_rolling_validation_tasks(
        data,
        context_length=24,
        prediction_length=6,
        step=6,
        max_origins=3,
    )

    assert len(tasks) == 3
    assert all(len(task["target"]) == 30 for task in tasks)
    np.testing.assert_array_equal(tasks[0]["target"][-6:], data.target[54:60])
    np.testing.assert_array_equal(tasks[-1]["target"][-6:], data.target[66:72])
    assert all(
        np.min(task["target"][-6:]) >= np.min(data.target[data.train_end :])
        for task in tasks
    )


def test_checkpoint_validation_skips_incomplete_labels_before_trailing_limit() -> None:
    data = _history()
    data.target[66] = np.nan

    tasks = build_rolling_validation_tasks(
        data,
        context_length=24,
        prediction_length=6,
        step=6,
        max_origins=3,
    )

    assert len(tasks) == 3
    np.testing.assert_array_equal(tasks[0]["target"][-6:], data.target[48:54])
    np.testing.assert_array_equal(tasks[-1]["target"][-6:], data.target[60:66])
    assert all(np.isfinite(task["target"][-6:]).all() for task in tasks)


def test_live_bundle_discards_source_lag_and_starts_strictly_after_issue() -> None:
    issued = datetime(2026, 1, 2, 10, 37, tzinfo=UTC)
    cutoff = issued
    data = _history(availability_mode=AvailabilityMode.EXACT_VINTAGE)
    # Restrict context to a source whose newest observation is 06:00 UTC.
    keep = data.ts_utc <= np.datetime64("2026-01-02T06:00", "us")
    data.ts_utc = data.ts_utc[keep]
    data.target = data.target[keep]
    data.covariates = {key: value[keep] for key, value in data.covariates.items()}

    bundle = build_live_bundle_from_data(
        data,
        issued_at_utc=issued,
        cutoff_utc=cutoff,
        horizon=3,
        context_length=24,
    )

    assert bundle.discard_prefix == 4  # model covers 07:00-10:00, serves from 11:00
    assert bundle.prediction_length == 7
    assert bundle.forecast_start_utc == datetime(2026, 1, 2, 11, tzinfo=UTC)
    assert all(timestamp > issued for timestamp in map(_numpy_utc, bundle.served_ts_utc))
    assert set(bundle.task["future_covariates"]) == set(CALENDAR_COVARIATES)


def test_live_bundle_rejects_stale_required_source_even_with_fresh_context() -> None:
    issued = datetime(2026, 1, 2, 10, 37, tzinfo=UTC)
    data = _history(availability_mode=AvailabilityMode.EXACT_VINTAGE)
    data.provenance["weather_hourly"]["observed_end_utc"] = datetime(2026, 1, 1, 20, tzinfo=UTC)

    with pytest.raises(ValueError, match="weather_hourly is stale"):
        build_live_bundle_from_data(
            data,
            issued_at_utc=issued,
            cutoff_utc=issued,
            horizon=3,
            context_length=24,
        )


def test_contract_preserves_historical_nan_but_rejects_infinity_and_future_nan() -> None:
    task = build_evaluation_task(_history(), origin=72, context_length=48, prediction_length=24)
    task["target"][3] = np.nan
    task["past_covariates"]["temp_c"][4] = np.nan
    validate_task(task, prediction_length=24)

    task["target"][5] = np.inf
    with pytest.raises(FeatureContractError, match="infinite"):
        validate_task(task, prediction_length=24)

    task["target"][5] = 1_000
    task["future_covariates"]["hour_sin"][0] = np.nan
    with pytest.raises(FeatureContractError, match="fully known and finite"):
        validate_task(task, prediction_length=24)


def _numpy_utc(value: np.datetime64) -> datetime:
    micros = int(value.astype("datetime64[us]").astype(np.int64))
    return datetime.fromtimestamp(micros / 1_000_000, tz=UTC)


def test_exact_vintage_and_retrospective_final_are_distinct(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(hours=index) for index in range(30)]
    early_as_of = datetime(2024, 1, 3, tzinfo=UTC)
    cutoff = datetime(2024, 1, 4, tzinfo=UTC)
    late_as_of = datetime(2024, 1, 5, tzinfo=UTC)

    load = pl.DataFrame(
        {
            "ts_utc": timestamps,
            "ba": ["TEST"] * len(timestamps),
            "load_mw": np.arange(30, dtype=float) + 1_000,
            "source": ["fixture"] * len(timestamps),
            "as_of": [early_as_of] * len(timestamps),
        }
    )
    weather = pl.DataFrame(
        {
            "ts_utc": timestamps,
            "ba": ["TEST"] * len(timestamps),
            "temp_c": np.arange(30, dtype=float),
            "source": ["fixture"] * len(timestamps),
            "as_of": [early_as_of] * len(timestamps),
        }
    )
    store.append("load_hourly", load)
    store.append("weather_hourly", weather)
    store.append(
        "load_hourly",
        load.tail(1).with_columns(
            pl.lit(9_999.0).alias("load_mw"), pl.lit(late_as_of).alias("as_of")
        ),
    )
    store.append(
        "weather_hourly",
        weather.tail(1).with_columns(
            pl.lit(999.0).alias("temp_c"), pl.lit(late_as_of).alias("as_of")
        ),
    )

    exact = load_ba_data("TEST", availability_mode=AvailabilityMode.EXACT_VINTAGE, cutoff=cutoff)
    final = load_ba_data("TEST", availability_mode=AvailabilityMode.RETROSPECTIVE_FINAL)

    assert exact.target[-1] == pytest.approx(1_029.0)
    assert exact.covariates["temp_c"][-1] == pytest.approx(29.0)
    assert final.target[-1] == pytest.approx(9_999.0)
    assert final.covariates["temp_c"][-1] == pytest.approx(999.0)
    assert exact.provenance["load_hourly"]["max_available_at_utc"] <= cutoff
    assert final.availability_mode is AvailabilityMode.RETROSPECTIVE_FINAL


def test_loader_preserves_missing_observations_instead_of_forward_filling(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(hours=index) for index in range(30)]
    as_of = datetime(2024, 1, 3, tzinfo=UTC)
    load_values: list[float | None] = [1_000.0 + index for index in range(30)]
    load_values[5] = None
    store.append(
        "load_hourly",
        pl.DataFrame(
            {
                "ts_utc": timestamps,
                "ba": ["TEST"] * 30,
                "load_mw": load_values,
                "source": ["fixture"] * 30,
                "as_of": [as_of] * 30,
            }
        ),
    )
    store.append(
        "weather_hourly",
        pl.DataFrame(
            {
                "ts_utc": timestamps,
                "ba": ["TEST"] * 30,
                "temp_c": [10.0 + index for index in range(30)],
                "source": ["fixture"] * 30,
                "as_of": [as_of] * 30,
            }
        ),
    )

    loaded = load_ba_data("TEST")

    assert np.isnan(loaded.target[5])
    assert loaded.target[6] == pytest.approx(1_006.0)
    assert any("preserved 1 missing load_mw" in warning for warning in loaded.warnings)


def test_rolling_eval_scores_the_served_p50_not_the_model_mean() -> None:
    from experiments.eval_c2 import rolling_eval_c2

    data = _history()
    truth = data.target[data.train_end : data.val_end]

    class DeliberatelyDifferentMeanPipeline:
        def predict_quantiles(self, tasks, **kwargs):
            assert len(tasks) == 1
            quantiles = np.stack((truth - 1, truth, truth + 1), axis=-1)
            wrong_mean = np.full_like(truth, -1_000_000)
            return [quantiles[None, ...]], [wrong_mean[None, ...]]

    metrics = rolling_eval_c2(
        DeliberatelyDifferentMeanPipeline(),
        {data.ba: data},
        on="val",
        context=48,
        horizon=24,
        step=24,
        batch_size=1,
    )

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["per_ba"][data.ba]["point_estimate_kind"] == "median"
    assert metrics["per_ba"][data.ba]["point_estimate_quantile"] == "p50"
