"""Serving invariants for the no-peeking forecaster."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import numpy as np
import pytest

from surge import model_loader
from surge.api import forecaster
from surge.features import (
    CALENDAR_COVARIATES,
    LOAD_V2_CORE,
    AvailabilityMode,
    BAData,
    calendar_covariates,
)
from surge.features import data as feature_data
from surge.model_loader import artifact_sha256


class FakePipeline:
    def __init__(self) -> None:
        self.task = None
        self.prediction_length = 0

    def predict_quantiles(
        self,
        tasks,
        *,
        prediction_length: int,
        quantile_levels: list[float],
        batch_size: int,
    ):
        assert quantile_levels == [0.1, 0.5, 0.9]
        assert batch_size == 1
        self.task = tasks[0]
        self.prediction_length = prediction_length
        base = np.arange(prediction_length, dtype=np.float32)
        quantiles = np.stack((base + 10, base + 20, base + 30), axis=-1)[None, ...]
        means = (base + 25)[None, ...]
        return [quantiles], [means]


def _lagged_history() -> BAData:
    length = 72
    end = np.datetime64("2026-01-02T06:00", "h")
    timestamps = np.arange(
        end - (length - 1) * np.timedelta64(1, "h"),
        end + np.timedelta64(1, "h"),
        np.timedelta64(1, "h"),
    ).astype("datetime64[us]")
    return BAData(
        ba="TEST",
        ts_utc=timestamps,
        target=np.arange(length, dtype=np.float32) + 20_000,
        covariates={
            "temp_c": np.linspace(-2, 8, length, dtype=np.float32),
            **calendar_covariates(timestamps),
        },
        future_keys=list(CALENDAR_COVARIATES),
        train_end=48,
        val_end=60,
        denom_mae=24.0,
        availability_mode=AvailabilityMode.EXACT_VINTAGE,
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


def test_forecast_returns_p50_mean_provenance_and_strict_future_points(monkeypatch) -> None:
    history = _lagged_history()
    monkeypatch.setattr(feature_data, "load_ba_data", lambda *args, **kwargs: history)
    monkeypatch.setattr(forecaster, "CONTEXT_LENGTH", 48)
    pipe = FakePipeline()
    issued = datetime(2026, 1, 2, 10, 37, tzinfo=UTC)

    result = forecaster.forecast_ba(
        pipe,
        "TEST",
        horizon=3,
        issued_at_utc=issued,
        feature_cutoff_utc=issued,
    )

    assert pipe.prediction_length == 7
    assert pipe.task is not None
    assert set(pipe.task["future_covariates"]) == set(CALENDAR_COVARIATES)
    assert "temp_c" not in pipe.task["future_covariates"]
    assert result["source_lag_hours"] == 4
    assert result["forecast_start_utc"] == datetime(2026, 1, 2, 11, tzinfo=UTC)
    assert result["feature_cutoff_utc"] == issued
    assert all(point["ts_utc"] > result["issued_at_utc"] for point in result["points"])

    first = result["points"][0]
    assert first["mean_mw"] == pytest.approx(29.0)
    assert first["p10_mw"] == pytest.approx(14.0)
    assert first["p50_mw"] == pytest.approx(24.0)
    assert first["median_mw"] == first["p50_mw"]
    assert first["p90_mw"] == pytest.approx(34.0)
    assert first["temp_c"] is None
    assert result["point_estimate_kind"] == "median"
    assert result["point_estimate_quantile"] == "p50"
    assert result["mase_scale_24"] == pytest.approx(24.0)
    assert result["provenance"]["model_revision"] == forecaster.MODEL_REVISION
    assert result["provenance"]["code_revision"] == forecaster.CODE_REVISION
    assert result["provenance"]["feature_snapshot_sha256"] == result["feature_snapshot_sha256"]


def test_forecast_rejects_cutoff_after_issuance(monkeypatch) -> None:
    history = _lagged_history()
    monkeypatch.setattr(feature_data, "load_ba_data", lambda *args, **kwargs: history)
    monkeypatch.setattr(forecaster, "CONTEXT_LENGTH", 48)
    issued = datetime(2026, 1, 2, 10, 37, tzinfo=UTC)

    with pytest.raises(ValueError, match="cannot be after"):
        forecaster.forecast_ba(
            FakePipeline(),
            "TEST",
            issued_at_utc=issued,
            feature_cutoff_utc=datetime(2026, 1, 2, 10, 38, tzinfo=UTC),
        )


def test_model_revision_default_only_applies_to_pinned_upstream() -> None:
    assert (
        forecaster._resolve_model_revision(forecaster.DEFAULT_CHRONOS2_MODEL, None)
        == forecaster.PINNED_CHRONOS2_REVISION
    )
    assert (
        forecaster._resolve_model_revision("/models/surge-candidate", None)
        == forecaster.UNKNOWN_MODEL_REVISION
    )
    assert (
        forecaster._resolve_model_revision("/models/surge-candidate", "candidate-1")
        == "candidate-1"
    )


def test_artifact_digest_cache_hashes_unchanged_tree_once(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "surge-candidate"
    artifact.mkdir()
    (artifact / "adapter_config.json").write_text("{}")
    (artifact / "adapter.safetensors").write_bytes(b"weights")
    model_loader.clear_artifact_sha256_cache()

    original_sha256_file = model_loader._sha256_file
    hashed_paths = []

    def counting_sha256_file(path):
        hashed_paths.append(path)
        return original_sha256_file(path)

    monkeypatch.setattr(model_loader, "_sha256_file", counting_sha256_file)

    first = artifact_sha256(artifact)
    second = artifact_sha256(artifact)

    assert second == first
    assert sorted(path.name for path in hashed_paths) == [
        "adapter.safetensors",
        "adapter_config.json",
    ]


def test_artifact_digest_cache_invalidates_same_size_content_change(tmp_path) -> None:
    artifact = tmp_path / "surge-candidate.bin"
    artifact.write_bytes(b"original")
    model_loader.clear_artifact_sha256_cache()
    original = artifact_sha256(artifact)
    before = artifact.stat()

    artifact.write_bytes(b"tampered")
    os.utime(
        artifact,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
    )
    after = artifact.stat()

    assert after.st_size == before.st_size
    assert (after.st_mtime_ns, after.st_ctime_ns) != (
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    assert artifact_sha256(artifact) != original


def test_artifact_digest_rejects_symlinks(tmp_path) -> None:
    artifact = tmp_path / "surge-candidate"
    artifact.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"weights")
    (artifact / "adapter.safetensors").symlink_to(outside)
    model_loader.clear_artifact_sha256_cache()

    with pytest.raises(ValueError, match="symlinks are not allowed"):
        artifact_sha256(artifact)


def test_custom_model_requires_contract_and_computed_artifact_attestation(
    tmp_path, monkeypatch
) -> None:
    artifact = tmp_path / "surge-candidate"
    artifact.mkdir()
    weights = artifact / "adapter.safetensors"
    weights.write_bytes(b"original weights")

    monkeypatch.setattr(forecaster, "MODEL_PATH", str(artifact))
    monkeypatch.setattr(forecaster, "MODEL_REVISION", "candidate-1")
    monkeypatch.setattr(
        forecaster,
        "MODEL_FEATURE_SPEC_SHA256",
        LOAD_V2_CORE.sha256,
    )
    monkeypatch.setattr(forecaster, "MODEL_ARTIFACT_SHA256", None)
    assert not forecaster.model_release_safe()

    monkeypatch.setattr(forecaster, "MODEL_ARTIFACT_SHA256", "not-a-sha")
    assert not forecaster.model_release_safe()

    expected_sha256 = artifact_sha256(artifact)
    monkeypatch.setattr(forecaster, "MODEL_ARTIFACT_SHA256", expected_sha256)
    assert forecaster.model_release_safe()

    monkeypatch.setattr(
        forecaster,
        "MODEL_REVISION",
        forecaster.UNKNOWN_MODEL_REVISION,
    )
    assert not forecaster.model_release_safe()
    monkeypatch.setattr(forecaster, "MODEL_REVISION", "candidate-1")

    before = weights.stat()
    weights.write_bytes(b"tampered weights")
    os.utime(
        weights,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
    )
    assert weights.stat().st_size == before.st_size
    assert artifact_sha256(artifact) != expected_sha256
    assert not forecaster.model_release_safe()


def test_custom_model_provenance_uses_computed_digest(tmp_path, monkeypatch) -> None:
    history = _lagged_history()
    monkeypatch.setattr(feature_data, "load_ba_data", lambda *args, **kwargs: history)
    monkeypatch.setattr(forecaster, "CONTEXT_LENGTH", 48)

    artifact = tmp_path / "surge-candidate.bin"
    artifact.write_bytes(b"candidate weights")
    expected_sha256 = artifact_sha256(artifact)
    monkeypatch.setattr(forecaster, "MODEL_PATH", str(artifact))
    monkeypatch.setattr(forecaster, "MODEL_REVISION", "candidate-1")
    monkeypatch.setattr(
        forecaster,
        "MODEL_FEATURE_SPEC_SHA256",
        LOAD_V2_CORE.sha256,
    )
    monkeypatch.setattr(forecaster, "MODEL_ARTIFACT_SHA256", expected_sha256)

    issued = datetime(2026, 1, 2, 10, 37, tzinfo=UTC)
    result = forecaster.forecast_ba(
        FakePipeline(),
        "TEST",
        horizon=1,
        issued_at_utc=issued,
        feature_cutoff_utc=issued,
    )

    assert result["provenance"]["model_artifact_sha256"] == expected_sha256
    assert result["provenance"]["model_release_safe"] is True

    artifact.write_bytes(b"tampered candidate weights")
    tampered = forecaster.forecast_ba(
        FakePipeline(),
        "TEST",
        horizon=1,
        issued_at_utc=issued,
        feature_cutoff_utc=issued,
    )
    assert tampered["provenance"]["model_artifact_sha256"] == artifact_sha256(artifact)
    assert tampered["provenance"]["model_artifact_sha256"] != expected_sha256
    assert tampered["provenance"]["model_release_safe"] is False
