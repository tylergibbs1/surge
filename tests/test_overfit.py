"""Fail-closed model-selection and generalization diagnostics."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import experiments.overfit as overfit_module
from experiments.eval_c2 import rolling_eval_c2
from experiments.overfit import (
    FROZEN_DATA_SNAPSHOT_SHA256,
    POLICY_VERSION,
    RELEASE_BASE_MODEL_ID,
    RELEASE_BASE_REVISION,
    TRUST_RTO_BAS,
    build_overfit_audit,
    complete_locked_test_run,
    fail_locked_test_run,
    rejected_overfit_audit,
    reserve_locked_test_run,
    revision_for_model_load,
    validate_promotion_inputs,
    validate_release_lineage,
    validate_reproducibility_runtime,
    validation_only_view,
    verify_promotion_artifact,
    verify_selection_artifact,
)
from surge import store
from surge.features import BAData, calendar_covariates, load_ba_data
from surge.model_loader import artifact_sha256


def _metrics(
    split: str,
    *,
    mase: float,
    wis_scaled: float,
    windows: int = 90,
) -> dict:
    origin_keys = np.arange(windows, dtype="<i8") * 86_400_000_000
    if split == "val":
        origin_keys += 1_727_827_200_000_000
    else:
        origin_keys += 1_672_531_200_000_000
    origin_strings = [str(np.datetime64(int(key), "us")) for key in origin_keys]
    origin_sha256 = hashlib.sha256(origin_keys.tobytes()).hexdigest()
    return {
        "split": split,
        "aggregation": "equal_ba_macro",
        "point_estimate_kind": "median",
        "point_estimate_quantile": "p50",
        "point_estimate_quantile_value": 0.5,
        "horizon": 24,
        "origin_step_hours": 24,
        "crps_approximation": "2x_mean_pinball",
        "crps_approx_quantile_levels": [0.1, 0.5, 0.9],
        "complete_target_origins_only": True,
        "shared_origin_schedule": True,
        "origin_metrics_emitted": split == "val",
        "origin_schedule": {
            "split": split,
            "shared_across_bas": True,
            "complete_target_origins_only": True,
            "origin_count": windows,
            "requested_origin_count": windows,
            "step_hours": 24,
            "origin_start_utc": origin_strings[0],
            "origin_end_utc": origin_strings[-1],
            "origins_utc": origin_strings,
            "origin_sha256": origin_sha256,
        },
        "per_ba": {
            ba: {
                "mase": mase + index * 0.01,
                "wis_scaled": wis_scaled + index * 0.01,
                "n_windows": windows,
                "n_points": windows * 24,
                "origin_start_utc": origin_strings[0],
                "origin_end_utc": origin_strings[-1],
                "origin_step_hours": 24,
                "origin_sha256": origin_sha256,
                "origin_mase": [
                    {"origin_utc": origin, "mase": mase + index * 0.01}
                    for origin in origin_strings
                ],
            }
            for index, ba in enumerate(TRUST_RTO_BAS)
        },
    }


def _shift_schedule(schedule: dict, *, days: int) -> dict:
    shifted = dict(schedule)
    origin_keys = np.asarray(
        [
            int(np.datetime64(origin).astype("datetime64[us]").astype(np.int64))
            for origin in schedule["origins_utc"]
        ],
        dtype="<i8",
    )
    origin_keys += days * 86_400_000_000
    origins = [str(np.datetime64(int(key), "us")) for key in origin_keys]
    shifted.update(
        {
            "origin_start_utc": origins[0],
            "origin_end_utc": origins[-1],
            "origins_utc": origins,
            "origin_sha256": hashlib.sha256(origin_keys.tobytes()).hexdigest(),
        }
    )
    return shifted


def _add_frozen_schedules(audit: dict) -> None:
    train_schedule = _metrics(
        "train", mase=0.55, wis_scaled=0.45
    )["origin_schedule"]
    promotion_schedule = _metrics(
        "val", mase=0.70, wis_scaled=0.60
    )["origin_schedule"]
    checkpoint_schedule = _shift_schedule(promotion_schedule, days=-90)
    checkpoint_schedule["excluded_latest_complete_count"] = 90
    audit["promotion_train_schedule"] = train_schedule
    audit["promotion_validation_schedule"] = promotion_schedule
    audit["checkpoint_selection_schedule"] = checkpoint_schedule


def _healthy_history() -> tuple[list[dict], dict]:
    events = [
        {"event": "log", "step": 100, "train_loss": 1.0},
        {"event": "log", "step": 100, "eval_loss": 0.90},
        {"event": "checkpoint", "step": 100},
        {"event": "log", "step": 200, "train_loss": 0.8},
        {"event": "log", "step": 200, "eval_loss": 0.85},
        {"event": "checkpoint", "step": 200},
        {"event": "log", "step": 300, "train_loss": 0.7},
        {"event": "log", "step": 300, "eval_loss": 0.87},
        {"event": "checkpoint", "step": 300},
    ]
    state = {
        "global_step": 300,
        "best_metric": 0.85,
        "best_model_checkpoint": "/tmp/output/checkpoint-200",
    }
    return events, state


def _runtime_identity() -> dict:
    return {
        "system": {
            "python_version": "test",
            "cuda_available": True,
            "accelerator_count": 1,
            "accelerators": [{"name": "NVIDIA H100 test"}],
            "deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": True,
            "cudnn_allow_tf32": True,
            "float32_matmul_precision": "high",
            "cublas_workspace_config": ":4096:8",
        },
        "chronos_fit": {
            "bf16": True,
            "tf32": True,
            "full_determinism": True,
            "seed": 42,
            "data_seed": 42,
            "disable_data_parallel": True,
        },
    }


def _data() -> BAData:
    timestamps = np.arange(
        np.datetime64("2023-12-31T20:00", "h"),
        np.datetime64("2024-01-01T08:00", "h"),
        np.timedelta64(1, "h"),
    ).astype("datetime64[us]")
    target = np.arange(1, len(timestamps) + 1, dtype=np.float32)
    calendar = calendar_covariates(timestamps)
    return BAData(
        ba="PJM",
        ts_utc=timestamps,
        target=target,
        covariates={"temp_c": np.zeros(len(target), dtype=np.float32), **calendar},
        future_keys=list(calendar),
        train_end=6,
        val_end=10,
        denom_mae=1.0,
    )


def test_frozen_runtime_policy_rejects_descriptive_only_determinism() -> None:
    runtime = _runtime_identity()
    validate_reproducibility_runtime(runtime)

    runtime["system"]["deterministic_algorithms"] = False
    with pytest.raises(ValueError, match="does not enforce deterministic"):
        validate_reproducibility_runtime(runtime)


def test_validation_only_view_physically_excludes_locked_test_rows() -> None:
    source = _data()
    view = validation_only_view(source)

    assert len(view.target) == source.val_end
    assert len(view.ts_utc) == source.val_end
    assert all(len(values) == source.val_end for values in view.covariates.values())
    assert any("locked 2025 test rows excluded" in warning for warning in view.warnings)

    source.target[source.val_end :] = 1_000_000
    assert np.max(view.target) < 1_000_000


def test_selection_loader_never_materializes_2025_valid_times(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    start = datetime(2024, 12, 31, tzinfo=UTC)
    timestamps = [start + timedelta(hours=index) for index in range(48)]
    available = datetime(2025, 1, 3, tzinfo=UTC)
    store.append(
        "load_hourly",
        pl.DataFrame(
            {
                "ts_utc": timestamps,
                "ba": ["PJM"] * 48,
                "load_mw": np.arange(48, dtype=float) + 1_000,
                "as_of": [available] * 48,
            }
        ),
    )
    store.append(
        "weather_hourly",
        pl.DataFrame(
            {
                "ts_utc": timestamps,
                "ba": ["PJM"] * 48,
                "temp_c": np.arange(48, dtype=float),
                "as_of": [available] * 48,
            }
        ),
    )
    locked_start = datetime(2025, 1, 1, tzinfo=UTC)

    data = load_ba_data("PJM", valid_before=locked_start)

    assert len(data.target) == 24
    assert data.ts_utc.max() < np.datetime64("2025-01-01T00:00", "us")


def test_rolling_diagnostics_report_wis_and_ignore_test_partition() -> None:
    data = _data()

    class CalendarEchoPipeline:
        def predict_quantiles(self, tasks, **kwargs):
            del kwargs
            rows = []
            for task in tasks:
                horizon = len(task["future_covariates"]["hour_sin"])
                median = task["target"][-1] + np.arange(1, horizon + 1)
                rows.append(np.stack((median - 1, median, median + 1), axis=-1)[None])
            return rows, [row[..., 1] for row in rows]

    first = rolling_eval_c2(
        CalendarEchoPipeline(),
        {data.ba: data},
        on="val",
        context=2,
        horizon=2,
        step=2,
        max_origins=1,
        batch_size=1,
    )
    data.target[data.val_end :] = 1_000_000
    second = rolling_eval_c2(
        CalendarEchoPipeline(),
        {data.ba: data},
        on="val",
        context=2,
        horizon=2,
        step=2,
        max_origins=1,
        batch_size=1,
    )

    assert first["split"] == "val"
    assert first["mase"] == 0.0
    assert first["wis"] == pytest.approx(0.2 / 1.5)
    assert first["wis"] == second["wis"]
    assert first["per_ba"]["PJM"]["n_windows"] == 1


def test_healthy_candidate_passes_and_reports_per_rto_dispersion() -> None:
    events, state = _healthy_history()
    audit = build_overfit_audit(
        _metrics("train", mase=0.55, wis_scaled=0.45),
        _metrics("val", mase=0.70, wis_scaled=0.60),
        _metrics("val", mase=0.80, wis_scaled=0.70),
        training_events=events,
        final_state=state,
        expected_steps=300,
    )

    assert audit["promotion_eligible"] is True
    assert audit["split_contract"]["test_opened"] is False
    assert set(audit["metrics"]["per_rto"]) == set(TRUST_RTO_BAS)
    assert audit["metrics"]["validation"]["worst_rto"] == "SWPP"
    assert audit["training_behavior"]["best_eval_step"] == 200
    assert audit["metrics"]["paired_validation"]["samples"] == 2_000
    assert audit["metrics"]["paired_validation"]["block_origins"] == 7
    assert audit["metrics"]["paired_validation"]["origin_count"] == 90
    assert all(gate["passed"] for gate in audit["gates"])


def test_generalization_gap_rejects_overfit_candidate() -> None:
    events, state = _healthy_history()
    audit = build_overfit_audit(
        _metrics("train", mase=0.20, wis_scaled=0.15),
        _metrics("val", mase=0.75, wis_scaled=0.65),
        _metrics("val", mase=0.80, wis_scaled=0.70),
        training_events=events,
        final_state=state,
        expected_steps=300,
    )

    assert audit["promotion_eligible"] is False
    assert "macro_mase_generalization_ratio" in audit["rejection_reasons"]
    assert "macro_wis_generalization_ratio" in audit["rejection_reasons"]


def test_missing_rto_and_unstable_training_fail_closed() -> None:
    events, state = _healthy_history()
    missing = _metrics("val", mase=0.70, wis_scaled=0.60)
    missing["per_ba"].pop("SWPP")
    malformed = build_overfit_audit(
        _metrics("train", mase=0.55, wis_scaled=0.45),
        missing,
        _metrics("val", mase=0.80, wis_scaled=0.70),
        training_events=events,
        final_state=state,
        expected_steps=300,
    )
    assert malformed["promotion_eligible"] is False
    assert malformed["rejection_reasons"][0].startswith("metrics_contract_valid")

    unstable_events = [
        {"event": "log", "step": 100, "train_loss": 1.0, "eval_loss": 0.8},
        {"event": "log", "step": 300, "train_loss": 0.3, "eval_loss": 1.2},
    ]
    unstable_state = {
        "global_step": 300,
        "best_metric": 0.8,
        "best_model_checkpoint": "checkpoint-100",
    }
    unstable = build_overfit_audit(
        _metrics("train", mase=0.55, wis_scaled=0.45),
        _metrics("val", mase=0.70, wis_scaled=0.60),
        _metrics("val", mase=0.80, wis_scaled=0.70),
        training_events=unstable_events,
        final_state=unstable_state,
        expected_steps=300,
    )
    assert unstable["promotion_eligible"] is False
    assert "eval_loss_rebound_ratio" in unstable["rejection_reasons"]


def test_partial_diagnostic_windows_fail_closed() -> None:
    events, state = _healthy_history()
    incomplete_validation = _metrics("val", mase=0.70, wis_scaled=0.60)
    incomplete_validation["per_ba"]["PJM"]["n_points"] -= 1

    audit = build_overfit_audit(
        _metrics("train", mase=0.55, wis_scaled=0.45),
        incomplete_validation,
        _metrics("val", mase=0.80, wis_scaled=0.70),
        training_events=events,
        final_state=state,
        expected_steps=300,
    )

    assert audit["promotion_eligible"] is False
    assert audit["gates"][0]["name"] == "metrics_contract_valid"
    assert "diagnostic windows are incomplete" in audit["rejection_reasons"][0]


def test_diagnostic_metrics_without_complete_origin_policy_fail_closed() -> None:
    events, state = _healthy_history()
    validation = _metrics("val", mase=0.70, wis_scaled=0.60)
    validation["complete_target_origins_only"] = False

    audit = build_overfit_audit(
        _metrics("train", mase=0.55, wis_scaled=0.45),
        validation,
        _metrics("val", mase=0.80, wis_scaled=0.70),
        training_events=events,
        final_state=state,
        expected_steps=300,
    )

    assert audit["promotion_eligible"] is False
    assert audit["gates"][0]["name"] == "metrics_contract_valid"
    assert "unexpected complete_target_origins_only" in audit["rejection_reasons"][0]


def test_candidate_worse_than_frozen_baseline_cannot_promote() -> None:
    events, state = _healthy_history()
    audit = build_overfit_audit(
        _metrics("train", mase=0.60, wis_scaled=0.50),
        _metrics("val", mase=0.75, wis_scaled=0.65),
        _metrics("val", mase=0.70, wis_scaled=0.60),
        training_events=events,
        final_state=state,
        expected_steps=300,
    )

    assert audit["promotion_eligible"] is False
    assert "validation_mase_vs_baseline_ratio" in audit["rejection_reasons"]
    assert "validation_wis_vs_baseline_ratio" in audit["rejection_reasons"]


def test_paired_bootstrap_rejects_uncertain_baseline_noninferiority() -> None:
    events, state = _healthy_history()
    validation = _metrics("val", mase=0.80, wis_scaled=0.70)
    baseline = _metrics("val", mase=0.85, wis_scaled=0.75)
    for index, ba in enumerate(TRUST_RTO_BAS):
        center = 0.80 + index * 0.01
        values = [center - 0.70] * 45 + [center + 0.70] * 45
        for row, value in zip(
            validation["per_ba"][ba]["origin_mase"], values, strict=True
        ):
            row["mase"] = value

    audit = build_overfit_audit(
        _metrics("train", mase=0.55, wis_scaled=0.50),
        validation,
        baseline,
        training_events=events,
        final_state=state,
        expected_steps=300,
    )

    gate = next(
        gate
        for gate in audit["gates"]
        if gate["name"] == "paired_bootstrap_mase_ratio_upper_95"
    )
    assert gate["passed"] is False
    assert gate["actual"] > gate["threshold"]


def test_promotion_identity_rejects_extra_duplicate_and_unknown_inputs() -> None:
    valid = {
        "base_revision": "a" * 40,
        "code_revision": "b" * 40,
        "data_snapshot_sha256": FROZEN_DATA_SNAPSHOT_SHA256,
    }
    assert validate_promotion_inputs(list(reversed(TRUST_RTO_BAS)), **valid) == list(
        TRUST_RTO_BAS
    )

    with pytest.raises(ValueError, match="exactly one"):
        validate_promotion_inputs([*TRUST_RTO_BAS, "AEC"], **valid)
    with pytest.raises(ValueError, match="exactly one"):
        validate_promotion_inputs([*TRUST_RTO_BAS[:-1], "PJM"], **valid)
    with pytest.raises(ValueError, match="base-revision"):
        validate_promotion_inputs(
            list(TRUST_RTO_BAS),
            **{**valid, "base_revision": "unknown"},
        )
    with pytest.raises(ValueError, match="code-revision"):
        validate_promotion_inputs(
            list(TRUST_RTO_BAS),
            **{**valid, "code_revision": "unknown"},
        )
    with pytest.raises(ValueError, match="data-snapshot-sha256"):
        validate_promotion_inputs(
            list(TRUST_RTO_BAS),
            **{**valid, "data_snapshot_sha256": "unknown"},
        )


def test_remote_model_load_is_pinned_but_local_artifact_is_revision_free(
    tmp_path: Path,
) -> None:
    local = tmp_path / "v3"
    local.mkdir()

    assert revision_for_model_load(local, "a" * 40) is None
    assert revision_for_model_load("amazon/chronos-2", "a" * 40) == "a" * 40
    with pytest.raises(ValueError, match="base-revision"):
        revision_for_model_load("amazon/chronos-2", "unknown")
    with pytest.raises(ValueError, match="base-revision"):
        revision_for_model_load(local, "unknown")


def test_only_release_safe_base_lineage_can_enter_promotion(tmp_path: Path) -> None:
    validate_release_lineage(
        RELEASE_BASE_MODEL_ID,
        base_model_id=RELEASE_BASE_MODEL_ID,
        base_revision=RELEASE_BASE_REVISION,
    )

    local_v3 = tmp_path / "surge-fm-v3"
    local_v3.mkdir()
    with pytest.raises(ValueError, match="research-only"):
        validate_release_lineage(
            local_v3,
            base_model_id="autogluon/chronos-2-surge-fm-v3",
            base_revision="a" * 40,
        )
    with pytest.raises(ValueError, match="load the pinned release model ID directly"):
        validate_release_lineage(
            "custom/chronos-fork",
            base_model_id=RELEASE_BASE_MODEL_ID,
            base_revision=RELEASE_BASE_REVISION,
        )
    with pytest.raises(ValueError, match="load the pinned release model ID directly"):
        validate_release_lineage(
            local_v3,
            base_model_id=RELEASE_BASE_MODEL_ID,
            base_revision=RELEASE_BASE_REVISION,
        )


def test_locked_test_requires_a_checksummed_eligible_promotion_chain(
    tmp_path: Path,
) -> None:
    best = tmp_path / "best"
    best.mkdir()
    model_file = best / "model.safetensors"
    model_file.write_bytes(b"immutable checkpoint")
    model_sha = hashlib.sha256(model_file.read_bytes()).hexdigest()
    model_artifact_sha = artifact_sha256(best)
    versions = {
        name: "test"
        for name in (
            "chronos-forecasting",
            "transformers",
            "torch",
            "peft",
            "numpy",
            "polars",
            "holidays",
            "pyarrow",
        )
    }
    identity = {
        "base_model": "amazon/chronos-2",
        "base_revision": RELEASE_BASE_REVISION,
        "code_revision": "b" * 40,
        "data_snapshot_sha256": FROZEN_DATA_SNAPSHOT_SHA256,
        "feature_spec_version": "test-v1",
        "feature_spec_sha256": "d" * 64,
        "bas": list(TRUST_RTO_BAS),
    }
    events, state = _healthy_history()
    audit = build_overfit_audit(
        _metrics("train", mase=0.55, wis_scaled=0.45),
        _metrics("val", mase=0.70, wis_scaled=0.60),
        _metrics("val", mase=0.80, wis_scaled=0.70),
        training_events=events,
        final_state=state,
        expected_steps=300,
    )
    assert audit["promotion_eligible"] is True
    _add_frozen_schedules(audit)
    audit["environment"] = versions
    audit["identity"] = identity
    audit["runtime"] = _runtime_identity()
    audit_path = tmp_path / "surge-overfit-audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    manifest = {
        "locked_test_opened": False,
        "base_model": identity["base_model"],
        "base_model_source": identity["base_model"],
        "base_model_revision": identity["base_revision"],
        "code_revision": identity["code_revision"],
        "data_snapshot_sha256": identity["data_snapshot_sha256"],
        "bas": list(TRUST_RTO_BAS),
        "feature_spec_version": identity["feature_spec_version"],
        "feature_spec_sha256": identity["feature_spec_sha256"],
        "config": {
            "context": 2_048,
            "horizon": 24,
            "with_generation": False,
            "complete_target_origins_only": True,
            "shared_origin_schedule": True,
            "checkpoint_selection_disjoint_from_promotion": True,
        },
        "versions": versions,
        "runtime": _runtime_identity(),
        "artifact_files": [
            {
                "path": model_file.name,
                "bytes": model_file.stat().st_size,
                "sha256": model_sha,
            }
        ],
        "model_artifact_hash_algorithm": "sha256-tree-v1",
        "model_artifact_sha256": model_artifact_sha,
        "selection": {
            "policy_version": POLICY_VERSION,
            "promotion_eligible": True,
            "overfit_audit": audit_path.name,
            "overfit_audit_sha256": audit_sha,
            "checkpoint_state": "promoted",
            "diagnostics_error": None,
        },
    }
    manifest_path = tmp_path / "surge-training-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    marker = {
        "policy_version": POLICY_VERSION,
        "promotion_eligible": True,
        "test_opened": False,
        "checkpoint": "best",
        "manifest": manifest_path.name,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "overfit_audit": audit_path.name,
        "overfit_audit_sha256": audit_sha,
        "model_artifact_hash_algorithm": "sha256-tree-v1",
        "model_artifact_sha256": model_artifact_sha,
    }
    marker_path = tmp_path / "surge-promotion.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    verified = verify_promotion_artifact(marker_path, model_path=best)
    assert verified["promotion_eligible"] is True
    assert verified["training_identity"]["bas"] == list(TRUST_RTO_BAS)
    assert verified["model_artifact_sha256"] == model_artifact_sha

    tampered_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    tampered_audit["metrics"]["paired_validation"]["mase_ratio_ci_high_95"] += 0.01
    audit_path.write_text(json.dumps(tampered_audit), encoding="utf-8")
    tampered_audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    manifest["selection"]["overfit_audit_sha256"] = tampered_audit_sha
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    marker["overfit_audit_sha256"] = tampered_audit_sha
    marker["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="does not recompute"):
        verify_promotion_artifact(marker_path, model_path=best)

    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    manifest["selection"]["overfit_audit_sha256"] = audit_sha
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    marker["overfit_audit_sha256"] = audit_sha
    marker["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    model_file.write_bytes(b"tampered checkpoint")
    with pytest.raises(ValueError, match=r"byte count|checksum"):
        verify_promotion_artifact(marker_path, model_path=best)
    model_file.write_bytes(b"immutable checkpoint")

    manifest["model_artifact_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    marker["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    marker["model_artifact_sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="model artifact SHA-256 changed"):
        verify_promotion_artifact(marker_path, model_path=best)
    manifest["model_artifact_sha256"] = model_artifact_sha
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    marker["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    marker["model_artifact_sha256"] = model_artifact_sha
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    manifest["base_model_source"] = "/tmp/custom-local-base"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    marker["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="load the pinned release model ID directly"):
        verify_promotion_artifact(marker_path, model_path=best)
    manifest["base_model_source"] = RELEASE_BASE_MODEL_ID
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    marker["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    audit_path.write_text(json.dumps({**audit, "promotion_eligible": False}))
    with pytest.raises(ValueError, match="checksum"):
        verify_promotion_artifact(marker_path, model_path=best)


def _write_selection_candidate(
    root: Path, *, num_steps: int, validation_mase: float
) -> tuple[Path, dict, dict]:
    root.mkdir()
    best = root / "best"
    best.mkdir()
    model_file = best / "model.safetensors"
    model_file.write_bytes(f"candidate-{num_steps}".encode())
    model_sha = hashlib.sha256(model_file.read_bytes()).hexdigest()
    model_artifact_sha = artifact_sha256(best)
    versions = {
        name: "test"
        for name in (
            "chronos-forecasting",
            "transformers",
            "torch",
            "peft",
            "numpy",
            "polars",
            "holidays",
            "pyarrow",
        )
    }
    identity = {
        "base_model": RELEASE_BASE_MODEL_ID,
        "base_revision": RELEASE_BASE_REVISION,
        "code_revision": "b" * 40,
        "data_snapshot_sha256": FROZEN_DATA_SNAPSHOT_SHA256,
        "feature_spec_version": "test-v1",
        "feature_spec_sha256": "d" * 64,
        "bas": list(TRUST_RTO_BAS),
    }
    first_step = 100
    events = [
        {"event": "log", "step": first_step, "train_loss": 1.0},
        {"event": "log", "step": first_step, "eval_loss": 0.90},
        {"event": "checkpoint", "step": first_step},
        {"event": "log", "step": num_steps, "train_loss": 0.7},
        {"event": "log", "step": num_steps, "eval_loss": 0.85},
        {"event": "checkpoint", "step": num_steps},
    ]
    state = {
        "global_step": num_steps,
        "best_metric": 0.85,
        "best_model_checkpoint": f"checkpoint-{num_steps}",
    }
    audit = build_overfit_audit(
        _metrics("train", mase=0.55, wis_scaled=0.45),
        _metrics("val", mase=validation_mase, wis_scaled=validation_mase - 0.1),
        _metrics("val", mase=0.85, wis_scaled=0.75),
        training_events=events,
        final_state=state,
        expected_steps=num_steps,
    )
    assert audit["promotion_eligible"] is True
    _add_frozen_schedules(audit)
    audit["environment"] = versions
    audit["identity"] = identity
    audit["runtime"] = _runtime_identity()
    audit_path = root / "surge-overfit-audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    config = {
        "context": 2_048,
        "horizon": 24,
        "mode": "lora",
        "learning_rate": 1e-5,
        "num_steps": num_steps,
        "batch_size": 32,
        "diagnostic_origins": 90,
        "diagnostic_step": 24,
        "diagnostic_batch_size": 16,
        "complete_target_origins_only": True,
        "shared_origin_schedule": True,
        "checkpoint_selection_disjoint_from_promotion": True,
        "seed": 42,
        "with_generation": False,
    }
    manifest = {
        "locked_test_opened": False,
        "base_model": identity["base_model"],
        "base_model_source": identity["base_model"],
        "base_model_revision": identity["base_revision"],
        "code_revision": identity["code_revision"],
        "data_snapshot_sha256": identity["data_snapshot_sha256"],
        "bas": identity["bas"],
        "feature_spec_version": identity["feature_spec_version"],
        "feature_spec_sha256": identity["feature_spec_sha256"],
        "config": config,
        "versions": versions,
        "runtime": _runtime_identity(),
        "artifact_files": [
            {
                "path": model_file.name,
                "bytes": model_file.stat().st_size,
                "sha256": model_sha,
            }
        ],
        "model_artifact_hash_algorithm": "sha256-tree-v1",
        "model_artifact_sha256": model_artifact_sha,
        "selection": {
            "policy_version": POLICY_VERSION,
            "promotion_eligible": True,
            "overfit_audit": audit_path.name,
            "overfit_audit_sha256": audit_sha,
            "checkpoint_state": "promoted",
            "diagnostics_error": None,
        },
    }
    manifest_path = root / "surge-training-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    marker = {
        "policy_version": POLICY_VERSION,
        "promotion_eligible": True,
        "test_opened": False,
        "checkpoint": "best",
        "manifest": manifest_path.name,
        "manifest_sha256": manifest_sha,
        "overfit_audit": audit_path.name,
        "overfit_audit_sha256": audit_sha,
        "model_artifact_hash_algorithm": "sha256-tree-v1",
        "model_artifact_sha256": model_artifact_sha,
    }
    marker_path = root / "surge-promotion.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    marker_sha = hashlib.sha256(marker_path.read_bytes()).hexdigest()
    generalization = audit["metrics"]["generalization"]
    record = {
        "candidate": root.name,
        "num_steps": num_steps,
        "promotion_eligible": True,
        "selection_score": 0.5
        * (
            generalization["macro_mase_vs_baseline_ratio"]
            + generalization["macro_wis_scaled_vs_baseline_ratio"]
        ),
        "validation_mase_vs_base_ratio": generalization[
            "macro_mase_vs_baseline_ratio"
        ],
        "validation_scaled_wis_vs_base_ratio": generalization[
            "macro_wis_scaled_vs_baseline_ratio"
        ],
        "rejection_reasons": [],
        "manifest_sha256": manifest_sha,
        "overfit_audit_sha256": audit_sha,
        "promotion_marker_sha256": marker_sha,
    }
    return marker_path, record, {
        **identity,
        "versions": versions,
        "runtime": _runtime_identity(),
    }


def _reject_selection_candidate(marker_path: Path, record: dict) -> dict:
    root = marker_path.parent
    (root / "best").rename(root / "candidate-unpromoted")
    audit_path = root / "surge-overfit-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    failed_gate = audit["gates"][0]["name"]
    audit["gates"][0]["passed"] = False
    audit["promotion_eligible"] = False
    audit["rejection_reasons"] = [failed_gate]
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()

    manifest_path = root / "surge-training-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selection"].update(
        {
            "promotion_eligible": False,
            "overfit_audit_sha256": audit_sha,
            "checkpoint_state": "candidate-rejected",
            "diagnostics_error": None,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    marker_path.unlink()
    return {
        **record,
        "promotion_eligible": False,
        "selection_score": None,
        "rejection_reasons": [failed_gate],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "overfit_audit_sha256": audit_sha,
        "promotion_marker_sha256": None,
    }


def test_frozen_selection_verifier_recomputes_winner_and_binds_marker(
    tmp_path: Path,
) -> None:
    marker_1000, record_1000, identity = _write_selection_candidate(
        tmp_path / "official-lora-1000",
        num_steps=1_000,
        validation_mase=0.70,
    )
    _marker_2000, record_2000, _ = _write_selection_candidate(
        tmp_path / "official-lora-2000",
        num_steps=2_000,
        validation_mase=0.72,
    )
    assert record_1000["selection_score"] < record_2000["selection_score"]
    selection = {
        "policy_version": "surge-v0.2-frozen-h100-selection-v1",
        "training_identity": identity,
        "candidates": [record_1000, record_2000],
        "winner": record_1000,
        "locked_test_opened": False,
    }
    selection_path = tmp_path / "v0.2-h100-selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    promotion = verify_promotion_artifact(
        marker_1000,
        model_path=marker_1000.parent / "best",
    )

    verified = verify_selection_artifact(
        selection_path,
        promotion_path=marker_1000,
        promotion=promotion,
    )
    assert verified["winner"]["candidate"] == "official-lora-1000"

    timestamped = json.loads(selection_path.read_text(encoding="utf-8"))
    timestamped["created_at_utc"] = "2099-01-01T00:00:00+00:00"
    selection_path.write_text(json.dumps(timestamped), encoding="utf-8")
    timestamped_verified = verify_selection_artifact(
        selection_path,
        promotion_path=marker_1000,
        promotion=promotion,
    )
    assert timestamped_verified["selection_sha256"] != verified["selection_sha256"]
    assert (
        timestamped_verified["selection_decision_sha256"]
        == verified["selection_decision_sha256"]
    )
    assert (
        timestamped_verified["experiment_protocol_sha256"]
        == verified["experiment_protocol_sha256"]
    )

    forged = json.loads(selection_path.read_text(encoding="utf-8"))
    forged["candidates"][0]["selection_score"] = 0.001
    forged["winner"] = forged["candidates"][0]
    selection_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="record disagrees"):
        verify_selection_artifact(
            selection_path,
            promotion_path=marker_1000,
            promotion=promotion,
        )


def test_frozen_selection_verifies_rejected_candidate_failure_inventory(
    tmp_path: Path,
) -> None:
    marker_1000, record_1000, identity = _write_selection_candidate(
        tmp_path / "official-lora-1000",
        num_steps=1_000,
        validation_mase=0.70,
    )
    marker_2000, record_2000, _ = _write_selection_candidate(
        tmp_path / "official-lora-2000",
        num_steps=2_000,
        validation_mase=0.72,
    )
    rejected_2000 = _reject_selection_candidate(marker_2000, record_2000)
    selection = {
        "policy_version": "surge-v0.2-frozen-h100-selection-v1",
        "training_identity": identity,
        "candidates": [record_1000, rejected_2000],
        "winner": record_1000,
        "locked_test_opened": False,
    }
    selection_path = tmp_path / "v0.2-h100-selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    promotion = verify_promotion_artifact(
        marker_1000,
        model_path=marker_1000.parent / "best",
    )

    verify_selection_artifact(
        selection_path,
        promotion_path=marker_1000,
        promotion=promotion,
    )

    rejected_model = (
        marker_2000.parent / "candidate-unpromoted" / "model.safetensors"
    )
    rejected_model.unlink()
    with pytest.raises(ValueError, match="missing regular file"):
        verify_selection_artifact(
            selection_path,
            promotion_path=marker_1000,
            promotion=promotion,
        )


def test_locked_test_receipt_is_atomic_one_shot_and_records_result(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "surge-promotion.json"
    marker_path.write_text("{}", encoding="utf-8")
    registry = tmp_path / "authoritative-registry"
    identity = {
        "base_revision": "a" * 40,
        "code_revision": "b" * 40,
        "data_snapshot_sha256": "c" * 64,
        "bas": list(TRUST_RTO_BAS),
    }

    receipt_path = reserve_locked_test_run(
        tmp_path / "v0.2-h100-selection.json",
        experiment="v0.2-locked-test",
        training_identity=identity,
        selection_sha256="f" * 64,
        selection_decision_sha256="a" * 64,
        experiment_protocol_sha256="b" * 64,
        promotion_path=marker_path,
        marker_sha256="d" * 64,
        checkpoint_inventory_sha256="e" * 64,
        model_artifact_sha256="c" * 64,
        registry_root=registry,
    )
    started = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert started["status"] == "started"
    assert started["test_opened"] is True
    assert started["model_artifact_hash_algorithm"] == "sha256-tree-v1"
    assert started["model_artifact_sha256"] == "c" * 64

    with pytest.raises(RuntimeError, match="already consumed"):
        reserve_locked_test_run(
            tmp_path / "v0.2-h100-selection.json",
            experiment="forbidden-second-look",
            training_identity=identity,
            selection_sha256="f" * 64,
            selection_decision_sha256="a" * 64,
            experiment_protocol_sha256="b" * 64,
            promotion_path=marker_path,
            marker_sha256="d" * 64,
            checkpoint_inventory_sha256="e" * 64,
            model_artifact_sha256="c" * 64,
            registry_root=registry,
        )

    copied_selection_dir = tmp_path / "copied-selection"
    copied_selection_dir.mkdir()
    with pytest.raises(RuntimeError, match="already consumed"):
        reserve_locked_test_run(
            copied_selection_dir / "v0.2-h100-selection.json",
            experiment="forbidden-copied-look",
            training_identity=identity,
            selection_sha256="f" * 64,
            selection_decision_sha256="a" * 64,
            experiment_protocol_sha256="b" * 64,
            promotion_path=marker_path,
            marker_sha256="d" * 64,
            checkpoint_inventory_sha256="e" * 64,
            model_artifact_sha256="c" * 64,
            registry_root=registry,
        )

    with pytest.raises(ValueError, match="Out of range float"):
        complete_locked_test_run(receipt_path, {"mase": float("nan")})
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == "started"

    complete_locked_test_run(receipt_path, {"mase": 0.73})
    completed = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["result"] == {"mase": 0.73}
    assert len(completed["result_sha256"]) == 64
    registry_receipt = json.loads((registry / f"{'b' * 64}.json").read_text())
    assert registry_receipt["status"] == "completed"


def test_locked_test_failure_omits_message_and_is_terminally_mirrored(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "surge-promotion.json"
    marker_path.write_text("{}", encoding="utf-8")
    registry = tmp_path / "authoritative-registry"
    identity = {
        "base_revision": "a" * 40,
        "code_revision": "b" * 40,
        "data_snapshot_sha256": "c" * 64,
        "bas": list(TRUST_RTO_BAS),
    }
    receipt_path = reserve_locked_test_run(
        tmp_path / "v0.2-h100-selection.json",
        experiment="v0.2-locked-test-failure",
        training_identity=identity,
        selection_sha256="f" * 64,
        selection_decision_sha256="a" * 64,
        experiment_protocol_sha256="b" * 64,
        promotion_path=marker_path,
        marker_sha256="d" * 64,
        checkpoint_inventory_sha256="e" * 64,
        model_artifact_sha256="c" * 64,
        registry_root=registry,
    )

    failure = ValueError(
        'Authorization: Basic dXNlcjpwYXNz {"token":"do-not-store"} '
        "postgresql://alice:p4ssword@example.test/db "
        "AWS_SECRET_ACCESS_KEY=do-not-store-either"
    )
    fail_locked_test_run(receipt_path, failure)

    failed = json.loads(receipt_path.read_text(encoding="utf-8"))
    registry_failed = json.loads((registry / f"{'b' * 64}.json").read_text())
    assert failed == registry_failed
    assert failed["status"] == "failed"
    assert failed["test_opened"] is True
    assert datetime.fromisoformat(failed["failed_at_utc"]).tzinfo == UTC
    assert failed["failure"] == {
        "exception_type": "ValueError",
        "message_omitted": True,
    }
    assert "do-not-store" not in json.dumps(failed)
    assert "dXNlcjpwYXNz" not in json.dumps(failed)
    assert "p4ssword" not in json.dumps(failed)
    encoded_failure = json.dumps(
        failed["failure"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert failed["failure_sha256"] == hashlib.sha256(encoded_failure).hexdigest()
    assert "result" not in failed

    with pytest.raises(ValueError, match="not in the started state"):
        complete_locked_test_run(receipt_path, {"mase": 0.73})
    with pytest.raises(ValueError, match="not in the started state"):
        fail_locked_test_run(receipt_path, RuntimeError("replacement failure"))
    with pytest.raises(RuntimeError, match="already consumed"):
        reserve_locked_test_run(
            tmp_path / "v0.2-h100-selection.json",
            experiment="forbidden-second-look-after-failure",
            training_identity=identity,
            selection_sha256="f" * 64,
            selection_decision_sha256="a" * 64,
            experiment_protocol_sha256="b" * 64,
            promotion_path=marker_path,
            marker_sha256="d" * 64,
            checkpoint_inventory_sha256="e" * 64,
            model_artifact_sha256="c" * 64,
            registry_root=registry,
        )


def test_locked_test_terminalization_reconciles_transient_receipt_write_failure(
    monkeypatch, tmp_path: Path
) -> None:
    marker_path = tmp_path / "surge-promotion.json"
    marker_path.write_text("{}", encoding="utf-8")
    registry = tmp_path / "authoritative-registry"
    protocol_sha256 = "b" * 64
    receipt_path = reserve_locked_test_run(
        tmp_path / "v0.2-h100-selection.json",
        experiment="v0.2-transient-receipt-write-failure",
        training_identity={"bas": list(TRUST_RTO_BAS)},
        selection_sha256="f" * 64,
        selection_decision_sha256="a" * 64,
        experiment_protocol_sha256=protocol_sha256,
        promotion_path=marker_path,
        marker_sha256="d" * 64,
        checkpoint_inventory_sha256="e" * 64,
        model_artifact_sha256="c" * 64,
        registry_root=registry,
    )
    real_replace = overfit_module._replace_json_atomically
    calls = 0

    def fail_second_replace_once(path: Path, value: dict) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected receipt replace failure")
        real_replace(path, value)

    monkeypatch.setattr(
        overfit_module,
        "_replace_json_atomically",
        fail_second_replace_once,
    )

    fail_locked_test_run(receipt_path, ValueError("sensitive text omitted"))

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    authoritative = json.loads((registry / f"{protocol_sha256}.json").read_text())
    assert calls == 3
    assert receipt == authoritative
    assert receipt["status"] == "failed"


def test_diagnostic_exception_artifact_is_explicitly_rejected() -> None:
    audit = rejected_overfit_audit("GPU evaluation failed", expected_steps=2_000)

    assert audit["promotion_eligible"] is False
    assert audit["split_contract"]["test_opened"] is False
    assert audit["rejection_reasons"] == [
        "diagnostics_completed: GPU evaluation failed"
    ]
