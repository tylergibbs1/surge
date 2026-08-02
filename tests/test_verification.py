from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from surge import ledger, store, verification


def _commit_forecast() -> ledger.ForecastRecord:
    record = ledger.ForecastRecord(
        ba="PJM",
        scheduled_for_utc=datetime(2026, 1, 1, 6, tzinfo=UTC),
        feature_cutoff_utc=datetime(2026, 1, 1, 6, 15, tzinfo=UTC),
        issued_at_utc=datetime(2026, 1, 1, 6, 15, tzinfo=UTC),
        context_start_utc=datetime(2025, 10, 8, 22, tzinfo=UTC),
        context_end_utc=datetime(2026, 1, 1, 5, tzinfo=UTC),
        model_name="surge-fm-v4-nopeek",
        model_revision="model-sha",
        code_revision="code-sha",
        feature_spec_version="load-v2-core",
        feature_spec_sha256="spec-sha",
        feature_snapshot_sha256="snapshot-sha",
        availability_mode=ledger.AvailabilityMode.EXACT_VINTAGE,
        mase_scale_24=200.0,
        points=(
            ledger.ForecastPointRecord(
                valid_at_utc=datetime(2026, 1, 1, 7, tzinfo=UTC),
                mean_mw=1_120.0,
                p10_mw=900.0,
                p50_mw=1_100.0,
                p90_mw=1_300.0,
            ),
        ),
    )
    return ledger.commit_forecast(record)


def _append_outcomes() -> None:
    valid_at = datetime(2026, 1, 1, 7, tzinfo=UTC)
    store.append(
        "load_hourly",
        pl.DataFrame({
            "ts_utc": [valid_at, valid_at],
            "ba": ["PJM", "PJM"],
            "load_mw": [1_000.0, 1_500.0],
            "source": ["eia-930", "eia-930"],
            "as_of": [
                datetime(2026, 1, 2, tzinfo=UTC),
                datetime(2026, 1, 5, tzinfo=UTC),
            ],
        }),
    )


def test_verification_pins_revision_available_at_maturity(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    record = _commit_forecast()
    _append_outcomes()

    committed = verification.verify_forecast(
        record.issuance_id,
        verified_at_utc=datetime(2026, 1, 5, 12, tzinfo=UTC),
    )
    assert committed == 1
    # A later run is a no-op and cannot rewrite the pinned observation.
    assert verification.verify_forecast(
        record.issuance_id,
        verified_at_utc=datetime(2026, 1, 8, tzinfo=UTC),
    ) == 0

    rows = verification.verification_rows(record.issuance_id)
    assert rows["actual_mw"][0] == 1_000.0
    assert rows["outcome_as_of_utc"][0] == datetime(2026, 1, 2, tzinfo=UTC)

    score = verification.score_forecast(record.issuance_id)
    assert score.mae_mw == 100.0
    assert score.bias_mw == 100.0
    assert score.wape_pct == 10.0
    assert score.mase_24 == 0.5
    assert score.pi80_coverage_pct == 100.0
    assert score.pi80_calibration_error_pct == 20.0


def test_verification_refuses_immature_forecast(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    record = _commit_forecast()

    with pytest.raises(ValueError, match="not fully mature"):
        verification.verify_forecast(
            record.issuance_id,
            verified_at_utc=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_maturity_derives_policy_identity_and_scores_only_current_policy(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    record = _commit_forecast()
    _append_outcomes()
    verified_at = datetime(2026, 1, 5, 12, tzinfo=UTC)

    assert verification.verify_forecast(
        record.issuance_id,
        verified_at_utc=verified_at,
        maturity_hours=24,
    ) == 1
    policy_24h = verification.outcome_policy_version(24)
    rows_24h = verification.verification_rows(
        record.issuance_id,
        policy_version=policy_24h,
    )
    assert rows_24h.height == 1
    assert rows_24h["outcome_policy_version"][0] == "eia-latest-at-plus24h-v1"

    assert verification.verify_forecast(
        record.issuance_id,
        verified_at_utc=verified_at,
    ) == 1
    current_rows = verification.verification_rows(
        record.issuance_id,
        policy_version=verification.OUTCOME_POLICY_VERSION,
    )
    assert current_rows.height == 1
    assert current_rows["verification_id"][0] != rows_24h["verification_id"][0]
    assert verification.verification_rows(record.issuance_id).height == 2

    # Scoring has an explicit current-policy boundary and cannot double-count
    # an alternate settlement of the same valid timestamp.
    score = verification.score_forecast(record.issuance_id)
    assert score.n_points == 1
    assert score.mae_mw == 100.0

    with pytest.raises(ValueError, match="does not match maturity_hours"):
        verification.verify_forecast(
            record.issuance_id,
            verified_at_utc=verified_at,
            maturity_hours=24,
            policy_version=verification.OUTCOME_POLICY_VERSION,
        )


def test_metric_formulas() -> None:
    assert verification.pinball_loss(100.0, 90.0, 0.1) == 1.0
    assert verification.pinball_loss(80.0, 90.0, 0.1) == 9.0
    assert verification.interval_score_80(100.0, 90.0, 110.0) == 20.0
    assert verification.weighted_interval_score(100.0, 90.0, 100.0, 110.0) == pytest.approx(4 / 3)
