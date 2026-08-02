from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from scripts import audit_ledger
from surge import ledger, store
from surge.api import ledger_api


def _record(
    *,
    ba: str = "PJM",
    p50: float = 1_100.0,
    scheduled: datetime | None = None,
    horizon: int = 2,
    code_revision: str = "code-sha",
) -> ledger.ForecastRecord:
    scheduled = scheduled or datetime(2026, 1, 1, 6, 0, tzinfo=UTC)
    issued = scheduled + timedelta(minutes=15)
    first = scheduled + timedelta(hours=1)
    points = tuple(
        ledger.ForecastPointRecord(
            valid_at_utc=first + timedelta(hours=index),
            mean_mw=p50 + 5.0,
            p10_mw=p50 - 100.0,
            p50_mw=p50,
            p90_mw=p50 + 100.0,
        )
        for index in range(horizon)
    )
    return ledger.ForecastRecord(
        ba=ba,
        scheduled_for_utc=scheduled,
        feature_cutoff_utc=issued,
        issued_at_utc=issued,
        context_start_utc=scheduled - timedelta(hours=2_048),
        context_end_utc=scheduled - timedelta(hours=1),
        model_name="surge-fm-v4-nopeek",
        model_revision="model-sha",
        code_revision=code_revision,
        feature_spec_version="load-v2-core",
        feature_spec_sha256="spec-sha",
        feature_snapshot_sha256=f"snapshot-{ba.lower()}",
        availability_mode=ledger.AvailabilityMode.EXACT_VINTAGE,
        mase_scale_24=200.0,
        points=points,
    )


def _commit_complete_run(
    *, scheduled: datetime | None = None
) -> list[ledger.ForecastRecord]:
    records = [_record(ba=ba, scheduled=scheduled) for ba in ledger.REQUIRED_RTO_BAS]
    for record in records:
        ledger.commit_forecast(record)
    return records


def test_commit_round_trip_and_audit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    record = _record()

    ledger.commit_forecast(record)
    ledger.commit_forecast(record)  # exact retry
    restored = ledger.get_forecast(record.issuance_id)

    assert restored == record
    assert (
        ledger.latest_forecast("PJM", include_unpublished=True).issuance_id
        == record.issuance_id
    )
    assert ledger.list_forecasts(ba="PJM") == []
    with pytest.raises(KeyError):
        ledger.get_run(record.run_id)
    assert ledger.audit_forecast(record.issuance_id) == []


def test_audit_script_includes_unpublished_staged_issuances(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    record = ledger.commit_forecast(_record())
    assert ledger.list_forecasts() == []

    monkeypatch.setattr(audit_ledger.sys, "argv", ["audit_ledger.py"])
    audit_ledger.main()

    assert capsys.readouterr().out == "audited=1 failures=0\n"
    assert ledger.get_forecast(record.issuance_id) == record


def test_same_identity_with_changed_points_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    ledger.commit_forecast(_record())

    with pytest.raises(store.ImmutableConflictError, match=r"different (content|issuance)"):
        ledger.commit_forecast(_record(p50=1_200.0))


def test_seventh_rto_atomically_publishes_one_immutable_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    records = [_record(ba=ba) for ba in ledger.REQUIRED_RTO_BAS]

    for record in records[:-1]:
        ledger.commit_forecast(record)
        assert ledger.list_forecasts() == []
        with pytest.raises(KeyError):
            ledger.get_run(record.run_id)

    ledger.commit_forecast(records[-1])
    marker = ledger.get_run(records[-1].run_id)

    assert marker.required_bas == ledger.REQUIRED_RTO_BAS
    assert marker.quantiles == ledger.FORECAST_QUANTILES
    assert marker.issuance_ids == {record.ba: record.issuance_id for record in records}
    assert marker.feature_snapshot_sha256s == {
        record.ba: record.feature_snapshot_sha256 for record in records
    }
    assert {record.issuance_id for record in ledger.list_forecasts()} == {
        record.issuance_id for record in records
    }
    assert [record.ba for record in ledger.forecasts_for_run(marker.run_id)] == list(
        ledger.REQUIRED_RTO_BAS
    )

    ledger.commit_forecast(records[-1])
    ledger.publish_run_if_complete(marker.run_id)
    assert store.scan("forecast_runs").collect().height == 1


def test_mixed_run_provenance_is_rejected_before_staging(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    ledger.commit_forecast(_record(ba="PJM"))
    incompatible = _record(ba="CISO", code_revision="different-code")

    with pytest.raises(store.ImmutableConflictError, match="mixed code_revision"):
        ledger.commit_forecast(incompatible)

    with pytest.raises(KeyError):
        ledger.get_forecast(incompatible.issuance_id)
    assert ledger.list_forecasts() == []


def test_concurrent_seventh_run_writes_cannot_miss_publication(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    records = [_record(ba=ba) for ba in ledger.REQUIRED_RTO_BAS]

    with ThreadPoolExecutor(max_workers=len(records)) as pool:
        assert list(pool.map(ledger.commit_forecast, records)) == records

    marker = ledger.get_run(records[0].run_id)
    assert set(marker.issuance_ids) == set(ledger.REQUIRED_RTO_BAS)
    assert len(ledger.list_forecasts()) == len(ledger.REQUIRED_RTO_BAS)
    assert store.scan("forecast_runs").collect().height == 1


def test_newer_partial_run_cannot_create_mixed_scoreboard(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    complete = _commit_complete_run()
    complete_run_id = complete[0].run_id
    partial = ledger.commit_forecast(
        _record(
            ba="PJM",
            scheduled=datetime(2026, 1, 2, 6, 0, tzinfo=UTC),
        )
    )

    assert ledger.get_forecast(partial.issuance_id) == partial
    assert ledger.latest_forecast("PJM").run_id == complete_run_id
    assert len(ledger.list_forecasts()) == len(ledger.REQUIRED_RTO_BAS)

    board = ledger_api.build_scoreboard(
        list(ledger.REQUIRED_RTO_BAS),
        now=datetime(2026, 1, 1, 6, 30, tzinfo=UTC),
    )
    assert board.available_regions == len(ledger.REQUIRED_RTO_BAS)
    assert {row.forecast.run_id for row in board.regions if row.forecast} == {
        complete_run_id
    }


def test_live_forecast_must_precede_first_valid_hour() -> None:
    base = _record().model_dump()
    base["issued_at_utc"] = datetime(2026, 1, 1, 7, 0, tzinfo=UTC)
    base["feature_cutoff_utc"] = base["issued_at_utc"]
    with pytest.raises(ValueError, match="before their first valid time"):
        ledger.ForecastRecord(**base)


def test_v02_ledger_rejects_non_hourly_frequency() -> None:
    base = _record().model_dump()
    base["frequency_minutes"] = 15

    with pytest.raises(ValueError, match="hourly frequency"):
        ledger.ForecastRecord(**base)


def test_crossing_quantiles_are_rejected() -> None:
    with pytest.raises(ValueError, match="p10 <= p50 <= p90"):
        ledger.ForecastPointRecord(
            valid_at_utc=datetime(2026, 1, 1, 7, tzinfo=UTC),
            mean_mw=100.0,
            p10_mw=120.0,
            p50_mw=110.0,
            p90_mw=130.0,
        )


def test_orphan_points_are_not_visible(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    record = _record()
    store.write_immutable(
        "forecast_points",
        record.issuance_id,
        ledger._points_frame(record),
        partitions={
            "ledger_mode": record.mode.value,
            "issued_date": record.issued_at_utc.date().isoformat(),
        },
    )

    with pytest.raises(KeyError):
        ledger.points_for(record.issuance_id)


def test_public_adapter_exposes_provenance_and_honest_freshness(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    record = ledger.commit_forecast(_record())

    response = ledger_api.response_from_record(
        record,
        now=datetime(2026, 1, 1, 6, 30, tzinfo=UTC),
    )
    assert response.issuance_id == record.issuance_id
    assert response.feature_spec_version == "load-v2-core"
    assert response.points[0].median_mw == 1_100.0
    assert response.quality.status == "fresh"

    stale, reasons = ledger_api.forecast_state(
        record,
        now=datetime(2026, 1, 2, 12, tzinfo=UTC),
    )
    assert stale == "stale"
    assert reasons


def test_scoreboard_never_turns_missing_regions_into_zero(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    ledger.commit_forecast(_record())

    board = ledger_api.build_scoreboard(
        ["PJM", "CISO"],
        now=datetime(2026, 1, 1, 6, 30, tzinfo=UTC),
    )

    assert board.available_regions == 0
    assert board.regions[0].state == "unavailable"
    assert board.regions[0].forecast is None
    assert board.regions[1].state == "unavailable"
    assert board.regions[1].forecast is None
