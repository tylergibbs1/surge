from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient

from surge import ledger
from surge.api import forecaster
from surge.api.main import app


def _fake_forecast(
    _pipe: Any,
    ba: str,
    horizon: int = 24,
    *,
    issued_at_utc: datetime,
    feature_cutoff_utc: datetime,
    **_kwargs: Any,
) -> dict[str, Any]:
    start = issued_at_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    points = [
        {
            "ts_utc": start + timedelta(hours=index),
            "valid_at_utc": start + timedelta(hours=index),
            "mean_mw": 1_105.0,
            "p10_mw": 900.0,
            "p50_mw": 1_100.0,
            "median_mw": 1_100.0,
            "p90_mw": 1_300.0,
            "future_temp_c": None,
            "future_temp_vintage_id": None,
        }
        for index in range(horizon)
    ]
    return {
        "points": points,
        "issued_at_utc": issued_at_utc,
        "feature_cutoff_utc": feature_cutoff_utc,
        "context_start_utc": start - timedelta(hours=2_049),
        "context_end_utc": start - timedelta(hours=2),
        "feature_spec_version": "load-v2-core",
        "feature_spec_sha256": forecaster.LOAD_V2_CORE.sha256,
        "feature_snapshot_sha256": "snapshot-sha",
        "availability_mode": "exact_vintage",
        "point_estimate_kind": "median",
        "mase_scale_24": 200.0,
        "warnings": [],
        "provenance": {
            "model_revision": forecaster.MODEL_REVISION,
            "code_revision": forecaster.CODE_REVISION,
            "feature_snapshot_sha256": "snapshot-sha",
        },
        "ba": ba,
    }


def _client(monkeypatch, tmp_path) -> TestClient:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(forecaster, "CODE_REVISION", "test-code-sha")
    app.state.pipe = object()
    app.state.model_name = "chronos-2"
    return TestClient(app)


def test_live_is_distinct_from_readiness(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    app.state.pipe = None

    assert client.get("/live").status_code == 200
    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["status"] == "loading"


def test_ready_rejects_stale_data(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(
        forecaster,
        "data_watermarks_utc",
        lambda: {
            "load_hourly": datetime.now(tz=UTC) - timedelta(hours=1),
            "weather_hourly": datetime.now(tz=UTC) - timedelta(hours=13),
        },
    )

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "error"
    assert "weather_hourly" in response.json()["reasons"][0]
    assert "critically stale" in response.json()["reasons"][0]


def test_authenticated_commit_is_queryable_and_retry_is_idempotent(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    monkeypatch.setenv("SURGE_LEDGER_KEY", "test-ledger-key")
    calls = 0

    def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _fake_forecast(*args, **kwargs)

    monkeypatch.setattr(forecaster, "forecast_ba", fake)
    url = "/forecast/PJM?horizon=2&commit=true&scheduled_for_utc=2026-08-01T06:15:00Z"
    headers = {"x-surge-ledger-key": "test-ledger-key"}

    first = client.get(url, headers=headers)
    retry = client.get(url, headers=headers)

    assert first.status_code == 200
    assert retry.status_code == 200
    assert calls == 1
    payload = first.json()
    assert payload["issuance_id"] == retry.json()["issuance_id"]
    assert payload["data_cutoff_utc"]
    assert payload["feature_spec_version"] == "load-v2-core"
    assert payload["quality"]["status"] == "fresh"
    assert payload["committed"] is True
    assert payload["run_published"] is False
    assert payload["published_at_utc"] is None
    assert first.headers["cache-control"] == "no-store"
    assert first.headers["x-surge-issuance-id"] == payload["issuance_id"]

    fetched = client.get(f"/ledger/issuances/{payload['issuance_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["points"][0]["mean_mw"] == 1_105.0
    assert fetched.json()["committed"] is True
    assert fetched.json()["run_published"] is False

    listing = client.get("/ledger/issuances?ba=PJM")
    assert listing.status_code == 200
    assert listing.json()["count"] == 0


def test_ephemeral_forecast_is_not_presented_as_a_ledger_issuance(
    monkeypatch, tmp_path
) -> None:
    client = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(forecaster, "forecast_ba", _fake_forecast)

    response = client.get("/forecast/PJM?horizon=2")

    assert response.status_code == 200
    assert response.json()["committed"] is False
    assert response.json()["run_published"] is False
    assert response.json()["published_at_utc"] is None
    assert "x-surge-issuance-id" not in response.headers


def test_complete_run_publish_barrier_is_authenticated_and_idempotent(
    monkeypatch, tmp_path
) -> None:
    client = _client(monkeypatch, tmp_path)
    monkeypatch.setenv("SURGE_LEDGER_KEY", "test-ledger-key")
    monkeypatch.setattr(forecaster, "forecast_ba", _fake_forecast)
    headers = {"x-surge-ledger-key": "test-ledger-key"}
    query = "horizon=2&commit=true&scheduled_for_utc=2026-08-01T06:15:00Z"

    staged = []
    for ba in ledger.REQUIRED_RTO_BAS[:-1]:
        response = client.get(f"/forecast/{ba}?{query}", headers=headers)
        assert response.status_code == 200
        staged.append(response.json())

    run_id = staged[0]["run_id"]
    assert client.post(f"/ledger/runs/{run_id}/publish").status_code == 401
    incomplete = client.post(f"/ledger/runs/{run_id}/publish", headers=headers)
    assert incomplete.status_code == 409
    assert client.get("/ledger/issuances").json()["count"] == 0

    final = client.get(
        f"/forecast/{ledger.REQUIRED_RTO_BAS[-1]}?{query}", headers=headers
    )
    assert final.status_code == 200
    assert final.json()["committed"] is True
    assert final.json()["run_published"] is True
    assert final.json()["published_at_utc"] is not None

    published = client.post(f"/ledger/runs/{run_id}/publish", headers=headers)
    retry = client.post(f"/ledger/runs/{run_id}/publish", headers=headers)
    assert published.status_code == 200
    assert retry.status_code == 200
    assert published.json() == retry.json()
    assert published.json()["run_id"] == run_id
    assert published.json()["required_bas"] == list(ledger.REQUIRED_RTO_BAS)
    assert client.get("/ledger/issuances").json()["count"] == 7
    detail = client.get(f"/ledger/issuances/{staged[0]['issuance_id']}").json()
    assert detail["committed"] is True
    assert detail["run_published"] is True
    assert detail["published_at_utc"] == published.json()["published_at_utc"]

    scoreboard = client.get("/ledger/scoreboard").json()
    assert scoreboard["available_regions"] == 7
    assert {row["forecast"]["run_id"] for row in scoreboard["regions"]} == {run_id}


def test_commit_rejects_missing_or_wrong_key(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    monkeypatch.setenv("SURGE_LEDGER_KEY", "expected")
    monkeypatch.setattr(forecaster, "forecast_ba", _fake_forecast)

    response = client.get("/forecast/PJM?horizon=1&commit=true")

    assert response.status_code == 401


def test_commit_retry_never_returns_an_existing_wrong_horizon(
    monkeypatch, tmp_path
) -> None:
    client = _client(monkeypatch, tmp_path)
    monkeypatch.setenv("SURGE_LEDGER_KEY", "test-ledger-key")
    calls = 0

    def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _fake_forecast(*args, **kwargs)

    monkeypatch.setattr(forecaster, "forecast_ba", fake)
    headers = {"x-surge-ledger-key": "test-ledger-key"}
    slot = "2026-08-01T06:15:00Z"

    first = client.get(
        f"/forecast/PJM?horizon=2&commit=true&scheduled_for_utc={slot}",
        headers=headers,
    )
    wrong_horizon_retry = client.get(
        f"/forecast/PJM?horizon=3&commit=true&scheduled_for_utc={slot}",
        headers=headers,
    )

    assert first.status_code == 200
    assert len(first.json()["points"]) == 2
    assert wrong_horizon_retry.status_code == 409
    assert calls == 2


def test_commit_retry_never_reuses_different_code_identity(
    monkeypatch, tmp_path
) -> None:
    client = _client(monkeypatch, tmp_path)
    monkeypatch.setenv("SURGE_LEDGER_KEY", "test-ledger-key")
    monkeypatch.setattr(forecaster, "forecast_ba", _fake_forecast)
    headers = {"x-surge-ledger-key": "test-ledger-key"}
    slot = "2026-08-01T06:15:00Z"
    url = f"/forecast/PJM?horizon=2&commit=true&scheduled_for_utc={slot}"

    first = client.get(url, headers=headers)
    monkeypatch.setattr(forecaster, "CODE_REVISION", "different-code-sha")
    changed_code_retry = client.get(url, headers=headers)

    assert first.status_code == 200
    assert changed_code_retry.status_code == 409


def test_batch_bake_commits_and_publishes_all_seven_in_one_request(
    monkeypatch, tmp_path
) -> None:
    client = _client(monkeypatch, tmp_path)
    monkeypatch.setenv("SURGE_LEDGER_KEY", "test-ledger-key")
    monkeypatch.setattr(forecaster, "forecast_ba", _fake_forecast)
    slot = (
        datetime.now(tz=UTC)
        .replace(minute=0, second=0, microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    response = client.post(
        f"/ledger/runs/bake?horizon=2&scheduled_for_utc={slot}",
        headers={"x-surge-ledger-key": "test-ledger-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["committed_regions"] == 7
    assert payload["run_published"] is True
    assert payload["run"]["required_bas"] == list(ledger.REQUIRED_RTO_BAS)
    assert [item["ba"] for item in payload["forecasts"]] == list(
        ledger.REQUIRED_RTO_BAS
    )
    assert all(item["committed"] for item in payload["forecasts"])
    assert all(item["run_published"] for item in payload["forecasts"])


def test_batch_bake_rejects_an_old_release_slot(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    monkeypatch.setenv("SURGE_LEDGER_KEY", "test-ledger-key")
    monkeypatch.setattr(forecaster, "forecast_ba", _fake_forecast)
    slot = (
        (datetime.now(tz=UTC) - timedelta(hours=2))
        .replace(minute=0, second=0, microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    response = client.post(
        f"/ledger/runs/bake?horizon=2&scheduled_for_utc={slot}",
        headers={"x-surge-ledger-key": "test-ledger-key"},
    )

    assert response.status_code == 422
    assert "too old for a live release" in response.json()["detail"]


def test_empty_scoreboard_is_explicitly_unavailable(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)

    response = client.get("/ledger/scoreboard")

    assert response.status_code == 200
    payload = response.json()
    assert payload["expected_regions"] == 7
    assert payload["available_regions"] == 0
    assert payload["state"] == "unavailable"
    assert all(region["forecast"] is None for region in payload["regions"])


def test_public_ledger_listing_caps_expensive_page_size(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)

    response = client.get("/ledger/issuances?limit=101")

    assert response.status_code == 422
