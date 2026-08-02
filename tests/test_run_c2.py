"""Result-publication tests for the Chronos-2 experiment runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments import run_c2


def test_locked_receipt_gets_raw_metrics_before_display_rounding(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    receipt_path = tmp_path / "surge-locked-test-receipt.json"
    metrics: dict[str, Any] = {
        "mase": 0.12345678901234568,
        "per_ba": {
            "PJM": {
                "mase": 0.23456789012345678,
                "per_horizon": [
                    {"horizon": 1, "mase": 0.3456789012345679},
                ],
            }
        },
        "per_horizon": [
            {"horizon": 1, "mase": 0.4567890123456789},
        ],
        "load_weighted": {"mase": 0.567890123456789},
    }
    recorded: dict[str, Any] = {}

    def capture(path: Path, result: dict[str, Any]) -> None:
        recorded["path"] = path
        recorded["result"] = result

    monkeypatch.setattr(run_c2, "complete_locked_test_run", capture)

    run_c2._record_and_emit_result(
        {"exp": "locked-test"},
        metrics,
        load_s=1.2345678901234567,
        eval_s=9.876543210987654,
        locked_test_receipt=receipt_path,
    )

    assert recorded["path"] == receipt_path
    raw = recorded["result"]
    assert raw["mase"] == 0.12345678901234568
    assert raw["per_ba"]["PJM"]["mase"] == 0.23456789012345678
    assert raw["per_ba"]["PJM"]["per_horizon"][0]["mase"] == 0.3456789012345679
    assert raw["per_horizon"][0]["mase"] == 0.4567890123456789
    assert raw["load_weighted"]["mase"] == 0.567890123456789
    assert raw["load_s"] == 1.2345678901234567
    assert raw["eval_s"] == 9.876543210987654

    display = json.loads(capsys.readouterr().out.removeprefix("METRIC: "))
    assert display["mase"] == 0.1235
    assert display["per_ba"]["PJM"]["mase"] == 0.2346
    assert display["load_s"] == 1.23
    assert display["eval_s"] == 9.88


def _reserve_for_failure(tmp_path: Path, registry: Path) -> Path:
    marker_path = tmp_path / "surge-promotion.json"
    marker_path.write_text("{}", encoding="utf-8")
    return run_c2.reserve_locked_test_run(
        tmp_path / "v0.2-h100-selection.json",
        experiment="v0.2-locked-test-failure",
        training_identity={"bas": ["PJM"]},
        selection_sha256="f" * 64,
        selection_decision_sha256="a" * 64,
        experiment_protocol_sha256="b" * 64,
        promotion_path=marker_path,
        marker_sha256="d" * 64,
        checkpoint_inventory_sha256="e" * 64,
        model_artifact_sha256="c" * 64,
        registry_root=registry,
    )


def test_main_reraises_a_pre_metric_failure_without_consuming_the_look(
    monkeypatch, tmp_path: Path
) -> None:
    """This is the exact failure that destroyed the v0.2 attempt."""
    registry = tmp_path / "authoritative-registry"
    receipt_path = _reserve_for_failure(tmp_path, registry)

    def fail_after_reservation(on_locked_test_reserved) -> None:
        on_locked_test_reserved(receipt_path)
        raise ValueError("incomplete locked target window")

    monkeypatch.setattr(run_c2, "_main", fail_after_reservation)

    with pytest.raises(ValueError, match="incomplete locked target window"):
        run_c2.main()

    assert not receipt_path.exists()
    assert not (registry / f"{'b' * 64}.json").exists()
    aborts = sorted((registry / "aborts").glob("*"))
    assert len(aborts) == 1
    assert json.loads(aborts[0].read_text())["failure"] == {
        "exception_type": "ValueError",
        "message_omitted": True,
    }


def test_main_records_a_terminal_failure_once_the_look_is_spent(
    monkeypatch, tmp_path: Path
) -> None:
    registry = tmp_path / "authoritative-registry"
    receipt_path = _reserve_for_failure(tmp_path, registry)

    def fail_after_spending(on_locked_test_reserved) -> None:
        on_locked_test_reserved(receipt_path)
        run_c2.spend_locked_test_look(receipt_path)
        raise ValueError("crashed while scoring")

    monkeypatch.setattr(run_c2, "_main", fail_after_spending)

    with pytest.raises(ValueError, match="crashed while scoring"):
        run_c2.main()

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    registry_receipt = json.loads((registry / f"{'b' * 64}.json").read_text())
    assert receipt == registry_receipt
    assert receipt["status"] == "failed"
    assert receipt["test_opened"] is True
    assert receipt["failure"] == {
        "exception_type": "ValueError",
        "message_omitted": True,
    }
