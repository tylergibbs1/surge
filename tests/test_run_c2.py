"""Result-publication tests for the Chronos-2 experiment runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
