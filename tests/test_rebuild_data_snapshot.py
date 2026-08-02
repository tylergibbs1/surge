"""Recovery-snapshot integrity and destructive-action guards."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from scripts import rebuild_data_snapshot as snapshot


def test_empty_dataset_directory_is_treated_as_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    (tmp_path / "load_hourly").mkdir()

    assert snapshot._canonical_dataset("load_hourly").is_empty()


def test_validation_uses_non_null_rows_and_rejects_future_data() -> None:
    now = datetime.now(tz=UTC)
    frame = pl.DataFrame(
        {
            "ts_utc": [now + timedelta(hours=3)],
            "ba": ["PJM"],
            "load_mw": [1_000.0],
        }
    )

    with pytest.raises(ValueError, match="future-dated BAs=PJM"):
        snapshot._validate_dataset(
            "load_hourly",
            frame,
            max_age_hours=None,
            allow_incomplete=True,
            require_all_demand_bas=False,
        )

    timestamps = [now - timedelta(hours=index) for index in range(2_049)]
    all_null = pl.DataFrame(
        {
            "ts_utc": timestamps,
            "ba": ["PJM"] * len(timestamps),
            "load_mw": [None] * len(timestamps),
        },
        schema_overrides={"load_mw": pl.Float64},
    )
    with pytest.raises(ValueError, match="fewer than 2049 usable rows=PJM"):
        snapshot._validate_dataset(
            "load_hourly",
            all_null,
            max_age_hours=12,
            allow_incomplete=False,
            require_all_demand_bas=False,
        )


def test_force_does_not_replace_an_arbitrary_directory(tmp_path: Path) -> None:
    output = tmp_path / "data_snapshot"
    output.mkdir()
    user_file = output / "user-file.txt"
    user_file.write_text("preserve me", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to replace non-snapshot directory"):
        snapshot.build_snapshot(
            output,
            force=True,
            max_input_age_hours=None,
            allow_incomplete=False,
            require_all_demand_bas=False,
        )

    assert user_file.read_text(encoding="utf-8") == "preserve me"


def test_verify_rejects_tampered_dataset_summary(tmp_path: Path) -> None:
    files: list[dict[str, object]] = []
    datasets: dict[str, dict[str, object]] = {}
    for dataset, value_column in snapshot.DATASET_VALUE_COLUMNS.items():
        relative = f"{dataset}/year=2026/month=08/part.parquet"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "ts_utc": [datetime(2026, 8, 1, tzinfo=UTC)],
                "ba": ["PJM"],
                value_column: [1_000.0],
            }
        ).write_parquet(path)
        files.append({"path": relative, "rows": 1, "sha256": snapshot._sha256(path)})
        datasets[dataset] = {"rows": 1}

    manifest = {
        # A v1 fixture also proves old frozen research snapshots remain
        # verifiable after the recovery format gained verbatim ledger files.
        "format_version": 1,
        "validation": {
            "max_input_age_hours": None,
            "allow_incomplete": True,
            "require_all_demand_bas": False,
        },
        "datasets": datasets,
        "files": files,
    }
    manifest_path = tmp_path / "snapshot-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    snapshot.verify_snapshot(tmp_path)

    manifest["validation"]["allow_incomplete"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="missing BAs="):
        snapshot.verify_snapshot(tmp_path)
    manifest["validation"]["allow_incomplete"] = True

    manifest["datasets"]["weather_hourly"]["rows"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset row-count mismatch: weather_hourly"):
        snapshot.verify_snapshot(tmp_path)


def test_snapshot_copies_all_ledger_datasets_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "recovery"
    monkeypatch.setenv("SURGE_DATA_DIR", str(source))
    observed_at = datetime.now(tz=UTC) - timedelta(hours=1)

    for dataset, value_column in snapshot.DATASET_VALUE_COLUMNS.items():
        frame = pl.DataFrame(
            {
                "ts_utc": [observed_at],
                "ba": ["PJM"],
                value_column: [1_000.0],
                "source": ["fixture"],
                "as_of": [observed_at],
            }
        )
        path = source / dataset / "year=2026" / "month=08" / "input.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)

    original_bytes: dict[str, bytes] = {}
    for index, dataset in enumerate(snapshot.LEDGER_DATASETS):
        relative = f"{dataset}/ledger_mode=live/record-{index}.parquet"
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"record": [dataset], "ordinal": [index]}).write_parquet(path)
        original_bytes[relative] = path.read_bytes()

    snapshot.build_snapshot(
        output,
        force=False,
        max_input_age_hours=None,
        allow_incomplete=True,
        require_all_demand_bas=False,
    )
    snapshot.verify_snapshot(output)

    manifest = json.loads((output / "snapshot-manifest.json").read_text())
    assert manifest["format_version"] == snapshot.FORMAT_VERSION
    assert len(manifest["content_sha256"]) == 64
    for dataset in snapshot.LEDGER_DATASETS:
        summary = manifest["datasets"][dataset]
        assert summary["storage_mode"] == "verbatim"
        assert summary["file_count"] == 1
        assert len(summary["content_sha256"]) == 64
    for relative, expected in original_bytes.items():
        assert (output / relative).read_bytes() == expected

    tampered = output / next(iter(original_bytes))
    tampered.write_bytes(tampered.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        snapshot.verify_snapshot(output)
