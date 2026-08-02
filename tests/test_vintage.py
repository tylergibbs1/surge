"""The vintage archive is evidence, so its guarantees are tested as such."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from surge import vintage


def _rows(load: float) -> list[dict[str, object]]:
    return [{"period": "2026-01-01T00", "respondent": "PJM", "value": load}]


def _capture(root: Path, load: float, **overrides: object) -> vintage.CapturedVintage:
    kwargs: dict[str, object] = {
        "dataset": "eia930-demand",
        "ba": "PJM",
        "start": "2026-01-01",
        "end": "2026-01-02",
        "rows": _rows(load),
        "root": root,
    }
    kwargs.update(overrides)
    return vintage.capture(**kwargs)  # type: ignore[arg-type]


def test_a_revision_is_stored_beside_the_original_never_over_it(tmp_path: Path) -> None:
    """This is the whole point: revisions must remain measurable."""
    first = _capture(tmp_path, 100.0)
    revised = _capture(tmp_path, 101.0)

    assert first.payload_sha256 != revised.payload_sha256
    assert first.payload_path.exists()
    assert revised.payload_path.exists()
    assert vintage.load_payload(first.payload_sha256, "eia930-demand", "PJM", tmp_path)[
        "rows"
    ] == _rows(100.0)


def test_recapturing_an_unchanged_window_adds_no_duplicate_payload(tmp_path: Path) -> None:
    first = _capture(tmp_path, 100.0)
    again = _capture(tmp_path, 100.0)

    assert again.payload_sha256 == first.payload_sha256
    assert again.already_present is True
    payloads = list((tmp_path / "eia930-demand" / "PJM").glob("*.json.gz"))
    assert len(payloads) == 1
    # Both polls are still recorded, so capture cadence stays auditable.
    assert len(vintage.read_index(tmp_path)) == 2


def test_capture_time_does_not_change_the_payload_identity(tmp_path: Path) -> None:
    """Otherwise every poll would look like a revision."""
    early = _capture(tmp_path, 100.0, captured_at=datetime(2026, 1, 2, tzinfo=UTC))
    late = _capture(tmp_path, 100.0, captured_at=datetime(2026, 6, 2, tzinfo=UTC))
    assert early.payload_sha256 == late.payload_sha256
    assert early.captured_at_utc != late.captured_at_utc


def test_credentials_never_reach_the_archive(tmp_path: Path) -> None:
    captured = _capture(
        tmp_path,
        100.0,
        request={"api_key": "SECRET-KEY-VALUE", "frequency": "hourly"},
    )
    document = vintage.load_payload(
        captured.payload_sha256, "eia930-demand", "PJM", tmp_path
    )
    assert document["request"]["api_key"] == "<redacted>"
    assert document["request"]["frequency"] == "hourly"
    archived = (tmp_path / "eia930-demand" / "PJM").rglob("*")
    for path in archived:
        if path.is_file():
            assert b"SECRET-KEY-VALUE" not in path.read_bytes()


def test_a_tampered_payload_is_rejected_on_read(tmp_path: Path) -> None:
    captured = _capture(tmp_path, 100.0)
    captured.payload_path.write_bytes(b"not the archived bytes")
    with pytest.raises(ValueError, match="no longer matches its digest"):
        vintage.load_payload(captured.payload_sha256, "eia930-demand", "PJM", tmp_path)


def test_the_index_is_append_only(tmp_path: Path) -> None:
    _capture(tmp_path, 100.0)
    _capture(tmp_path, 101.0)
    _capture(tmp_path, 102.0)
    entries = vintage.read_index(tmp_path)
    assert [entry["row_count"] for entry in entries] == [1, 1, 1]
    lines = (tmp_path / vintage.INDEX_NAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert all(json.loads(line)["ba"] == "PJM" for line in lines)


def test_distinct_vintages_collapse_repeat_polls_but_keep_revisions(tmp_path: Path) -> None:
    _capture(tmp_path, 100.0)
    _capture(tmp_path, 100.0)
    _capture(tmp_path, 101.0)
    distinct = vintage.distinct_vintages("eia930-demand", "PJM", tmp_path)
    assert len(distinct) == 2


def test_dataset_and_ba_are_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset and ba are required"):
        _capture(tmp_path, 100.0, ba="")


def test_the_modal_archive_lives_on_the_committed_volume() -> None:
    """A container-local archive would be discarded on every cron exit.

    Read as source rather than imported: the Modal SDK is not a test dependency.
    """
    source = (
        Path(__file__).resolve().parents[1] / "modal_app" / "app.py"
    ).read_text(encoding="utf-8")
    assert '"SURGE_VINTAGE_DIR": "/workspace/data/vintage"' in source
    assert '"SURGE_DATA_DIR": "/workspace/data"' in source


def test_archiving_can_be_disabled_for_offline_use(monkeypatch, tmp_path: Path) -> None:
    """Ingest must never fail because archiving is unavailable."""
    monkeypatch.setenv("SURGE_VINTAGE_ARCHIVE", "0")
    monkeypatch.setenv("SURGE_VINTAGE_DIR", str(tmp_path))
    assert vintage.archive_root() == tmp_path
