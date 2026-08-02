"""Suite-wide isolation from a developer's persistent Surge data store."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with an empty, private parquet root."""
    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path / "surge-data"))
