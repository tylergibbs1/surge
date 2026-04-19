"""Smoke tests for the central BA registry."""
from __future__ import annotations

import pytest

from surge import bas as _bas


def test_registry_counts() -> None:
    # EIA-930 publishes 67 BAs (81 facets minus 14 regional rollups).
    assert len(_bas.all_codes()) == 67
    # 14 of those are generator-/transmission-only with no demand series.
    assert len(_bas.demand_codes()) == 53
    # The 7 organized markets that made up v1.
    assert len(_bas.rto_codes()) == 7


def test_rto_codes_stable() -> None:
    assert set(_bas.rto_codes()) == {
        "PJM", "CISO", "ERCO", "MISO", "NYIS", "ISNE", "SWPP",
    }


def test_every_ba_has_a_station() -> None:
    # Stations are used for ASOS weather; every BA should have one so map
    # rendering and weather backfill can proceed without special cases.
    for code in _bas.all_codes():
        b = _bas.get(code)
        assert b.station is not None, f"{code} missing station"
        assert len(b.station) == 3, f"{code} station '{b.station}' not 3-letter"


def test_interconnects_partition_cleanly() -> None:
    valid = {"Eastern", "Western", "Texas"}
    for code in _bas.all_codes():
        assert _bas.get(code).interconnect in valid


def test_filter_by_interconnect() -> None:
    east = _bas.filter_codes(interconnect="Eastern")
    west = _bas.filter_codes(interconnect="Western")
    texas = _bas.filter_codes(interconnect="Texas")
    assert "PJM" in east and "CISO" in west and "ERCO" in texas
    assert len(east) + len(west) + len(texas) == len(_bas.all_codes())


def test_get_raises_on_unknown() -> None:
    with pytest.raises(KeyError):
        _bas.get("NOPE")
