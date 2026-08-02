"""Implausible demand must be refused where it enters, not rescued downstream."""

from __future__ import annotations

import polars as pl

from surge.features.data import LOAD_VALIDITY_MW  # type: ignore[attr-defined]
from surge.scrapers.eia import PLAUSIBLE_LOAD_MW, _reject_implausible_load


def _frame(values: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame({"load_mw": values, "ba": ["NYIS"] * len(values)})


def test_the_int32_sentinel_is_refused() -> None:
    """PJM ships 2,147,480,000 MW. It reached the store for years."""
    out = _reject_implausible_load(_frame([100.0, 2_147_480_000.0]), ba="PJM")
    assert out["load_mw"].to_list() == [100.0, None]


def test_hard_zeros_are_refused() -> None:
    """Two NYIS zeros consumed the v0.2 locked-test look."""
    out = _reject_implausible_load(_frame([0.0, 20_000.0]), ba="NYIS")
    assert out["load_mw"].to_list() == [None, 20_000.0]


def test_an_absurd_but_finite_value_is_refused() -> None:
    """SWPP peaks near 50 GW; the store held 3,621,097 MW."""
    out = _reject_implausible_load(_frame([3_621_097.0]), ba="SWPP")
    assert out["load_mw"].to_list() == [None]


def test_plausible_values_and_row_count_survive() -> None:
    out = _reject_implausible_load(_frame([1.0, 50_000.0, 200_000.0, None]), ba="PJM")
    assert out["load_mw"].to_list() == [1.0, 50_000.0, 200_000.0, None]
    assert out.height == 4


def test_the_boundary_matches_the_feature_layer_rule() -> None:
    """A value the store accepts and the feature layer rejects is invisible."""
    assert PLAUSIBLE_LOAD_MW == LOAD_VALIDITY_MW
