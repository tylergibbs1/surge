"""Compatibility facade over Surge's shared, versioned feature layer.

Historical experiments intentionally use ``retrospective_final`` data. That
mode is useful for model development but must not be described as a replay of
what was knowable at an old issuance time.
"""

from __future__ import annotations

from datetime import datetime

from surge.features import (
    LOAD_V2_CORE,
    AvailabilityMode,
    BAData,
    load_ba_data,
    load_multi_ba_data,
)

__all__ = ["BAData", "load_multi_ba"]


def _join_ba(
    ba: str,
    *,
    with_gen: bool = False,
    availability_mode: AvailabilityMode | str = AvailabilityMode.RETROSPECTIVE_FINAL,
    cutoff: datetime | None = None,
    valid_before: datetime | None = None,
) -> BAData:
    return load_ba_data(
        ba,
        availability_mode=availability_mode,
        cutoff=cutoff,
        valid_before=valid_before,
        include_generation=with_gen,
        spec=LOAD_V2_CORE,
    )


def load_multi_ba(
    bas: list[str],
    *,
    with_gen: bool = False,
    availability_mode: AvailabilityMode | str = AvailabilityMode.RETROSPECTIVE_FINAL,
    cutoff: datetime | None = None,
    valid_before: datetime | None = None,
) -> dict[str, BAData]:
    """Load multi-BA experiment data; observed generation is opt-in."""
    return load_multi_ba_data(
        bas,
        availability_mode=availability_mode,
        cutoff=cutoff,
        valid_before=valid_before,
        include_generation=with_gen,
        spec=LOAD_V2_CORE,
    )
