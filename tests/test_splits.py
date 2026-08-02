"""Evaluation-lane boundaries are protocol commitments, not tunables."""

from __future__ import annotations

import pytest

from surge.features.splits import ACTIVE, DECLARATIONS, V0_2, V0_3, SplitDeclaration


def test_active_declaration_matches_the_frozen_v02_protocol() -> None:
    """Every published v0.2 artifact assumes exactly these boundaries.

    If this fails, either the active split moved or v0.2 was edited. Both
    invalidate the frozen artifacts under artifacts/v0.2/ and neither may be
    done to make another test pass.
    """
    assert ACTIVE is V0_2
    assert V0_2.train_before_year == 2024
    assert V0_2.validation_year == 2024
    assert V0_2.locked_test_from_year == 2025


def test_v03_is_declared_but_not_active() -> None:
    assert V0_3.validation_year == 2025
    assert V0_3.locked_test_from_year == 2026
    assert ACTIVE is not V0_3


def test_lanes_partition_the_calendar() -> None:
    assert V0_2.lane_of(2023) == "train"
    assert V0_2.lane_of(2024) == "validation"
    assert V0_2.lane_of(2025) == "locked-test"
    assert V0_2.lane_of(2026) == "locked-test"
    # The v0.2 lane assignment is why 2026 data has no usable home today.
    assert V0_3.lane_of(2025) == "validation"
    assert V0_3.lane_of(2026) == "locked-test"


def test_the_feature_builder_uses_the_active_declaration() -> None:
    """The boundaries must come from the declaration, not from literals."""
    from surge.features import data

    assert data.ACTIVE_SPLIT is ACTIVE


def test_an_overlapping_declaration_is_rejected() -> None:
    with pytest.raises(ValueError, match="must start after the validation year"):
        SplitDeclaration(
            name="bad",
            validation_year=2025,
            locked_test_from_year=2025,
            frozen_by="nowhere",
        )


def test_every_declaration_names_the_document_that_froze_it() -> None:
    for declaration in DECLARATIONS.values():
        assert declaration.frozen_by.endswith(".md")
