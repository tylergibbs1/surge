"""Named, frozen evaluation-lane boundaries.

A split is a protocol commitment, not a tuning parameter. Before this module the
boundaries were two bare integers inside the feature builder, so moving a lane
boundary looked like editing a number rather than redefining what every published
metric means.

Each declaration here is immutable and paired with the document that froze it.
Exactly one is ``ACTIVE``. Changing which one is active changes the meaning of
every artifact produced afterwards, so it is a deliberate, reviewed act with a
test that fails if the active declaration drifts from what was frozen.

Boundaries are calendar years in UTC and are half-open: training is everything
before ``validation_year``, validation is that single year, and the locked test
lane is ``locked_test_from_year`` onward.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitDeclaration:
    """One frozen assignment of calendar years to evaluation lanes."""

    name: str
    validation_year: int
    locked_test_from_year: int
    frozen_by: str

    def __post_init__(self) -> None:
        if self.locked_test_from_year <= self.validation_year:
            raise ValueError(
                f"{self.name}: the locked test lane must start after the validation year"
            )

    @property
    def train_before_year(self) -> int:
        """Training data is everything strictly before this year."""
        return self.validation_year

    def lane_of(self, year: int) -> str:
        if year >= self.locked_test_from_year:
            return "locked-test"
        if year == self.validation_year:
            return "validation"
        return "train"


V0_2 = SplitDeclaration(
    name="v0.2",
    validation_year=2024,
    locked_test_from_year=2025,
    frozen_by="docs/model-selection-experiment.md",
)

# Declared, not active. v0.2's single authorized look at its locked lane was
# consumed by a fail-closed error, so that lane can never yield a metric again.
# With a calendar boundary, every day of 2026 onward is otherwise born into that
# burned lane. Activating this requires the checklist in
# docs/v0.3-split-declaration.md, and amending it is only legitimate while no
# 2026 row has been ingested.
V0_3 = SplitDeclaration(
    name="v0.3",
    validation_year=2025,
    locked_test_from_year=2026,
    frozen_by="docs/v0.3-split-declaration.md",
)

DECLARATIONS = {declaration.name: declaration for declaration in (V0_2, V0_3)}

#: The declaration every feature build, experiment and artifact currently uses.
ACTIVE = V0_2
