"""Frozen H100 candidate-selection policy tests."""

from __future__ import annotations

from experiments.select_c2_candidate import choose_candidate


def test_selection_score_and_tie_break_are_predeclared() -> None:
    slower = {
        "candidate": "official-lora-2000",
        "num_steps": 2_000,
        "promotion_eligible": True,
        "selection_score": 0.91234549,
    }
    faster = {
        "candidate": "official-lora-1000",
        "num_steps": 1_000,
        "promotion_eligible": True,
        "selection_score": 0.91234541,
    }

    assert choose_candidate([slower, faster]) == faster


def test_selection_excludes_rejected_candidate_and_can_fall_back() -> None:
    rejected = {
        "candidate": "official-lora-1000",
        "num_steps": 1_000,
        "promotion_eligible": False,
        "selection_score": 0.1,
    }
    eligible = {
        "candidate": "official-lora-2000",
        "num_steps": 2_000,
        "promotion_eligible": True,
        "selection_score": 0.95,
    }

    assert choose_candidate([rejected, eligible]) == eligible
    assert choose_candidate([rejected]) is None


def test_selection_prefers_lower_frozen_composite_score() -> None:
    first = {
        "candidate": "official-lora-1000",
        "num_steps": 1_000,
        "promotion_eligible": True,
        "selection_score": 0.90,
    }
    second = {
        "candidate": "official-lora-2000",
        "num_steps": 2_000,
        "promotion_eligible": True,
        "selection_score": 0.89,
    }

    assert choose_candidate([first, second]) == second
