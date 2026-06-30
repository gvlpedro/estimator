"""Stress scenario shape + fact-tracker slicing."""

from __future__ import annotations

import pytest

from evals.stress.attachments import (
    ATTACHMENT_SIZES,
    MARKER_PHRASE,
    generate_attachment,
)
from evals.stress.scenarios import (
    PROFILES,
    SUPPORTED_TURN_LENGTHS,
    scenarios_for_length,
)


def test_each_profile_has_exactly_twenty_turns() -> None:
    """20 is the longest supported slice. Every profile must define enough
    material to fill it; otherwise the longest stress run would crash
    silently."""
    for name, generator in PROFILES.items():
        turns = generator()
        assert len(turns) == 20, f"{name!r} only has {len(turns)} turns"


@pytest.mark.parametrize("n", SUPPORTED_TURN_LENGTHS)
def test_slicing_keeps_assertions_introduced_before_n(n: int) -> None:
    scenario = scenarios_for_length("growing", n)
    assert len(scenario.turns) == n
    for assertion in scenario.assertions:
        assert assertion.introduced_at_turn <= n


def test_growing_profile_persists_project_name_assertion_at_20() -> None:
    """The growing profile pins ``Nimbus`` as the project name on turn 1
    and reinforces the persistence assertion at turn 20."""
    scenario = scenarios_for_length("growing", 20)
    persistence = [
        a for a in scenario.assertions
        if a.fact == "Nimbus" and a.label == "project_name_persistence"
    ]
    assert len(persistence) == 1


def test_pivot_profile_marks_old_stack_forbidden_after_turn_5() -> None:
    scenario = scenarios_for_length("pivot", 6)
    forbidden_react = [
        a for a in scenario.assertions
        if a.fact == "React" and a.kind == "forbidden"
    ]
    assert forbidden_react, "pivot profile must mark React as forbidden after turn 5"
    assert all(a.introduced_at_turn >= 5 for a in forbidden_react)


def test_contradiction_profile_supersedes_30k_after_turn_8() -> None:
    """The 30k€ budget assertion introduced at turn 3 should declare
    ``superseded_at_turn=8`` so the metric can relax it past that turn."""
    scenario = scenarios_for_length("contradiction", 10)
    initial_budget = [
        a for a in scenario.assertions
        if a.label == "budget_initial"
    ]
    assert initial_budget, "initial budget assertion missing"
    assert initial_budget[0].superseded_at_turn == 8


def test_scenarios_for_length_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        scenarios_for_length("unknown_profile", 3)
    with pytest.raises(ValueError):
        scenarios_for_length("growing", 7)  # not in {1, 3, 6, 10, 20}


def test_attachment_generator_returns_marker_phrase_for_non_zero_sizes() -> None:
    """The recall metric leans on a known marker phrase to know that any
    summary echo came from the attachment, not from the transcript."""
    for spec in ATTACHMENT_SIZES:
        attachment = generate_attachment(spec)
        if spec.target_chars == 0:
            assert attachment is None
            continue
        assert attachment is not None
        assert attachment.marker_phrase == MARKER_PHRASE
        assert attachment.data.startswith(b"%PDF")
