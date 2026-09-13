"""Unit tests for the ``Phase`` grounding/citation schema (Session 10).

Pure Pydantic validation — no LLM, no mocks, no I/O. Covers the
``grounded`` <-> ``sources`` consistency invariant added to ``Phase`` so a
phase's citation claim (grounded=True/False) always matches the actual
presence/absence of ``SourceReference`` entries.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.estimation import Phase, SourceReference


def _phase(**overrides) -> Phase:
    defaults = dict(
        name="Backend build",
        duration_weeks=4,
        cost_eur=12_000,
        summary="Core API, auth and the Postgres schema for the CRM module.",
    )
    defaults.update(overrides)
    return Phase(**defaults)


def _source(**overrides) -> SourceReference:
    defaults = dict(
        chunk_id="42",
        document_id="7",
        evidence="Backend build estimated at 320h across 3 engineers.",
    )
    defaults.update(overrides)
    return SourceReference(**defaults)


def test_grounded_true_with_one_source_is_valid() -> None:
    phase = _phase(grounded=True, sources=[_source()])
    assert phase.grounded is True
    assert len(phase.sources) == 1


def test_grounded_true_with_zero_sources_raises() -> None:
    with pytest.raises(ValidationError, match="grounded"):
        _phase(grounded=True, sources=[])


def test_grounded_false_with_zero_sources_is_valid() -> None:
    phase = _phase(grounded=False, sources=[])
    assert phase.grounded is False
    assert phase.sources == []


def test_grounded_false_with_one_source_raises() -> None:
    with pytest.raises(ValidationError, match="grounded"):
        _phase(grounded=False, sources=[_source()])


def test_phase_defaults_to_ungrounded_with_no_sources_when_omitted() -> None:
    """Backward compatibility: existing call sites (guardrails' fallback
    phase, older tests) that build a ``Phase`` without mentioning grounding
    must keep working — defaulting to an honest "ungrounded" phase rather
    than breaking construction."""
    phase = _phase()
    assert phase.grounded is False
    assert phase.sources == []
