"""Three stress metrics: latency / cost / memory drift.

These tests do NOT call the LLM. They build hand-crafted ``TurnObservation``
and ``SessionSnapshot`` instances and check that each metric routes the
pass/fail decision correctly, including the inverted semantics of a
``forbidden`` MemoryDriftMetric."""

from __future__ import annotations

import pytest

from evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
    SessionSnapshot,
    TurnObservation,
)


def _observation(*, latency_ms: int = 1000, cost_usd: float = 0.01) -> TurnObservation:
    return TurnObservation(
        turn_index=1,
        session_id="sess-1",
        enriched_transcript_chars=120,
        attachments_total_chars=0,
        messages_in_window=2,
        anchors_count=0,
        summary_chars=0,
        tokens_in=500,
        tokens_out=200,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        cache_hit_kind="none",
    )


def test_latency_budget_passes_when_under_budget() -> None:
    metric = LatencyBudgetMetric(budget_ms=5_000)
    result = metric.evaluate(_observation(latency_ms=3_200))
    assert result.passed
    assert result.score == 1.0
    assert "3200ms" in result.details


def test_latency_budget_fails_when_over_budget() -> None:
    metric = LatencyBudgetMetric(budget_ms=5_000)
    result = metric.evaluate(_observation(latency_ms=8_000))
    assert not result.passed
    assert result.score == 0.0


def test_latency_budget_rejects_zero_or_negative_budget() -> None:
    with pytest.raises(ValueError):
        LatencyBudgetMetric(budget_ms=0)


def test_cost_budget_passes_at_boundary() -> None:
    metric = CostBudgetMetric(budget_usd=0.05)
    result = metric.evaluate(_observation(cost_usd=0.05))
    assert result.passed, "boundary must be inclusive (<=, not <)"


def test_cost_budget_fails_when_over_budget() -> None:
    metric = CostBudgetMetric(budget_usd=0.05)
    result = metric.evaluate(_observation(cost_usd=0.06))
    assert not result.passed


def test_memory_drift_passes_when_fact_in_summary() -> None:
    snapshot = SessionSnapshot(
        session_id="sess-1",
        summary="Project Nimbus delivers a CRM for sales teams.",
        metadata={"project_name": "Nimbus"},
    )
    result = MemoryDriftMetric(fact="Nimbus", label="project_name").evaluate(snapshot)
    assert result.passed
    assert "summary" in result.details


def test_memory_drift_case_insensitive_match() -> None:
    snapshot = SessionSnapshot(
        session_id="sess-1",
        last_assistant_summary="The project nimbus is the agreed name.",
    )
    result = MemoryDriftMetric(fact="Nimbus").evaluate(snapshot)
    assert result.passed


def test_memory_drift_fails_when_fact_absent() -> None:
    snapshot = SessionSnapshot(
        session_id="sess-1",
        summary="Different project name.",
        metadata={"project_name": "Atlas"},
    )
    result = MemoryDriftMetric(fact="Nimbus", label="project_name").evaluate(snapshot)
    assert not result.passed
    assert "missing" in result.details


def test_memory_drift_forbidden_passes_when_value_absent() -> None:
    """The pivot scenario forbids `React` after turn 5; success means the
    string is nowhere on the surfaces we scan."""
    snapshot = SessionSnapshot(
        session_id="sess-1",
        summary="Pure Flutter app: iOS and Android.",
        metadata={"mentioned_technologies": ["Flutter", "Firebase"]},
    )
    result = MemoryDriftMetric(
        fact="React", forbidden=True, label="forbidden_react"
    ).evaluate(snapshot)
    assert result.passed


def test_memory_drift_forbidden_fails_when_old_value_leaks_in() -> None:
    snapshot = SessionSnapshot(
        session_id="sess-1",
        summary="Pure Flutter app. Originally React, but pivoted.",
        metadata={"mentioned_technologies": ["Flutter", "React", "Firebase"]},
    )
    result = MemoryDriftMetric(
        fact="React", forbidden=True, label="forbidden_react"
    ).evaluate(snapshot)
    assert not result.passed
    assert "leaked" in result.details


def test_memory_drift_rejects_unknown_where_surface() -> None:
    with pytest.raises(ValueError):
        MemoryDriftMetric(fact="x", where=["bogus_surface"])


def test_memory_drift_scans_metadata_nested_lists() -> None:
    """Nested values (lists of strings) must be reachable by the substring
    match. The flatten helper is what makes that work."""
    snapshot = SessionSnapshot(
        session_id="sess-1",
        metadata={"mentioned_technologies": ["React", "Postgres", "FastAPI"]},
    )
    assert MemoryDriftMetric(fact="Postgres").evaluate(snapshot).passed
