"""Stress-specific metrics.

Why this lives next to ``evals/metrics.py`` instead of inside it
---------------------------------------------------------------
The base ``evals/metrics.py`` family scores a single ``(GoldenCase,
EstimationResult)`` pair: input + final answer, nothing else. The stress
metrics consume different shapes:

- ``LatencyBudgetMetric`` and ``CostBudgetMetric`` evaluate a single
  ``TurnObservation`` — the per-turn rollup carried by the ``turn_observed``
  log event. They have nothing to say about an estimation under review.
- ``MemoryDriftMetric`` evaluates a ``SessionSnapshot`` — the *state* of the
  conversation after N turns: metadata, history messages, anchors, summary.
  No ``GoldenCase`` here.

Forcing those into the base ``MetricResult`` family would either bloat the
base contract or hide stress concerns behind type-unions that bite later. A
sibling module is cheaper.

Determinism > sophistication. No embeddings, no LLM-as-judge. Substring match
(case-insensitive) over the fields we control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evals.metrics import MetricResult


# ---------------------------------------------------------------------------
# Inputs the metrics evaluate against
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnObservation:
    """One row of the per-turn CSV. Field-for-field mirror of the
    ``turn_observed`` log event so reading the event off structlog and
    feeding it here is a one-liner."""

    turn_index: int
    session_id: str
    enriched_transcript_chars: int
    attachments_total_chars: int
    messages_in_window: int
    anchors_count: int
    summary_chars: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int
    cache_hit_kind: str
    last_resolved_tier: str | None = None


@dataclass(frozen=True)
class SessionSnapshot:
    """What ``MemoryDriftMetric`` reads.

    Provided by the runner from the GET /sessions/{id} response plus the
    final assistant reply summary captured during the turn.
    """

    session_id: str
    summary: str = ""
    anchors: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_assistant_summary: str = ""


# ---------------------------------------------------------------------------
# Budget metrics — per turn
# ---------------------------------------------------------------------------


class LatencyBudgetMetric:
    """``1.0`` if ``latency_ms <= budget_ms``; ``0.0`` otherwise.

    Used to gate the per-turn CSV row. Budget is an explicit number; nothing
    is inferred. Pass it at construction and reuse the same instance across
    a run.
    """

    def __init__(self, budget_ms: int) -> None:
        if budget_ms <= 0:
            raise ValueError("budget_ms must be > 0")
        self.budget_ms = budget_ms
        self.name = f"latency_budget_{budget_ms}ms"

    def evaluate(self, observation: TurnObservation) -> MetricResult:
        passed = observation.latency_ms <= self.budget_ms
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=(
                f"latency {observation.latency_ms}ms"
                f" {'<=' if passed else '>'} budget {self.budget_ms}ms"
            ),
        )


class CostBudgetMetric:
    """``1.0`` if ``cost_usd <= budget_usd``; ``0.0`` otherwise."""

    def __init__(self, budget_usd: float) -> None:
        if budget_usd <= 0:
            raise ValueError("budget_usd must be > 0")
        self.budget_usd = budget_usd
        self.name = f"cost_budget_{budget_usd:g}usd"

    def evaluate(self, observation: TurnObservation) -> MetricResult:
        passed = observation.cost_usd <= self.budget_usd
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=(
                f"cost ${observation.cost_usd:.4f}"
                f" {'<=' if passed else '>'} budget ${self.budget_usd:.4f}"
            ),
        )


# ---------------------------------------------------------------------------
# Memory drift — per fact, evaluated against the post-turn session snapshot
# ---------------------------------------------------------------------------


_DEFAULT_WHERE = ("summary", "anchors", "metadata")


class MemoryDriftMetric:
    """``1.0`` if the declared fact appears in any of the surfaces in
    ``where`` of the post-turn snapshot; ``0.0`` otherwise.

    Substring match, case-insensitive, no normalisation beyond ``casefold``.
    The point is to catch the system *forgetting* a fact that an earlier turn
    introduced — not to grade prose quality.

    For a ``forbidden`` fact (e.g. the pre-pivot React stack), invert the
    semantics with ``forbidden=True``: ``1.0`` if the fact is **absent** from
    every scanned surface, ``0.0`` otherwise.
    """

    def __init__(
        self,
        fact: str,
        where: list[str] | tuple[str, ...] = _DEFAULT_WHERE,
        *,
        forbidden: bool = False,
        label: str = "",
    ) -> None:
        if not fact or not fact.strip():
            raise ValueError("fact must be a non-empty string")
        unknown = [w for w in where if w not in _DEFAULT_WHERE]
        if unknown:
            raise ValueError(
                f"unknown surfaces in `where`: {unknown}; valid: {_DEFAULT_WHERE}"
            )
        self.fact = fact
        self.where = tuple(where)
        self.forbidden = forbidden
        base = "memory_forbidden" if forbidden else "memory_recall"
        suffix = label or fact.replace(" ", "_")[:40]
        self.name = f"{base}::{suffix}"

    def evaluate(self, snapshot: SessionSnapshot) -> MetricResult:
        needle = self.fact.casefold()
        hits: list[str] = []
        if "summary" in self.where:
            blob = (snapshot.summary or "") + " " + (snapshot.last_assistant_summary or "")
            if needle in blob.casefold():
                hits.append("summary")
        if "anchors" in self.where and snapshot.anchors:
            joined = " ".join(snapshot.anchors).casefold()
            if needle in joined:
                hits.append("anchors")
        if "metadata" in self.where and snapshot.metadata:
            joined = _flatten_metadata(snapshot.metadata).casefold()
            if needle in joined:
                hits.append("metadata")

        if self.forbidden:
            passed = not hits
            details = (
                f"forbidden fact {self.fact!r} absent from {list(self.where)}"
                if passed
                else f"forbidden fact {self.fact!r} leaked into {hits}"
            )
        else:
            passed = bool(hits)
            details = (
                f"fact {self.fact!r} found in {hits}"
                if passed
                else f"fact {self.fact!r} missing from {list(self.where)}"
            )
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=details,
        )


def _flatten_metadata(meta: dict[str, Any]) -> str:
    """Stringify a nested metadata dict into a single haystack.

    The metric does substring matching, so the exact layout does not matter —
    we just need every value to be discoverable in one pass.
    """
    parts: list[str] = []

    def _walk(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (str, int, float, bool)):
            parts.append(str(value))
            return
        if isinstance(value, dict):
            for v in value.values():
                _walk(v)
            return
        if isinstance(value, (list, tuple, set)):
            for v in value:
                _walk(v)
            return
        parts.append(str(value))

    _walk(meta)
    return " ".join(parts)
