"""Stress test harness — multi-turn conversations + attachment pressure.

Four pieces, in dependency order:

- ``scenarios.py``: three conversational profiles (growing / pivot /
  contradict) with a per-turn fact tracker. Pure data — no network.
- ``attachments.py``: synthetic PDF generator that produces calibrated
  payloads (0 / 5 / 20 / 50 / 100 KB) for the attachment-pressure axis.
- ``metrics.py``: ``LatencyBudgetMetric``, ``CostBudgetMetric``,
  ``MemoryDriftMetric``. Deterministic, no LLM-as-judge, no embeddings.
- ``run.py``: CLI orchestrator that combines the three axes
  (scenario × attachment size × repeats), writes a per-turn CSV and a
  human-readable ``REPORT.md`` for the live session.

Why a separate ``evals/stress/`` package
----------------------------------------
The base ``evals/`` framework scores one shot at a time against a
``GoldenCase``. Stress tests score *trajectories* — a sequence of turns
where state matters and where the same input deliberately reaches the
model under different conditions. Mixing both in ``evals/metrics.py``
would force the existing metrics to know about turn observations and
session snapshots they have no business reading.
"""

from evals.stress.attachments import (
    ATTACHMENT_SIZES,
    MARKER_PHRASE,
    SyntheticAttachment,
    SyntheticAttachmentSpec,
    generate_attachment,
    spec_by_label,
)
from evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
    SessionSnapshot,
    TurnObservation,
)
from evals.stress.scenarios import (
    FactAssertion,
    StressScenario,
    StressTurn,
    PROFILES,
    contradicting_project,
    growing_project,
    pivoting_project,
    scenarios_for_length,
)

__all__ = [
    "FactAssertion",
    "StressScenario",
    "StressTurn",
    "PROFILES",
    "contradicting_project",
    "growing_project",
    "pivoting_project",
    "scenarios_for_length",
    "ATTACHMENT_SIZES",
    "MARKER_PHRASE",
    "SyntheticAttachment",
    "SyntheticAttachmentSpec",
    "generate_attachment",
    "spec_by_label",
    "LatencyBudgetMetric",
    "CostBudgetMetric",
    "MemoryDriftMetric",
    "SessionSnapshot",
    "TurnObservation",
]
