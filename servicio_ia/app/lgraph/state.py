"""Typed state for the multi-agent estimation graph (extends the S13 state).

``budget_matches``, ``errors`` and ``agent_contributions`` use an
``operator.add`` reducer because they grow incrementally as the graph runs
(one component's search results, one node's failure, one agent's turn)
rather than being replaced wholesale by each node. ``agent_contributions``
doubles as the supervisor's own memory of who has already run: it decides
the next agent by checking which names are in that list, not by inspecting
business-data shape (an empty ``budget_matches`` is a valid outcome of
``budget_searcher`` having run, not a sign it hasn't).
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict


class Component(TypedDict):
    name: str
    category: str


class BudgetMatch(TypedDict):
    component: str
    reference_budget_id: str
    amount: float


class AgentContribution(TypedDict):
    agent: str
    summary: str


class TokenUsage(TypedDict):
    agent: str
    call: str
    input_tokens: int
    output_tokens: int


class EstimationState(TypedDict):
    transcript: str
    estimation_id: str
    requirements: list[str]
    components: list[Component]
    # Accumulator field: grows as each component is searched.
    budget_matches: Annotated[list[BudgetMatch], operator.add]
    estimate: Optional[dict]
    validation: Optional[dict]
    confidence: Optional[float]
    status: Optional[str]  # "validated" | "needs_review" | "rejected"
    errors: Annotated[list[str], operator.add]
    # Accumulator field: one entry per model call that reports token usage
    # (currently only requirements_extractor's two Responses API calls) --
    # feeds the per-request cost/latency metrics in app.observability.
    token_usage: Annotated[list[TokenUsage], operator.add]
    # Accumulator field: one entry per agent turn, in run order.
    agent_contributions: Annotated[list[AgentContribution], operator.add]
    # Set by human_review_gate only when a human actually reviewed the case.
    human_decision: Optional[dict]
