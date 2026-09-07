"""Typed state for the estimation graph.

``budget_matches`` and ``errors`` use an ``operator.add`` reducer because
they grow incrementally as the graph runs (one component's search results,
one node's failure) rather than being replaced wholesale by each node.
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


class EstimationState(TypedDict):
    transcript: str
    estimation_id: str
    requirements: list[str]
    components: list[Component]
    # Accumulator field: grows as each component is searched.
    budget_matches: Annotated[list[BudgetMatch], operator.add]
    estimate: Optional[dict]
    status: Optional[str]  # "validated" | "needs_review"
    errors: Annotated[list[str], operator.add]
