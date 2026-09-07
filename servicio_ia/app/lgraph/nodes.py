"""The five estimation-graph nodes.

Each node is a pure function of the state: it reads ``state`` and returns a
partial update, never mutates ``state`` in place and never holds state
across calls (the OpenAI client and DB session are opened per call, mirroring
the dependency lifecycle already used by ``scripts/run_agent_s12.py``).

``search_budgets`` and ``generate_estimate`` reuse the S9-S10 retrieval
pipeline and the S12 costing function verbatim (``app.generation.rag`` /
``app.agents.tools``) instead of reimplementing them for the graph.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import logfire
from openai import OpenAI

from app.agents.tools import calculate_estimate
from app.agents.tools import search_budgets as retrieve_budget_for
from app.dependencies import get_embedder, get_reranker
from app.lgraph.state import BudgetMatch, EstimationState
from app.storage.db import session_factory

GRAPH_MODEL = os.getenv("GRAPH_MODEL", "gpt-5-mini")

EXTRACT_REQUIREMENTS_PROMPT = """\
You are a requirements analyst. Read the transcript of a discovery meeting \
with a client and extract the concrete functional and technical \
requirements it describes, as a flat list of short, self-contained \
statements. One requirement per distinct need -- do not merge unrelated \
needs into one bullet, and do not invent requirements the transcript does \
not support."""

CLASSIFY_COMPONENTS_PROMPT = """\
You group a list of extracted requirements into the distinct components of \
work they imply. A component is a piece of scope that would reasonably be \
estimated and budgeted on its own (e.g. a backend service, a mobile app, a \
third-party integration, a reporting dashboard). Assign each component a \
short category (e.g. "backend", "mobile", "integration", "frontend", \
"infrastructure", "data"). Do not split one coherent piece of scope into \
several components, and do not merge unrelated pieces of scope into one.

Keep "name" a SHORT label (at most 6 words, no parentheses, no restating \
every requirement it covers) -- e.g. "SAP/IDoc integration", not a full \
sentence describing everything the component does. The detail lives in the \
requirements list, not in the component name."""

_REQUIREMENTS_SCHEMA = {
    "type": "json_schema",
    "name": "requirements",
    "schema": {
        "type": "object",
        "properties": {"requirements": {"type": "array", "items": {"type": "string"}}},
        "required": ["requirements"],
        "additionalProperties": False,
    },
    "strict": True,
}

_COMPONENTS_SCHEMA = {
    "type": "json_schema",
    "name": "components",
    "schema": {
        "type": "object",
        "properties": {
            "components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "category": {"type": "string"},
                    },
                    "required": ["name", "category"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["components"],
        "additionalProperties": False,
    },
    "strict": True,
}


@lru_cache
def _get_client() -> OpenAI:
    return OpenAI()


def extract_requirements(state: EstimationState) -> dict[str, Any]:
    with logfire.span("node: extract_requirements"):
        response = _get_client().responses.create(
            model=GRAPH_MODEL,
            instructions=EXTRACT_REQUIREMENTS_PROMPT,
            input=state["transcript"],
            text={"format": _REQUIREMENTS_SCHEMA},
        )
        data = json.loads(response.output_text)
        return {"requirements": data["requirements"]}


def classify_components(state: EstimationState) -> dict[str, Any]:
    with logfire.span("node: classify_components"):
        response = _get_client().responses.create(
            model=GRAPH_MODEL,
            instructions=CLASSIFY_COMPONENTS_PROMPT,
            input=json.dumps({"requirements": state["requirements"]}, ensure_ascii=False),
            text={"format": _COMPONENTS_SCHEMA},
        )
        data = json.loads(response.output_text)
        return {"components": data["components"]}


async def search_budgets(state: EstimationState) -> dict[str, Any]:
    """Sequential, one component after another -- fan-out is the live-session case."""
    matches: list[BudgetMatch] = []
    errors: list[str] = []

    with logfire.span("node: search_budgets"):
        embedder = get_embedder()
        reranker = get_reranker()
        async with session_factory() as session:
            for component in state["components"]:
                query = f"{component['name']} ({component['category']})"
                try:
                    result = await retrieve_budget_for(
                        {"query": query, "filters": None},
                        session=session,
                        embedder=embedder,
                        reranker=reranker,
                    )
                except Exception as exc:  # noqa: BLE001 -- surfaced via state, not raised
                    errors.append(f"search_budgets failed for {component['name']!r}: {exc}")
                    continue

                for item in result["results"]:
                    budget_id = item.get("budget_id")
                    hours = item.get("estimated_hours")
                    if budget_id is None or hours is None:
                        continue
                    matches.append(
                        BudgetMatch(
                            component=component["name"],
                            reference_budget_id=str(budget_id),
                            amount=float(hours),
                        )
                    )

    return {"budget_matches": matches, "errors": errors}


def generate_estimate(state: EstimationState) -> dict[str, Any]:
    with logfire.span("node: generate_estimate"):
        by_component: dict[str, list[float]] = {c["name"]: [] for c in state["components"]}
        for match in state["budget_matches"]:
            by_component.setdefault(match["component"], []).append(match["amount"])

        args = {
            "components": [
                {"name": name, "reference_amounts": amounts}
                for name, amounts in by_component.items()
            ]
        }
        estimate = calculate_estimate(args)
        return {"estimate": estimate}


def validate_and_consolidate(state: EstimationState) -> dict[str, Any]:
    with logfire.span("node: validate_and_consolidate"):
        estimate = state.get("estimate") or {}
        components: list[dict[str, Any]] = estimate.get("components", [])
        unbudgeted = [c["name"] for c in components if c.get("unbudgeted")]

        needs_review = bool(unbudgeted) or bool(state.get("errors")) or not components
        status = "needs_review" if needs_review else "validated"
        return {"status": status}


def route_after_validation(state: EstimationState) -> str:
    return "needs_review" if state.get("status") == "needs_review" else "validated"


__all__ = [
    "extract_requirements",
    "classify_components",
    "search_budgets",
    "generate_estimate",
    "validate_and_consolidate",
    "route_after_validation",
]
