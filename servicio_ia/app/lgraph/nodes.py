"""The multi-agent estimation graph: a hand-built supervisor + four specialists.

Each specialist is a pure function of the state: it reads ``state`` and
returns a partial update, never mutates ``state`` in place, and never holds
state across calls (the OpenAI client and DB session are opened per call,
mirroring the dependency lifecycle already used by
``scripts/run_agent_s12.py``). Every specialist appends one entry to the
``agent_contributions`` accumulator -- that list is both the audit trail and
the supervisor's own memory of who has already run.

Least privilege: a specialist calls a business tool only through
``tool_access.call_tool``, which enforces ``AGENT_TOOL_PRIVILEGES`` and logs
the attempt. ``requirements_extractor`` calls none at all -- it only reasons
over text, per its declared (empty) privilege set.

``budget_searcher`` and ``estimate_generator`` reuse the S9-S10 retrieval
pipeline and the S12 costing/validation functions verbatim
(``app.generation.rag`` / ``app.agents.tools``) instead of reimplementing
them for the graph.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Literal

import logfire
import structlog
from langgraph.types import Command, interrupt
from openai import OpenAI

from app.agents.tools import calculate_estimate, validate_estimate
from app.agents.tools import search_budgets as retrieve_budget_for
from app.dependencies import get_embedder, get_reranker
from app.lgraph.state import AgentContribution, BudgetMatch, EstimationState, TokenUsage
from app.lgraph.tool_access import call_tool
from app.observability import record_token_usage
from app.storage.db import session_factory

log = structlog.get_logger()

GRAPH_MODEL = os.getenv("GRAPH_MODEL", "gpt-5-mini")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

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


def _has_run(state: EstimationState, agent: str) -> bool:
    return any(c["agent"] == agent for c in state.get("agent_contributions", []))


def _contribution(agent: str, summary: str) -> list[AgentContribution]:
    return [AgentContribution(agent=agent, summary=summary)]


# ---------------------------------------------------------------------------
# Specialists (workers) -- no tool privilege beyond what's declared for them
# ---------------------------------------------------------------------------


def requirements_extractor(state: EstimationState) -> dict[str, Any]:
    """Transcript -> requirements -> components. Model only, no business tool."""
    agent = "requirements_extractor"
    with logfire.span(f"agent: {agent}"):
        log.info("agent_turn_started", agent=agent, tools=[])

        requirements_response = _get_client().responses.create(
            model=GRAPH_MODEL,
            instructions=EXTRACT_REQUIREMENTS_PROMPT,
            input=state["transcript"],
            text={"format": _REQUIREMENTS_SCHEMA},
        )
        requirements = json.loads(requirements_response.output_text)["requirements"]

        components_response = _get_client().responses.create(
            model=GRAPH_MODEL,
            instructions=CLASSIFY_COMPONENTS_PROMPT,
            input=json.dumps({"requirements": requirements}, ensure_ascii=False),
            text={"format": _COMPONENTS_SCHEMA},
        )
        components = json.loads(components_response.output_text)["components"]

        summary = f"{len(requirements)} requirement(s) -> {len(components)} component(s)"
        log.info("agent_turn_finished", agent=agent, summary=summary)
        token_usage: list[TokenUsage] = [
            TokenUsage(
                agent=agent,
                call="extract_requirements",
                input_tokens=requirements_response.usage.input_tokens,
                output_tokens=requirements_response.usage.output_tokens,
            ),
            TokenUsage(
                agent=agent,
                call="classify_components",
                input_tokens=components_response.usage.input_tokens,
                output_tokens=components_response.usage.output_tokens,
            ),
        ]
        # Also feed the per-HTTP-request accumulator (Session 16): the state
        # list above is this graph run's own audit trail and survives
        # replay/inspection; the ContextVar in app.observability is what the
        # request middleware reads to emit one request_completed event,
        # regardless of which node happened to call a model.
        for usage in token_usage:
            record_token_usage(
                agent=usage["agent"],
                call=usage["call"],
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
            )
        return {
            "requirements": requirements,
            "components": components,
            "agent_contributions": _contribution(agent, summary),
            "token_usage": token_usage,
        }


async def budget_searcher(state: EstimationState) -> dict[str, Any]:
    """One search_budgets call per component, sequential -- fan-out is the live-session case."""
    agent = "budget_searcher"
    matches: list[BudgetMatch] = []
    errors: list[str] = []

    with logfire.span(f"agent: {agent}"):
        log.info("agent_turn_started", agent=agent, tools=["search_budgets"])
        embedder = get_embedder()
        reranker = get_reranker()
        async with session_factory() as session:
            for component in state["components"]:
                query = f"{component['name']} ({component['category']})"
                try:
                    result = await call_tool(
                        agent,
                        "search_budgets",
                        retrieve_budget_for,
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

        summary = f"{len(matches)} historical match(es) across {len(state['components'])} component(s)"
        log.info("agent_turn_finished", agent=agent, summary=summary)

    return {
        "budget_matches": matches,
        "errors": errors,
        "agent_contributions": _contribution(agent, summary),
    }


async def estimate_generator(state: EstimationState) -> dict[str, Any]:
    agent = "estimate_generator"
    with logfire.span(f"agent: {agent}"):
        log.info("agent_turn_started", agent=agent, tools=["calculate_estimate"])

        by_component: dict[str, list[float]] = {c["name"]: [] for c in state["components"]}
        for match in state["budget_matches"]:
            by_component.setdefault(match["component"], []).append(match["amount"])

        args = {
            "components": [
                {"name": name, "reference_amounts": amounts}
                for name, amounts in by_component.items()
            ]
        }
        estimate = await call_tool(agent, "calculate_estimate", calculate_estimate, args)

        log.info("agent_turn_finished", agent=agent, summary=estimate["summary"])
        return {
            "estimate": estimate,
            "agent_contributions": _contribution(agent, estimate["summary"]),
        }


async def coherence_validator(state: EstimationState) -> dict[str, Any]:
    agent = "coherence_validator"
    with logfire.span(f"agent: {agent}"):
        log.info("agent_turn_started", agent=agent, tools=["validate_estimate"])

        args = {
            "estimate": state.get("estimate") or {},
            "budget_matches": state.get("budget_matches", []),
        }
        validation = await call_tool(agent, "validate_estimate", validate_estimate, args)

        log.info("agent_turn_finished", agent=agent, summary=validation["summary"])
        return {
            "validation": validation,
            "confidence": validation["confidence"],
            "agent_contributions": _contribution(agent, validation["summary"]),
        }


# ---------------------------------------------------------------------------
# Supervisor -- hand-built with StateGraph + Command, no create_supervisor
# ---------------------------------------------------------------------------


def supervisor(
    state: EstimationState,
) -> Command[
    Literal[
        "requirements_extractor",
        "budget_searcher",
        "estimate_generator",
        "coherence_validator",
        "human_review_gate",
    ]
]:
    """Decide the next specialist from what has already run, not business-data shape.

    Checking ``agent_contributions`` (not e.g. ``bool(budget_matches)``)
    means a component search that legitimately found nothing doesn't get
    mistaken for "hasn't run yet" and looped forever.
    """
    with logfire.span("agent: supervisor"):
        if not _has_run(state, "requirements_extractor"):
            goto = "requirements_extractor"
        elif not _has_run(state, "budget_searcher"):
            goto = "budget_searcher"
        elif not _has_run(state, "estimate_generator"):
            goto = "estimate_generator"
        elif not _has_run(state, "coherence_validator"):
            goto = "coherence_validator"
        else:
            goto = "human_review_gate"

        log.info("supervisor_routed", goto=goto)
        return Command(goto=goto)


# ---------------------------------------------------------------------------
# Human-in-the-loop gate + finalize
# ---------------------------------------------------------------------------


def _needs_human_review(state: EstimationState) -> bool:
    confidence = state.get("confidence")
    validation = state.get("validation") or {}
    return (
        confidence is None
        or confidence < CONFIDENCE_THRESHOLD
        or validation.get("no_precedent", False)
        or not validation.get("in_historical_range", True)
    )


def human_review_gate(state: EstimationState) -> Command[Literal["finalize"]]:
    """Pause with ``interrupt()`` when the estimate isn't trustworthy enough to ship as-is.

    The trigger folds three signals into the single ``confidence`` score
    computed by ``validate_estimate`` (unbudgeted components, no historical
    precedent, out-of-range total), so this only needs one threshold check
    plus the two raw flags for cases confidence alone might mask.
    """
    with logfire.span("agent: human_review_gate"):
        if not _needs_human_review(state):
            log.info("human_review_skipped", confidence=state.get("confidence"))
            return Command(goto="finalize")

        # On resume, LangGraph re-runs this node from the top up to the
        # interrupt() call (which this time returns the resume value instead
        # of pausing again) -- so this log line fires twice per human review,
        # once on the initial pause and once on resume. Expected, not a bug.
        log.info(
            "human_review_requested",
            confidence=state.get("confidence"),
            estimation_id=state.get("estimation_id"),
        )
        decision = interrupt(
            {
                "reason": "low_confidence_estimate",
                "estimate": state.get("estimate"),
                "confidence": state.get("confidence"),
                "validation": state.get("validation"),
            }
        )
        log.info("human_review_resolved", decision=decision)
        return Command(
            goto="finalize",
            update={
                "human_decision": decision,
                "agent_contributions": _contribution(
                    "human_review_gate", f"human decision: {decision}"
                ),
            },
        )


_DECISION_STATUS = {"approved": "validated", "adjusted": "validated", "rejected": "rejected"}


def finalize(state: EstimationState) -> dict[str, Any]:
    with logfire.span("agent: finalize"):
        decision = state.get("human_decision")
        if decision is not None:
            outcome = decision.get("outcome", "approved") if isinstance(decision, dict) else "approved"
            status = _DECISION_STATUS.get(outcome, "needs_review")
        else:
            validation = state.get("validation") or {}
            status = "needs_review" if (validation.get("issues") or state.get("errors")) else "validated"

        log.info("estimation_finalized", status=status)
        return {"status": status}


__all__ = [
    "requirements_extractor",
    "budget_searcher",
    "estimate_generator",
    "coherence_validator",
    "supervisor",
    "human_review_gate",
    "finalize",
    "CONFIDENCE_THRESHOLD",
]
