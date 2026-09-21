"""Unit tests for the multi-agent estimation graph: supervisor, gate, tool privilege.

``requirements_extractor``, ``budget_searcher``, ``estimate_generator`` and
``coherence_validator`` all call out (OpenAI, Postgres) and are exercised via
``scripts/run_multiagent_s14.py`` instead; what's tested here is what needs
no I/O: the supervisor's routing logic, the human-review gate's pause/resume
mechanics (via a real, minimal compiled graph -- no mocks), least-privilege
tool access, and the closing ``finalize`` logic.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.lgraph.build import build_graph
from app.lgraph.nodes import finalize, human_review_gate, supervisor
from app.lgraph.state import EstimationState
from app.lgraph.tool_access import AGENT_TOOL_PRIVILEGES, ToolAccessDenied, call_tool


# ---------------------------------------------------------------------------
# Supervisor routing
# ---------------------------------------------------------------------------


def test_supervisor_routes_to_requirements_extractor_first():
    state = {"agent_contributions": []}

    command = supervisor(state)

    assert command.goto == "requirements_extractor"


def test_supervisor_routes_through_agents_in_declared_order():
    already_run = []
    expected_order = [
        "requirements_extractor",
        "budget_searcher",
        "estimate_generator",
        "coherence_validator",
        "human_review_gate",
    ]

    for expected_next in expected_order:
        command = supervisor({"agent_contributions": already_run})
        assert command.goto == expected_next
        already_run = already_run + [{"agent": expected_next, "summary": "done"}]


def test_supervisor_ignores_empty_worker_output_and_still_advances():
    # budget_searcher legitimately found nothing for any component -- an
    # empty budget_matches must not look like "hasn't run yet".
    state = {
        "agent_contributions": [
            {"agent": "requirements_extractor", "summary": "..."},
            {"agent": "budget_searcher", "summary": "0 matches"},
        ],
        "budget_matches": [],
    }

    command = supervisor(state)

    assert command.goto == "estimate_generator"


# ---------------------------------------------------------------------------
# Least-privilege tool access
# ---------------------------------------------------------------------------


def test_every_declared_agent_has_a_privilege_entry():
    for agent in ("requirements_extractor", "budget_searcher", "estimate_generator", "coherence_validator"):
        assert agent in AGENT_TOOL_PRIVILEGES


def test_requirements_extractor_has_no_business_tool_privilege():
    assert AGENT_TOOL_PRIVILEGES["requirements_extractor"] == frozenset()


async def test_call_tool_denies_a_tool_outside_the_agents_privilege_set():
    with pytest.raises(ToolAccessDenied):
        await call_tool("requirements_extractor", "search_budgets", lambda: {"ok": True})


async def test_call_tool_allows_a_declared_sync_tool_and_returns_its_result():
    result = await call_tool("estimate_generator", "calculate_estimate", lambda args: {"echo": args}, {"x": 1})

    assert result == {"echo": {"x": 1}}


async def test_call_tool_allows_a_declared_async_tool_and_returns_its_result():
    async def fake_search(args):
        return {"echo": args}

    result = await call_tool("budget_searcher", "search_budgets", fake_search, {"x": 1})

    assert result == {"echo": {"x": 1}}


async def test_call_tool_propagates_the_underlying_tool_error():
    def boom(args):
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        await call_tool("estimate_generator", "calculate_estimate", boom, {})


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


def test_finalize_marks_validated_when_no_human_decision_and_no_issues():
    state = {"validation": {"issues": []}, "errors": []}

    update = finalize(state)

    assert update["status"] == "validated"


def test_finalize_marks_needs_review_when_validation_has_issues_and_no_human_decision():
    state = {"validation": {"issues": ["out of range"]}, "errors": []}

    update = finalize(state)

    assert update["status"] == "needs_review"


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [("approved", "validated"), ("adjusted", "validated"), ("rejected", "rejected")],
)
def test_finalize_maps_human_decision_outcome_to_status(outcome, expected_status):
    state = {"human_decision": {"outcome": outcome}, "validation": {"issues": ["low confidence"]}}

    update = finalize(state)

    assert update["status"] == expected_status


# ---------------------------------------------------------------------------
# human_review_gate: real interrupt/resume mechanics on a minimal graph
# ---------------------------------------------------------------------------


def _build_gate_only_graph():
    builder = StateGraph(EstimationState)
    builder.add_node("human_review_gate", human_review_gate)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "human_review_gate")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=InMemorySaver())


async def test_human_review_gate_skips_straight_to_finalize_when_confidence_is_high():
    graph = _build_gate_only_graph()
    config = {"configurable": {"thread_id": "gate-high-confidence"}}

    result = await graph.ainvoke(
        {
            "estimate": {"components": [], "total_hours": 10.0},
            "confidence": 0.95,
            "validation": {"in_historical_range": True, "no_precedent": False, "issues": []},
        },
        config,
    )

    assert "__interrupt__" not in result
    assert result["status"] == "validated"


async def test_human_review_gate_pauses_and_resumes_with_the_human_decision():
    graph = _build_gate_only_graph()
    config = {"configurable": {"thread_id": "gate-low-confidence"}}

    paused = await graph.ainvoke(
        {
            "estimate": {"components": [], "total_hours": 10.0},
            "confidence": 0.2,
            "validation": {"in_historical_range": False, "no_precedent": True, "issues": ["no precedent"]},
        },
        config,
    )

    assert "__interrupt__" in paused
    review = paused["__interrupt__"][0].value
    assert review["reason"] == "low_confidence_estimate"

    pending = await graph.aget_state(config)
    assert pending.next == ("human_review_gate",)

    resumed = await graph.ainvoke(Command(resume={"outcome": "approved"}), config)

    assert resumed["status"] == "validated"
    assert resumed["human_decision"] == {"outcome": "approved"}


# ---------------------------------------------------------------------------
# Graph shape
# ---------------------------------------------------------------------------


def test_build_graph_wires_the_supervisor_and_every_agent():
    graph = build_graph(InMemorySaver())

    node_names = set(graph.get_graph().nodes.keys())
    assert {
        "supervisor",
        "requirements_extractor",
        "budget_searcher",
        "estimate_generator",
        "coherence_validator",
        "human_review_gate",
        "finalize",
    } <= node_names
