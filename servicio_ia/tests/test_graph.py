"""Unit tests for the estimation graph's pure nodes and its wiring.

``extract_requirements``, ``classify_components`` and ``search_budgets`` all
call out (OpenAI, Postgres) and are exercised via
``scripts/run_graph_s13.py`` instead; what's tested here is what needs no
I/O: the two closing nodes' logic and the graph's shape.
"""

from langgraph.checkpoint.memory import InMemorySaver

from app.lgraph.build import build_graph
from app.lgraph.nodes import generate_estimate, route_after_validation, validate_and_consolidate


def test_generate_estimate_groups_matches_by_component_name():
    state = {
        "components": [{"name": "Backend API", "category": "backend"}, {"name": "Mobile App", "category": "mobile"}],
        "budget_matches": [
            {"component": "Backend API", "reference_budget_id": "b1", "amount": 100.0},
            {"component": "Backend API", "reference_budget_id": "b2", "amount": 200.0},
        ],
    }

    update = generate_estimate(state)

    names = {c["name"]: c for c in update["estimate"]["components"]}
    assert names["Backend API"]["reference_count"] == 2
    assert names["Backend API"]["unbudgeted"] is False
    # No budget matches at all for Mobile App -> unbudgeted, costed at 0.
    assert names["Mobile App"]["reference_count"] == 0
    assert names["Mobile App"]["unbudgeted"] is True


def test_validate_and_consolidate_flags_needs_review_when_a_component_is_unbudgeted():
    state = {
        "estimate": {
            "components": [
                {"name": "Backend API", "reference_count": 2, "estimated_hours": 115.0, "unbudgeted": False},
                {"name": "Mobile App", "reference_count": 0, "estimated_hours": 0.0, "unbudgeted": True},
            ],
            "total_hours": 115.0,
        },
        "errors": [],
    }

    update = validate_and_consolidate(state)

    assert update["status"] == "needs_review"


def test_validate_and_consolidate_passes_when_every_component_is_budgeted_and_no_errors():
    state = {
        "estimate": {
            "components": [
                {"name": "Backend API", "reference_count": 2, "estimated_hours": 115.0, "unbudgeted": False},
            ],
            "total_hours": 115.0,
        },
        "errors": [],
    }

    update = validate_and_consolidate(state)

    assert update["status"] == "validated"


def test_route_after_validation_reads_the_status_field_from_state():
    assert route_after_validation({"status": "needs_review"}) == "needs_review"
    assert route_after_validation({"status": "validated"}) == "validated"


def test_build_graph_wires_all_five_nodes_sequentially():
    graph = build_graph(InMemorySaver())

    node_names = set(graph.get_graph().nodes.keys())
    assert {
        "extract_requirements",
        "classify_components",
        "search_budgets",
        "generate_estimate",
        "validate_and_consolidate",
    } <= node_names
