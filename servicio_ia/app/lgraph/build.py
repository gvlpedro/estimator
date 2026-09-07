"""Wiring for the estimation graph: five sequential nodes, one conditional exit.

    START -> extract_requirements -> classify_components -> search_budgets
          -> generate_estimate -> validate_and_consolidate -> END

The last edge is conditional on ``state["status"]`` (set by
``validate_and_consolidate``): both branches currently terminate the graph,
but routing through named branches is what Session 13's retry/fallback/HITL
work (the live session) will hang off of.
"""

from langgraph.graph import END, START, StateGraph

from app.lgraph.nodes import (
    classify_components,
    extract_requirements,
    generate_estimate,
    route_after_validation,
    search_budgets,
    validate_and_consolidate,
)
from app.lgraph.state import EstimationState


def build_graph(checkpointer):
    builder = StateGraph(EstimationState)

    builder.add_node("extract_requirements", extract_requirements)
    builder.add_node("classify_components", classify_components)
    builder.add_node("search_budgets", search_budgets)  # sequential for now
    builder.add_node("generate_estimate", generate_estimate)
    builder.add_node("validate_and_consolidate", validate_and_consolidate)

    builder.add_edge(START, "extract_requirements")
    builder.add_edge("extract_requirements", "classify_components")
    builder.add_edge("classify_components", "search_budgets")
    builder.add_edge("search_budgets", "generate_estimate")
    builder.add_edge("generate_estimate", "validate_and_consolidate")
    builder.add_conditional_edges(
        "validate_and_consolidate",
        route_after_validation,
        {"needs_review": END, "validated": END},
    )

    return builder.compile(checkpointer=checkpointer)
