"""Wiring for the multi-agent estimation graph: a hand-built supervisor.

    START -> supervisor --(Command)--> requirements_extractor --> supervisor
                          --(Command)--> budget_searcher        --> supervisor
                          --(Command)--> estimate_generator     --> supervisor
                          --(Command)--> coherence_validator    --> supervisor
                          --(Command)--> human_review_gate --(Command)--> finalize -> END

``supervisor`` and ``human_review_gate`` route dynamically via
``Command(goto=...)`` (see ``nodes.py``) -- no ``create_supervisor``, no
``add_conditional_edges`` for the routing itself, so every decision is a
plain node call and shows up in the trace like any other agent turn. Workers
return plain state updates and flow back to ``supervisor`` through ordinary
static edges, which is what lets the supervisor decide the *next* step from
the state each worker just updated.
"""

from langgraph.graph import END, START, StateGraph

from app.lgraph.nodes import (
    budget_searcher,
    coherence_validator,
    estimate_generator,
    finalize,
    human_review_gate,
    requirements_extractor,
    supervisor,
)
from app.lgraph.state import EstimationState


def build_graph(checkpointer):
    builder = StateGraph(EstimationState)

    builder.add_node("supervisor", supervisor)
    builder.add_node("requirements_extractor", requirements_extractor)
    builder.add_node("budget_searcher", budget_searcher)  # sequential for now
    builder.add_node("estimate_generator", estimate_generator)
    builder.add_node("coherence_validator", coherence_validator)
    builder.add_node("human_review_gate", human_review_gate)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "supervisor")
    builder.add_edge("requirements_extractor", "supervisor")
    builder.add_edge("budget_searcher", "supervisor")
    builder.add_edge("estimate_generator", "supervisor")
    builder.add_edge("coherence_validator", "supervisor")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)
