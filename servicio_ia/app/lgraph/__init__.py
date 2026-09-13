"""LangGraph pipeline for the estimation flow (Session 13).

Five sequential nodes over a typed, checkpointed state: transcript ->
requirements -> components -> budget matches -> estimate -> validated
status. Reuses the S9-S12 retrieval (``app.generation.rag.retrieval``) and
costing (``app.agents.tools.calculate_estimate``) logic built in previous
sessions instead of reimplementing them.
"""
