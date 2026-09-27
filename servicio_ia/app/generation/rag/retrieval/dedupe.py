"""Collapse a ranked result list to at most one chunk per budget.

The seeded corpus's chunking template (see ``app/ingest/chunker.py``) prepends
the same project summary/sector/stack header to every component of a budget,
so a single budget's 3-6 chunks read as near-duplicates of one another. When
one budget is the clear best match for a query, its own redundant chunks can
dominate 3-4 of a top-5 result — crowding out other, genuinely different
budgets that would otherwise place. Deduping by ``budget_id`` restores
diversity across budgets in the final result set, which is what a search
result is actually supposed to surface: distinct candidates, not one
candidate's internal repetition.
"""

from sqlalchemy import Row


def dedupe_by_budget(ranked: list[tuple[Row, float]], k: int) -> list[tuple[Row, float]]:
    """Keep each budget's best-ranked chunk, then truncate to k.

    ``ranked`` MUST already be sorted best-first — that precondition is what
    makes "keep the first occurrence of each budget_id" correct: iterating in
    order and using ``setdefault`` means the first time a ``budget_id`` is
    seen is necessarily its highest-ranked occurrence, so every later
    duplicate of that same budget can simply be skipped without ever
    comparing scores explicitly. Getting this precondition wrong (an
    unsorted or worst-first list) would silently keep the wrong chunk per
    budget with no error to signal it.
    """
    best_by_budget: dict[str, tuple[Row, float]] = {}
    for row, score in ranked:
        best_by_budget.setdefault(row.meta["budget_id"], (row, score))

    return list(best_by_budget.values())[:k]
