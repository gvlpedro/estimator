"""Unit tests for dedupe_by_budget: pure per-budget collapsing logic.

No I/O, no SQL, no mocks needed — a minimal stand-in for a SQLAlchemy Row
(anything with a `.meta` dict) is enough to exercise the real logic.
"""

from types import SimpleNamespace

from app.generation.rag.retrieval.dedupe import dedupe_by_budget


def _row(budget_id: str) -> SimpleNamespace:
    return SimpleNamespace(meta={"budget_id": budget_id})


def test_no_duplicate_budgets_is_a_noop_besides_k_truncation():
    ranked = [
        (_row("BUD-001"), 0.9),
        (_row("BUD-002"), 0.8),
        (_row("BUD-003"), 0.7),
    ]

    deduped = dedupe_by_budget(ranked, k=5)

    assert deduped == ranked


def test_repeated_budget_keeps_only_its_first_best_ranked_occurrence():
    best = _row("BUD-001")
    worse_dupe_1 = _row("BUD-001")
    worse_dupe_2 = _row("BUD-001")
    other = _row("BUD-002")
    ranked = [
        (best, 0.95),
        (worse_dupe_1, 0.9),
        (other, 0.85),
        (worse_dupe_2, 0.8),
    ]

    deduped = dedupe_by_budget(ranked, k=5)

    assert deduped == [(best, 0.95), (other, 0.85)]


def test_fewer_distinct_budgets_than_k_returns_only_the_distinct_ones():
    ranked = [
        (_row("BUD-001"), 0.9),
        (_row("BUD-001"), 0.8),
        (_row("BUD-002"), 0.7),
    ]

    deduped = dedupe_by_budget(ranked, k=5)

    assert len(deduped) == 2
    assert [row.meta["budget_id"] for row, _ in deduped] == ["BUD-001", "BUD-002"]


def test_more_distinct_budgets_than_k_truncates_to_top_k():
    ranked = [
        (_row("BUD-001"), 0.9),
        (_row("BUD-002"), 0.8),
        (_row("BUD-003"), 0.7),
        (_row("BUD-004"), 0.6),
    ]

    deduped = dedupe_by_budget(ranked, k=2)

    assert [row.meta["budget_id"] for row, _ in deduped] == ["BUD-001", "BUD-002"]
