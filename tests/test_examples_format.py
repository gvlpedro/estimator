import json

import pytest

from app.context.examples import (
    CANONICAL_EXAMPLES,
    format_examples_for_prompt,
    select_examples,
)


def test_format_markdown_contains_table_header() -> None:
    out = format_examples_for_prompt(select_examples(2), fmt="markdown")
    assert "--- EXAMPLE 1 ---" in out
    assert "--- EXAMPLE 2 ---" in out
    assert "| Task | Hours | Cost (EUR) |" in out


def test_format_json_is_valid_json() -> None:
    out = format_examples_for_prompt(select_examples(3), fmt="json")
    assert out.startswith("Reference examples (JSON):")
    payload = json.loads(out.split("Reference examples (JSON):", 1)[1])
    assert isinstance(payload, list) and len(payload) == 3
    first = payload[0]
    assert {"meeting_summary", "title", "breakdown", "totals", "team", "duration_weeks"} <= set(
        first.keys()
    )
    assert first["totals"]["hours"] == CANONICAL_EXAMPLES[0].total_hours
    assert first["totals"]["cost_eur"] == CANONICAL_EXAMPLES[0].total_cost


def test_format_narrative_mentions_hours_and_weeks() -> None:
    out = format_examples_for_prompt(select_examples(1), fmt="narrative")
    ex = CANONICAL_EXAMPLES[0]
    assert f"{ex.total_hours} hours" in out
    assert f"{ex.duration_weeks} weeks" in out
    assert ex.title in out


def test_select_examples_zero_returns_empty() -> None:
    assert select_examples(0) == []
    assert format_examples_for_prompt([], fmt="markdown") == ""
    assert format_examples_for_prompt([], fmt="json") == ""
    assert format_examples_for_prompt([], fmt="narrative") == ""


def test_select_examples_caps_at_available() -> None:
    assert len(select_examples(99)) == len(CANONICAL_EXAMPLES)


def test_select_examples_negative_returns_empty() -> None:
    assert select_examples(-1) == []


def test_canonical_examples_have_calibrated_totals() -> None:
    for ex in CANONICAL_EXAMPLES:
        sum_h = sum(h for _, h, _ in ex.breakdown)
        sum_c = sum(c for _, _, c in ex.breakdown)
        assert sum_h == ex.total_hours, ex.title
        assert sum_c == ex.total_cost, ex.title


def test_format_unknown_raises() -> None:
    with pytest.raises(ValueError):
        format_examples_for_prompt(select_examples(1), fmt="yaml")  # type: ignore[arg-type]
