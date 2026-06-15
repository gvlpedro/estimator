from app.context.examples import CANONICAL_EXAMPLES
from app.services.evaluation import evaluate_estimation_structure


def test_well_formed_estimation_passes_all_checks() -> None:
    text = CANONICAL_EXAMPLES[0].estimation_markdown
    result = evaluate_estimation_structure(text, finish_reason="stop")
    assert result.has_title
    assert result.has_breakdown_table
    assert result.has_totals_section
    assert result.has_team_section
    assert result.has_duration_section
    assert result.declared_total_hours == CANONICAL_EXAMPLES[0].total_hours
    assert result.sum_row_hours == CANONICAL_EXAMPLES[0].total_hours
    assert result.hours_match is True
    assert result.declared_total_cost == CANONICAL_EXAMPLES[0].total_cost
    assert result.sum_row_cost == CANONICAL_EXAMPLES[0].total_cost
    assert result.cost_match is True
    assert result.finish_reason_ok is True
    assert result.score == 1.0
    assert result.issues == []


def test_well_formed_estimation_anthropic_end_turn_passes() -> None:
    text = CANONICAL_EXAMPLES[1].estimation_markdown
    result = evaluate_estimation_structure(text, finish_reason="end_turn")
    assert result.finish_reason_ok is True
    assert result.score == 1.0


def test_mismatched_total_hours_is_flagged() -> None:
    text = CANONICAL_EXAMPLES[0].estimation_markdown.replace(
        "**Total hours:** 200", "**Total hours:** 999"
    )
    result = evaluate_estimation_structure(text, finish_reason="stop")
    assert result.hours_match is False
    assert any("Total hours mismatch" in msg for msg in result.issues)
    assert result.score < 1.0


def test_missing_table_is_detected() -> None:
    text = "## Just a title\n\nNo table here, just prose."
    result = evaluate_estimation_structure(text, finish_reason="stop")
    assert result.has_title is True
    assert result.has_breakdown_table is False
    assert any("breakdown table" in msg for msg in result.issues)
    assert result.sum_row_hours is None


def test_finish_reason_length_fails_check() -> None:
    text = CANONICAL_EXAMPLES[2].estimation_markdown
    result = evaluate_estimation_structure(text, finish_reason="length")
    assert result.finish_reason_ok is False
    assert any("truncated" in msg.lower() or "finish_reason" in msg for msg in result.issues)
    assert result.score < 1.0


def test_empty_text_scores_zero_and_lists_all_issues() -> None:
    result = evaluate_estimation_structure("", finish_reason="stop")
    assert result.score < 0.5
    assert result.has_title is False
    assert result.has_breakdown_table is False
