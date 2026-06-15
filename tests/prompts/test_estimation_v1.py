"""Template-level tests for the v1 estimation prompt.

These tests render the templates with controlled inputs and assert on the
output strings. They must not touch the LLM, the network, or any provider —
they are unit tests of the Jinja templates themselves.
"""

from __future__ import annotations

import pytest

from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)


def _request(
    description: str = (
        "A small CRM with auth, contacts, deals and a simple Kanban — "
        "MVP for a 5-person sales team."
    ),
    project_type: ProjectType = ProjectType.WEB_SAAS,
    detail_level: DetailLevel = DetailLevel.MEDIUM,
    output_format: OutputFormat = OutputFormat.PHASES_TABLE,
) -> EstimationRequest:
    return EstimationRequest(
        description=description,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
    )


def test_user_template_includes_description_verbatim() -> None:
    description = (
        "Internal dashboard for warehouse staff to track inbound shipments, "
        "scan barcodes from the tablet camera and confirm receipt in SAP."
    )
    _, user = render_estimation_prompt(_request(description=description))
    assert "<project_description>" in user
    assert description in user
    assert "</project_description>" in user


@pytest.mark.parametrize(
    ("output_format", "expect_keyword"),
    [
        (OutputFormat.PHASES_TABLE, True),
        (OutputFormat.LINE_ITEMS, False),
        (OutputFormat.NARRATIVE, False),
    ],
)
def test_system_template_phases_table_keyword_is_conditional(
    output_format: OutputFormat, expect_keyword: bool
) -> None:
    system, _ = render_estimation_prompt(_request(output_format=output_format))
    has_keyword = "confidence_pct" in system
    assert has_keyword is expect_keyword


def test_system_template_detailed_adds_assumptions_block() -> None:
    detailed_system, _ = render_estimation_prompt(
        _request(detail_level=DetailLevel.DETAILED)
    )
    summary_system, _ = render_estimation_prompt(
        _request(detail_level=DetailLevel.SUMMARY)
    )
    assert "Assumptions" in detailed_system
    assert "Assumptions" not in summary_system


def test_narrative_format_forbids_tables_instruction_present() -> None:
    system, _ = render_estimation_prompt(_request(output_format=OutputFormat.NARRATIVE))
    assert "Do not use tables" in system
