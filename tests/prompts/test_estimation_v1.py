"""Template-level tests for the v1 estimation prompt.

These tests render the templates with controlled inputs and assert on the
output strings. They must not touch the LLM, the network, or any provider —
they are unit tests of the Jinja templates themselves.
"""

from __future__ import annotations

import pytest

import structlog

from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
    ReferenceProject,
)


def _request(
    description: str = (
        "A small CRM with auth, contacts, deals and a simple Kanban — "
        "MVP for a 5-person sales team."
    ),
    project_type: ProjectType = ProjectType.WEB_SAAS,
    detail_level: DetailLevel = DetailLevel.MEDIUM,
    output_format: OutputFormat = OutputFormat.PHASES_TABLE,
    reference_projects: list[ReferenceProject] | None = None,
) -> EstimationRequest:
    return EstimationRequest(
        description=description,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
        reference_projects=reference_projects,
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


# ---------------------------------------------------------------------------
# Bonus: v2 variation, reference_projects, and structlog event
# ---------------------------------------------------------------------------


def test_v2_system_diverges_from_v1_with_risk_register() -> None:
    v1_system, _ = render_estimation_prompt(_request(), version="v1")
    v2_system, _ = render_estimation_prompt(_request(), version="v2")
    assert "risk_register" in v2_system
    assert "risk_register" not in v1_system
    assert "20 % risk buffer" in v2_system


def test_reference_projects_loop_emits_each_title_when_present() -> None:
    refs = [
        ReferenceProject(
            title="Legacy CRM migration",
            summary="Lifted an old on-prem CRM into AWS over 4 months.",
            total_hours=320,
            total_cost_eur=20000,
            notes="Data migration was 40% of the effort.",
        ),
        ReferenceProject(
            title="Sales analytics dashboard",
            summary="Looker-style dashboard on top of the existing warehouse.",
            total_hours=120,
            total_cost_eur=7500,
        ),
    ]
    system_with, _ = render_estimation_prompt(_request(reference_projects=refs))
    system_without, _ = render_estimation_prompt(_request(reference_projects=None))

    assert "Similar past projects" in system_with
    assert "Legacy CRM migration" in system_with
    assert "Sales analytics dashboard" in system_with
    assert "Data migration was 40% of the effort." in system_with

    assert "Similar past projects" not in system_without
    assert "Legacy CRM migration" not in system_without


def test_loader_emits_prompt_rendered_event_with_hash_and_version() -> None:
    with structlog.testing.capture_logs() as logs:
        render_estimation_prompt(_request(), version="v1")
        render_estimation_prompt(_request(), version="v2")

    events = [e for e in logs if e.get("event") == "prompt_rendered"]
    assert len(events) == 2

    v1_event, v2_event = events
    assert v1_event["prompt_version"] == "v1"
    assert v2_event["prompt_version"] == "v2"

    for ev in events:
        assert isinstance(ev["content_hash"], str) and len(ev["content_hash"]) == 12
        assert ev["has_reference_projects"] is False
        assert ev["n_reference_projects"] == 0

    # Different prompt versions for the same request must produce different hashes.
    assert v1_event["content_hash"] != v2_event["content_hash"]
