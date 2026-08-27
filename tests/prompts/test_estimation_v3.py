"""Template-level tests for the v3 estimation prompt (citation grounding).

Same convention as ``tests/prompts/test_estimation_v1.py``: render the
templates with controlled inputs and assert on the output strings. No LLM,
no network — milliseconds, no red.
"""

from __future__ import annotations

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


def _retrieved_context() -> list[dict]:
    """Shaped like ``SearchResult.model_dump()`` (app/schemas/search.py)."""
    return [
        {
            "chunk_id": 101,
            "document_id": 9,
            "chunk_type": "line_item",
            "content": "Auth module estimated at 80h for a 5-person team.",
            "distance": 0.1234,
            "score": None,
            "metadata": {
                "budget_id": "B-2024-017",
                "component_id": "C-03",
                "client_sector": "fintech",
                "estimated_hours": 80,
                "complexity": "medium",
            },
        }
    ]


def test_chunk_id_appears_in_rendered_system_prompt_when_context_provided() -> None:
    system, _ = render_estimation_prompt(
        _request(), version="v3", retrieved_context=_retrieved_context()
    )
    assert "chunk_id 101" in system


def test_no_chunk_id_bracket_rendered_when_no_retrieved_context() -> None:
    """The grounding instructions block always mentions the *concept* of
    chunk_id (so the model knows what it means even with no context), but no
    concrete ``[chunk_id ...]`` evidence line should be rendered when there
    is nothing retrieved to ground against."""
    system, _ = render_estimation_prompt(_request(), version="v3", retrieved_context=None)
    assert "[chunk_id " not in system


def test_grounding_instructions_block_present() -> None:
    system, _ = render_estimation_prompt(
        _request(), version="v3", retrieved_context=_retrieved_context()
    )
    assert "grounded" in system
    assert "never invent" in system.lower()


def test_template_renders_successfully_with_no_retrieved_context() -> None:
    """Must not crash when there is nothing to ground against."""
    system, user = render_estimation_prompt(_request(), version="v3", retrieved_context=None)
    assert "grounded" in system
    assert len(system) > 0
    assert len(user) > 0


def test_v3_system_template_source_includes_v3_examples_not_v2() -> None:
    """v3 must be self-contained: no lingering reference to v2's examples
    partial (a copy-paste hazard when new versions are branched off v2).

    Asserts on the raw template *source*, not the rendered output — a
    ``{% include %}`` directive is resolved before rendering, so the literal
    path string never survives into rendered text either way.
    """
    from app.prompts.loader import PROMPTS_DIR

    source = (PROMPTS_DIR / "estimation" / "v3" / "system.j2").read_text()
    assert "estimation/v2/examples.j2" not in source
    assert "estimation/v3/examples.j2" in source
