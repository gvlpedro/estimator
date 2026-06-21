"""Jinja2-backed prompt loader.

Templates live under ``app/prompts/<family>/<version>/`` and are rendered with
``StrictUndefined`` so any missing variable surfaces as an error during testing
instead of silently producing an empty string. The version is a positional
argument of :func:`render_estimation_prompt` so the caller (the FastAPI
endpoint) can swap "v1" → "v2" without touching anything else.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.estimation import EstimationRequest
from app.schemas.log import PromptRendered
from app.services.sessions import ProjectMetadata

PROMPTS_DIR = Path(__file__).resolve().parent

_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    autoescape=False,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)


def _content_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = "v1",
    project_metadata: ProjectMetadata | None = None,
) -> tuple[str, str]:
    """Render the (system, user) pair for the given estimation request.

    The version selects the subdirectory under ``app/prompts/estimation/``; the
    template file names (``system.j2`` / ``user.j2``) are fixed.

    ``project_metadata`` is the running blob of facts the assistant has
    accumulated for the current session. It is injected into the
    ``<project_metadata>`` block of the system prompt; when ``None`` or empty,
    the block renders empty but is still present (so the LLM always sees the
    same scaffold).

    Emits a ``prompt_rendered`` structlog event with the version, a 12-char
    SHA-256 of the rendered content, and flags indicating whether
    ``reference_projects`` / ``project_metadata`` were injected. Useful for
    debugging which prompt actually went out the door in production.
    """
    metadata_for_template = (
        project_metadata
        if project_metadata is not None and _metadata_has_any_field(project_metadata)
        else None
    )
    ctx = {
        "description": request.description,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        "reference_projects": (
            [rp.model_dump() for rp in request.reference_projects]
            if request.reference_projects
            else None
        ),
        "project_metadata": metadata_for_template,
    }
    system = _env.get_template(f"estimation/{version}/system.j2").render(**ctx)
    user = _env.get_template(f"estimation/{version}/user.j2").render(**ctx)

    PromptRendered(
        prompt_version=version,
        content_hash=_content_hash(system, user),
        system_chars=len(system),
        user_chars=len(user),
        has_reference_projects=bool(ctx["reference_projects"]),
        n_reference_projects=len(ctx["reference_projects"] or []),
        has_project_metadata=metadata_for_template is not None,
    ).emit()

    return system, user


def _metadata_has_any_field(metadata: ProjectMetadata) -> bool:
    return bool(
        metadata.project_name
        or metadata.assumed_team_size
        or metadata.mentioned_technologies
        or metadata.agreed_scope
    )
