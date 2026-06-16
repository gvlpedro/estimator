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

import structlog
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.estimation import EstimationRequest

log = structlog.get_logger()

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
) -> tuple[str, str]:
    """Render the (system, user) pair for the given estimation request.

    The version selects the subdirectory under ``app/prompts/estimation/``; the
    template file names (``system.j2`` / ``user.j2``) are fixed.

    Emits a ``prompt_rendered`` structlog event with the version, a 12-char
    SHA-256 of the rendered content, and a flag indicating whether
    ``reference_projects`` was injected. Useful for debugging which prompt
    actually went out the door in production.
    """
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
    }
    system = _env.get_template(f"estimation/{version}/system.j2").render(**ctx)
    user = _env.get_template(f"estimation/{version}/user.j2").render(**ctx)

    log.info(
        "prompt_rendered",
        prompt_version=version,
        content_hash=_content_hash(system, user),
        system_chars=len(system),
        user_chars=len(user),
        has_reference_projects=bool(ctx["reference_projects"]),
        n_reference_projects=len(ctx["reference_projects"] or []),
    )

    return system, user
