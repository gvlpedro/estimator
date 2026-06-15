"""Jinja2-backed prompt loader.

Templates live under ``app/prompts/<family>/<version>/`` and are rendered with
``StrictUndefined`` so any missing variable surfaces as an error during testing
instead of silently producing an empty string. The version is a positional
argument of :func:`render_estimation_prompt` so the caller (the FastAPI
endpoint) can swap "v1" → "v2" without touching anything else.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.estimation import EstimationRequest

PROMPTS_DIR = Path(__file__).resolve().parent

_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    autoescape=False,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = "v1",
) -> tuple[str, str]:
    """Render the (system, user) pair for the given estimation request.

    The version selects the subdirectory under ``app/prompts/estimation/``; the
    template file names (``system.j2`` / ``user.j2``) are fixed.
    """
    ctx = {
        "description": request.description,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
    }
    system = _env.get_template(f"estimation/{version}/system.j2").render(**ctx)
    user = _env.get_template(f"estimation/{version}/user.j2").render(**ctx)
    return system, user
