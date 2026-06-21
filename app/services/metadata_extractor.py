"""LLM-based extractor that keeps ``ProjectMetadata`` in sync after each turn.

Why a second LLM call instead of regex / NER over the transcript
----------------------------------------------------------------
See the *Project Metadata Injection* section of the README for the long form.
Short version: facts in a free-text conversation are rarely literal. A user
saying "we have a Spring Boot service" implies Java; "the dashboard runs on
Next.js" implies React and JavaScript. Surface-level pattern matching cannot
collapse a framework onto its underlying language, dedupe stylistic variants
("Postgres" vs "PostgreSQL"), or rewrite ``agreed_scope`` after a refinement.
The LLM gives us comprehension, not just matching.

This module is intentionally thin: it builds the extractor prompt, calls
``LLMWrapper.complete_structured(response_model=ProjectMetadata)`` and returns
the new metadata. On failure it logs and returns ``current`` unchanged — a
broken extractor must never break the estimation pipeline.
"""

from __future__ import annotations

from app.schemas.log import ProjectMetadataExtractionFailed, ProjectMetadataUpdated
from app.services.llm_wrapper import LLMWrapper
from app.services.sessions import ProjectMetadata

EXTRACTOR_SYSTEM_PROMPT = """You extract structured project facts from one turn
of a software-estimation conversation.

You will receive:
1. The project metadata accumulated so far (may be empty on the first turn).
2. The latest user transcription (possibly enriched with attachment text).
3. The latest assistant estimation reply (JSON of the structured result).

Return a single JSON object that conforms to the ProjectMetadata schema:

- project_name: short human name of the project, or null.
- assumed_team_size: integer team size if stated or strongly implied, else null.
- mentioned_technologies: deduplicated list of technologies, frameworks,
  languages, databases, cloud providers, etc. that have appeared in the
  conversation so far.
- agreed_scope: 1-3 sentences summarising what user and assistant have
  converged on; updated incrementally.

Rules:

1. MERGE, do not replace. Carry over every previously-known fact unless the
   new turn explicitly contradicts it. When a refinement appears (e.g. "we are
   six people, not three"), prefer the latest value.
2. Infer transitively for technologies. Add the underlying language and the
   obvious siblings: Spring Boot implies Java; Next.js implies React and
   JavaScript; FastAPI implies Python; .NET MAUI implies C#; Kotlin runs on
   the JVM. Be conservative: only add what a reader would obviously infer.
3. mentioned_technologies must be deduplicated (case-insensitive) and sorted
   alphabetically.
4. If a field is unknown, return null (for scalars) or [] (for lists). Do not
   invent facts that are not supported by the conversation."""


def extract_project_metadata(
    *,
    wrapper: LLMWrapper,
    current: ProjectMetadata,
    user_message: str,
    assistant_reply: str,
) -> ProjectMetadata:
    """Return the updated ``ProjectMetadata`` for the session.

    Falls back to ``current`` and logs on any error so the caller's estimation
    response is never blocked by the extractor.
    """
    user_msg = (
        "## Current project_metadata (merge with this; do not drop facts)\n"
        f"{current.model_dump_json(indent=2)}\n\n"
        "## Latest user turn\n"
        f"{user_message}\n\n"
        "## Latest assistant reply (JSON)\n"
        f"{assistant_reply}\n\n"
        "Return the updated ProjectMetadata as JSON."
    )

    try:
        completion = wrapper.complete_structured(
            system_prompt=EXTRACTOR_SYSTEM_PROMPT,
            user_message=user_msg,
            response_model=ProjectMetadata,
        )
    except Exception as exc:  # noqa: BLE001 — extractor must not break estimation
        ProjectMetadataExtractionFailed(
            error=str(exc),
            error_type=type(exc).__name__,
        ).emit()
        return current

    new_metadata = completion.result
    ProjectMetadataUpdated(
        had_project_name=bool(current.project_name),
        has_project_name=bool(new_metadata.project_name),
        prev_n_technologies=len(current.mentioned_technologies),
        next_n_technologies=len(new_metadata.mentioned_technologies),
        had_agreed_scope=bool(current.agreed_scope),
        has_agreed_scope=bool(new_metadata.agreed_scope),
    ).emit()
    return new_metadata
