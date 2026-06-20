"""Pipeline orchestrator: glue between guardrails, cache, prompt rendering and
the LLM wrapper. The router only translates exceptions to HTTP status codes.

Pipeline:

    1. Input guardrails (moderation + prompt injection + PII heuristics)
    2. Exact-match cache lookup  → return cached=True on hit
    3. Render the versioned prompt
    4. LLM call via Instructor with response_model=EstimationResult
    5. Output guardrail (enforce_scope_response filter)
    6. Cache the validated payload
    7. Return EstimationResponse with cached=False

Order rationale: guardrails go before the cache because a malicious or PII
description should never be served from cache. The semantic cache layer (in
the reference project) would slot between (2) and (3) — left out here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import structlog

from app.guardrails import InputGuardrailViolation, check_input, enforce_scope_response
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
    EstimationResult,
)
from app.services.cache import EstimationCache
from app.services.llm_wrapper import LLMWrapper
from app.services.sessions import ProjectMetadata

log = structlog.get_logger()

__all__ = ["EstimationService", "InputGuardrailViolation"]


def _exact_cache_key(
    request: EstimationRequest,
    prompt_version: str,
    model: str,
    project_metadata: ProjectMetadata | None,
) -> str:
    """Deterministic SHA-256 key over the typed request + prompt_version + model.

    ``project_metadata`` is included so two turns of the same session with
    different known facts do not collide on the same cache entry.
    """
    payload = json.dumps(
        {
            "description": request.description,
            "project_type": request.project_type.value,
            "detail_level": request.detail_level.value,
            "output_format": request.output_format.value,
            "reference_projects": (
                [rp.model_dump() for rp in request.reference_projects]
                if request.reference_projects
                else None
            ),
            "project_metadata": (
                project_metadata.model_dump(mode="json") if project_metadata else None
            ),
            "prompt_version": prompt_version,
            "model": model,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"estimation:structured:{digest}"


class EstimationService:
    """Single entry point for the structured estimation pipeline."""

    def __init__(
        self,
        *,
        llm_wrapper: LLMWrapper,
        exact_cache: EstimationCache,
        openai_client: Any | None = None,
        prompt_version: str = "v1",
    ) -> None:
        self.llm_wrapper = llm_wrapper
        self.exact_cache = exact_cache
        self.openai_client = openai_client
        self.prompt_version = prompt_version

    def estimate(
        self,
        request: EstimationRequest,
        *,
        prompt_version: str | None = None,
        project_metadata: ProjectMetadata | None = None,
    ) -> EstimationResponse:
        version = prompt_version or self.prompt_version

        check_input(request.description, openai_client=self.openai_client)

        cache_key = _exact_cache_key(
            request, version, self.llm_wrapper.primary_model, project_metadata
        )
        cached = self.exact_cache.get(cache_key)
        if cached:
            log.info("estimation_cache_hit", kind="exact", key_prefix=cache_key[:24])
            result = EstimationResult.model_validate(cached["result"])
            return EstimationResponse(result=result, prompt_version=version, cached=True)

        system_prompt, user_message = render_estimation_prompt(
            request, version=version, project_metadata=project_metadata
        )

        completion = self.llm_wrapper.complete_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=EstimationResult,
        )
        result = completion.result
        log.info(
            "estimation_generated",
            prompt_version=version,
            confidence_pct=result.confidence_pct,
            total_cost_eur=result.total_cost_eur,
            phases=len(result.phases),
            model=completion.model,
            provider=completion.provider,
            latency_ms=completion.latency_ms,
        )

        result = enforce_scope_response(result)

        self.exact_cache.set(
            cache_key,
            {
                "result": result.model_dump(mode="json"),
                "prompt_version": version,
            },
        )

        return EstimationResponse(result=result, prompt_version=version, cached=False)
