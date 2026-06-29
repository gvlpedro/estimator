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

from app.guardrails import InputGuardrailViolation, check_input, enforce_scope_response
from app.prompts.loader import render_estimation_prompt
from app.schemas.critic import CriticFeedback
from app.schemas.estimation import (
    ACBResponse,
    EstimationRequest,
    EstimationResponse,
    EstimationResult,
)
from app.schemas.log import (
    ACBActorDraft,
    ACBCompleted,
    ACBRequestReceived,
    EstimationCacheHit,
    EstimationGenerated,
)
from app.services.boss import Boss
from app.services.cache import EstimationCache
from app.services.critic import Critic
from app.services.llm_wrapper import LLMWrapper
from app.services.sessions import ProjectMetadata

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
        boss_max_iterations: int = 2,
        critic_model: str | None = None,
        critic_prompt_version: str = "v1",
    ) -> None:
        self.llm_wrapper = llm_wrapper
        self.exact_cache = exact_cache
        self.openai_client = openai_client
        self.prompt_version = prompt_version
        self.boss_max_iterations = boss_max_iterations
        self.critic_model = critic_model
        self.critic_prompt_version = critic_prompt_version

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
            EstimationCacheHit(kind="exact", key_prefix=cache_key[:24]).emit()
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
        EstimationGenerated(
            prompt_version=version,
            confidence_pct=result.confidence_pct,
            total_cost_eur=result.total_cost_eur,
            phases=len(result.phases),
            model=completion.model,
            provider=completion.provider,
            latency_ms=completion.latency_ms,
        ).emit()

        result = enforce_scope_response(result)

        self.exact_cache.set(
            cache_key,
            {
                "result": result.model_dump(mode="json"),
                "prompt_version": version,
            },
        )

        return EstimationResponse(result=result, prompt_version=version, cached=False)

    def estimate_with_acb(
        self,
        request: EstimationRequest,
        *,
        prompt_version: str | None = None,
        project_metadata: ProjectMetadata | None = None,
    ) -> ACBResponse:
        """Actor-Critic-Boss variant of the structured pipeline.

        Runs an actor → critic loop up to ``boss_max_iterations`` rounds. The
        Boss either accepts the actor's draft, asks the actor to retry with
        the Critic's feedback, or — when iterations run out or the Critic
        rejects outright — returns the last draft annotated with the Critic's
        open caveats. The cache is intentionally bypassed: the value of ACB is
        the audit trail, and serving a previous cached envelope without a
        fresh review would mask drift between the actor and the critic.
        """
        version = prompt_version or self.prompt_version

        check_input(request.description, openai_client=self.openai_client)

        ACBRequestReceived(
            project_type=request.project_type.value,
            detail_level=request.detail_level.value,
            output_format=request.output_format.value,
            description_chars=len(request.description),
            max_iterations=self.boss_max_iterations,
        ).emit()

        def _actor(critic_feedback: CriticFeedback | None) -> EstimationResult:
            system_prompt, user_message = render_estimation_prompt(
                request, version=version, project_metadata=project_metadata
            )
            if critic_feedback is not None:
                system_prompt = _append_critic_feedback(system_prompt, critic_feedback)

            completion = self.llm_wrapper.complete_structured(
                system_prompt=system_prompt,
                user_message=user_message,
                response_model=EstimationResult,
            )
            draft = enforce_scope_response(completion.result)
            ACBActorDraft(
                with_critic_feedback=critic_feedback is not None,
                issues_in_feedback=(
                    len(critic_feedback.issues) if critic_feedback is not None else 0
                ),
                confidence_pct=draft.confidence_pct,
                total_cost_eur=draft.total_cost_eur,
            ).emit()
            return draft

        critic = Critic(
            llm_wrapper=self.llm_wrapper,
            model=self.critic_model,
            prompt_version=self.critic_prompt_version,
        )

        def _critic(draft: EstimationResult) -> CriticFeedback:
            return critic.review(
                request=request,
                result=draft,
                project_metadata=project_metadata,
            )

        boss = Boss(max_iterations=self.boss_max_iterations)
        final_result, trace = boss.run(actor=_actor, critic=_critic)

        ACBCompleted(
            final_decision=trace.final_decision,
            iterations_run=trace.iterations_run,
            confidence_pct=final_result.confidence_pct,
            total_cost_eur=final_result.total_cost_eur,
        ).emit()

        return ACBResponse(
            result=final_result,
            prompt_version=version,
            cached=False,
            acb=trace,
        )


def _append_critic_feedback(system_prompt: str, feedback: CriticFeedback) -> str:
    """Append the Critic's feedback as a structured block to the actor's
    system prompt for iteration N+1.

    Kept as a free-standing helper (vs. a template) so we do not couple the
    actor's prompt versioning to ACB plumbing. Feedback already validated by
    the ``CriticFeedback`` schema, so we can format it inline."""
    if not feedback.issues:
        return system_prompt
    lines = [
        "",
        "<previous_critic_feedback>",
        f"verdict: {feedback.verdict}",
        f"confidence_in_review: {feedback.confidence_in_review}",
        "issues:",
    ]
    for issue in feedback.issues:
        lines.append(
            f"- [{issue.severity}] {issue.category} @ {issue.field_path}: {issue.description}"
        )
        if issue.suggested_fix:
            lines.append(f"  fix: {issue.suggested_fix}")
    lines.append(
        "Fix every critical/major issue above before returning. Do not ignore "
        "field_path hints; they point at concrete fields of the estimation."
    )
    lines.append("</previous_critic_feedback>")
    return system_prompt + "\n" + "\n".join(lines)
