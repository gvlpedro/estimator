"""Critic service — independent audit of an EstimationResult.

The Critic is a stateless function: input (request, metadata, estimation) →
output ``CriticFeedback``. It does NOT make decisions about what to do with
its own output — that's the Boss's job. Keeping responsibilities separate is
what makes the Actor-Critic-Boss pattern auditable.

A failure inside the Critic does NOT block the pipeline. The Boss receives a
synthetic "accept with zero confidence" verdict so the actor's estimation
flows through unmodified — graceful degradation rather than a hard error in
front of the user.
"""

from __future__ import annotations

import structlog

from app.prompts.loader import render_critic_prompt
from app.schemas.critic import CriticFeedback
from app.schemas.estimation import EstimationRequest, EstimationResult
from app.services.llm_wrapper import LLMWrapper
from app.services.sessions import ProjectMetadata

log = structlog.get_logger()


class Critic:
    """Wraps a structured LLM call to audit an estimation draft."""

    def __init__(
        self,
        *,
        llm_wrapper: LLMWrapper,
        model: str | None = None,
        prompt_version: str = "v1",
    ) -> None:
        self.llm_wrapper = llm_wrapper
        self.model = model
        self.prompt_version = prompt_version

    def review(
        self,
        *,
        request: EstimationRequest,
        result: EstimationResult,
        project_metadata: ProjectMetadata | None = None,
    ) -> CriticFeedback:
        system_prompt, user_message = render_critic_prompt(
            request=request,
            result=result,
            project_metadata=project_metadata,
            version=self.prompt_version,
        )

        try:
            completion = self.llm_wrapper.complete_structured(
                system_prompt=system_prompt,
                user_message=user_message,
                response_model=CriticFeedback,
                model_override=self.model,
                max_tokens=1500,
                max_retries=3,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "critic_failed_fallback_accept",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            return CriticFeedback(verdict="accept", issues=[], confidence_in_review=0)

        feedback = completion.result
        log.info(
            "critic_completed",
            verdict=feedback.verdict,
            issue_count=len(feedback.issues),
            confidence=feedback.confidence_in_review,
            model=completion.model,
            latency_ms=completion.latency_ms,
        )
        return feedback
