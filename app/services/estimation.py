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

from dataclasses import dataclass

from app.guardrails import InputGuardrailViolation, check_input, enforce_scope_response
from app.prompts.loader import render_estimation_prompt
from app.schemas.critic import CriticFeedback
from app.schemas.estimation import (
    ACBResponse,
    DetailLevel,
    EstimationRequest,
    EstimationResponse,
    EstimationResult,
    OutputFormat,
    ProjectType,
)
from app.schemas.log import (
    ACBActorDraft,
    ACBCompleted,
    ACBRequestReceived,
    CacheHitKind,
    EstimationCacheHit,
    EstimationGenerated,
    TurnObserved,
)
from app.services.boss import Boss
from app.services.cache import EstimationCache
from app.services.critic import Critic
from app.services.llm_wrapper import LLMWrapper
from app.services.metadata_extractor import extract_project_metadata
from app.services.sessions import ProjectMetadata, Session

__all__ = ["EstimationService", "InputGuardrailViolation"]


@dataclass(frozen=True)
class _PipelineOps:
    """Per-call ops captured during the structured pipeline so the per-turn
    aggregate (TurnObserved) can roll them up without re-reading log events."""

    cache_hit_kind: CacheHitKind
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


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
        response, _ops = self._run_structured_pipeline(
            request,
            prompt_version=prompt_version,
            project_metadata=project_metadata,
        )
        return response

    def _run_structured_pipeline(
        self,
        request: EstimationRequest,
        *,
        prompt_version: str | None = None,
        project_metadata: ProjectMetadata | None = None,
    ) -> tuple[EstimationResponse, _PipelineOps]:
        """Run cache lookup → LLM → guardrail → cache set, returning both the
        response and the operational counters needed for the per-turn aggregate."""
        version = prompt_version or self.prompt_version

        check_input(request.description, openai_client=self.openai_client)

        cache_key = _exact_cache_key(
            request, version, self.llm_wrapper.primary_model, project_metadata
        )
        cached = self.exact_cache.get(cache_key)
        if cached:
            EstimationCacheHit(kind="exact", key_prefix=cache_key[:24]).emit()
            result = EstimationResult.model_validate(cached["result"])
            return (
                EstimationResponse(result=result, prompt_version=version, cached=True),
                _PipelineOps(
                    cache_hit_kind="exact",
                    latency_ms=0,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                ),
            )

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

        return (
            EstimationResponse(result=result, prompt_version=version, cached=False),
            _PipelineOps(
                cache_hit_kind="none",
                latency_ms=completion.latency_ms,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost_usd=completion.cost_usd,
            ),
        )

    def estimate_conversational(
        self,
        *,
        session: Session,
        enriched_transcript: str,
        attachments_total_chars: int,
        project_type: ProjectType,
        detail_level: DetailLevel,
        output_format: OutputFormat,
        prompt_version: str | None = None,
    ) -> EstimationResponse:
        """One full conversational turn: cache → LLM → guardrail → history →
        metadata extractor → ``turn_observed`` event, exactly in this order.

        Emits a single ``turn_observed`` event just before returning so a CSV
        extraction over the logs gets one row per turn without joining across
        the per-call events that still fire underneath (``cache_hit``,
        ``llm_structured_call_completed``, ``project_metadata_updated``...).

        ``attachments_total_chars`` is passed in by the caller (router) because
        the service does not parse uploads itself; the merged text arrives via
        ``enriched_transcript``. Passing the raw count keeps the observability
        contract intact without dragging file IO into the service layer."""
        request = EstimationRequest(
            description=enriched_transcript,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
        )

        response, pipeline_ops = self._run_structured_pipeline(
            request,
            prompt_version=prompt_version,
            project_metadata=session.metadata,
        )

        assistant_reply_json = response.result.model_dump_json()
        session.history.add("user", enriched_transcript)
        session.history.add("assistant", assistant_reply_json)

        new_metadata, extractor_ops = extract_project_metadata(
            wrapper=self.llm_wrapper,
            current=session.metadata,
            user_message=enriched_transcript,
            assistant_reply=assistant_reply_json,
        )
        session.metadata = new_metadata
        turn_index = session.next_turn()
        session.touch()

        history = session.history
        anchors = getattr(history, "anchors", []) or []
        summary = getattr(history, "summary", None) or ""

        TurnObserved(
            turn_index=turn_index,
            session_id=session.session_id,
            enriched_transcript_chars=len(enriched_transcript),
            attachments_total_chars=attachments_total_chars,
            messages_in_window=len(history.messages),
            anchors_count=len(anchors),
            summary_chars=len(summary),
            tokens_in=pipeline_ops.input_tokens + extractor_ops.input_tokens,
            tokens_out=pipeline_ops.output_tokens + extractor_ops.output_tokens,
            cost_usd=round(pipeline_ops.cost_usd + extractor_ops.cost_usd, 6),
            latency_ms=pipeline_ops.latency_ms + extractor_ops.latency_ms,
            cache_hit_kind=pipeline_ops.cache_hit_kind,
            last_resolved_tier=session.last_resolved_tier,
        ).emit()

        return response

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
