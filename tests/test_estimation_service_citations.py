"""Tests for citation-verification wiring inside ``EstimationService``
(Session 10).

Fakes ``LLMWrapper.complete_structured`` (never calls a real LLM — same
convention as ``tests/test_sessions.py``'s ``_StubEstimationService`` and
``tests/test_llm_wrapper.py``'s patched ``router.completion``) so we can
control exactly which phases the "model" returns and assert on the
``citation_verification_completed`` structlog event the pipeline emits.
"""

from __future__ import annotations

import fakeredis
import structlog

from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    EstimationResult,
    OutputFormat,
    Phase,
    ProjectType,
    SourceReference,
)
from app.services.cache import EstimationCache
from app.services.estimation import EstimationService
from app.services.llm_wrapper import StructuredCompletion


class _FakeLLMWrapper:
    """Stand-in for LLMWrapper: returns a canned EstimationResult without
    touching litellm/instructor."""

    def __init__(self, result: EstimationResult) -> None:
        self.primary_model = "fake-model"
        self._result = result
        self.calls: list[dict] = []

    def complete_structured(self, **kwargs) -> StructuredCompletion[EstimationResult]:
        self.calls.append(kwargs)
        return StructuredCompletion(
            result=self._result,
            model="fake-model",
            provider="fake",
            latency_ms=5,
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.0,
        )


def _request() -> EstimationRequest:
    return EstimationRequest(
        description=(
            "A small CRM with auth, contacts, deals and a simple Kanban — "
            "MVP for a 5-person sales team."
        ),
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )


def _result_with_phases(phases: list[Phase]) -> EstimationResult:
    return EstimationResult(
        summary="A CRM MVP with auth, contacts and a Kanban board for a 5-person team.",
        confidence_pct=80,
        phases=phases,
        total_duration_weeks=1,
        total_cost_eur=1,
    )


def _service(result: EstimationResult) -> tuple[EstimationService, _FakeLLMWrapper]:
    wrapper = _FakeLLMWrapper(result)
    cache = EstimationCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)
    service = EstimationService(llm_wrapper=wrapper, exact_cache=cache, prompt_version="v3")
    return service, wrapper


def test_dangling_citation_is_detected_and_logged_without_raising() -> None:
    """A phase that claims grounded=True but cites a chunk_id absent from the
    retrieved context must be caught and logged — and must NOT raise or
    otherwise block the response (detection, not prevention)."""
    phases = [
        Phase(
            name="Backend build",
            duration_weeks=4,
            cost_eur=12_000,
            summary="Core API, auth and the Postgres schema for the CRM module.",
            grounded=True,
            sources=[SourceReference(chunk_id="does-not-exist", document_id="9", evidence="80h.")],
        )
    ]
    service, _wrapper = _service(_result_with_phases(phases))

    with structlog.testing.capture_logs() as logs:
        response = service.estimate(
            _request(),
        )

    # Detection must not block the response.
    assert response.result.phases[0].name == "Backend build"

    events = [e for e in logs if e.get("event") == "citation_verification_completed"]
    assert len(events) == 1
    ev = events[0]
    assert ev["dangling_count"] == 1
    assert ev["grounded_count"] == 0
    assert ev["unsupported_count"] == 0
    assert ev["dangling_details"] == [
        {"phase_name": "Backend build", "dangling_chunk_ids": ["does-not-exist"]}
    ]
    assert isinstance(ev["request_id"], str) and len(ev["request_id"]) > 0


def test_grounded_citation_against_matching_retrieved_context_is_grounded() -> None:
    phases = [
        Phase(
            name="Backend build",
            duration_weeks=4,
            cost_eur=12_000,
            summary="Core API, auth and the Postgres schema for the CRM module.",
            grounded=True,
            sources=[SourceReference(chunk_id="101", document_id="9", evidence="80h.")],
        )
    ]
    service, _wrapper = _service(_result_with_phases(phases))
    retrieved_context = [
        {
            "chunk_id": 101,
            "document_id": 9,
            "chunk_type": "line_item",
            "content": "Backend build 80h.",
            "distance": 0.1,
            "score": None,
            "metadata": {},
        }
    ]

    with structlog.testing.capture_logs() as logs:
        service._run_structured_pipeline(
            _request(),
            retrieved_context_provider=lambda: retrieved_context,
        )

    events = [e for e in logs if e.get("event") == "citation_verification_completed"]
    assert len(events) == 1
    assert events[0]["grounded_count"] == 1
    assert events[0]["dangling_count"] == 0


def test_ungrounded_phase_counts_as_unsupported_not_dangling() -> None:
    phases = [
        Phase(
            name="Discovery",
            duration_weeks=1,
            cost_eur=2_000,
            summary="Stakeholder interviews and scope clarification workshop.",
            grounded=False,
            sources=[],
        )
    ]
    service, _wrapper = _service(_result_with_phases(phases))

    with structlog.testing.capture_logs() as logs:
        service.estimate(_request())

    events = [e for e in logs if e.get("event") == "citation_verification_completed"]
    assert len(events) == 1
    assert events[0]["unsupported_count"] == 1
    assert events[0]["dangling_count"] == 0
    assert events[0]["dangling_details"] == []


def test_request_id_differs_across_calls() -> None:
    """Each call mints its own request_id (a NEW per-call id, distinct from
    session_id) — two calls must not share one."""
    phases = [
        Phase(
            name="Discovery",
            duration_weeks=1,
            cost_eur=2_000,
            summary="Stakeholder interviews and scope clarification workshop.",
            grounded=False,
            sources=[],
        )
    ]

    ids = []
    for _ in range(2):
        service, _wrapper = _service(_result_with_phases(phases))
        with structlog.testing.capture_logs() as logs:
            service.estimate(_request())
        events = [e for e in logs if e.get("event") == "citation_verification_completed"]
        ids.append(events[0]["request_id"])

    assert ids[0] != ids[1]
