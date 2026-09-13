"""Tests for the session lifecycle endpoints.

These tests exercise the new ``/api/v1/sessions`` surface end-to-end through
FastAPI's TestClient, stubbing the LLM-facing collaborators so we keep the
suite hermetic. The stubs intentionally mirror the production contract of
``EstimationService.estimate`` and ``LLMWrapper.complete_structured`` so the
router's wiring (history append + metadata extractor) is what's under test.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.dependencies import (
    get_estimation_service,
    get_llm_wrapper,
    get_session_store,
)
from app.main import app
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    EstimationResponse,
    EstimationResult,
    OutputFormat,
    Phase,
    ProjectType,
)
from app.schemas.log import TurnObserved
from app.services.llm_wrapper import StructuredCompletion
from app.services.metadata_extractor import extract_project_metadata
from app.services.sessions import (
    ConversationHistory,
    ProjectMetadata,
    Session,
    SessionStore,
)


def _valid_result() -> EstimationResult:
    """A schema-valid EstimationResult: phases sum to total_cost_eur, confidence >= 30."""
    return EstimationResult(
        summary="Inventory SaaS with auth, dashboard and a Stripe-backed billing module.",
        confidence_pct=70,
        phases=[
            Phase(
                name="Setup",
                duration_weeks=2,
                cost_eur=5_000,
                summary="Repo bootstrap, CI/CD, infra-as-code and staging environment.",
            ),
            Phase(
                name="Build",
                duration_weeks=6,
                cost_eur=15_000,
                summary="Core CRUD, dashboards and Stripe integration.",
            ),
        ],
        total_duration_weeks=8,
        total_cost_eur=20_000,
    )


class _StubEstimationService:
    """Stand-in for EstimationService that records calls and returns a canned response.

    Mirrors the real service's split of responsibilities: ``estimate`` produces
    the structured response, ``estimate_conversational`` does the per-turn
    orchestration (history append + metadata extraction + ``turn_observed``)
    so the router stays thin. The orchestration here intentionally goes
    through the same helpers as the real service so the test stub does not
    diverge from production logic — only the LLM-facing calls are stubbed."""

    def __init__(self, wrapper: "_StubLLMWrapper") -> None:
        self.calls: list[dict] = []
        self._wrapper = wrapper

    def estimate(
        self,
        request: EstimationRequest,
        *,
        prompt_version: str | None = None,
        project_metadata: ProjectMetadata | None = None,
    ) -> EstimationResponse:
        self.calls.append(
            {
                "description": request.description,
                "project_metadata": project_metadata.model_copy() if project_metadata else None,
            }
        )
        return EstimationResponse(result=_valid_result(), prompt_version="v1", cached=False)

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
        request = EstimationRequest(
            description=enriched_transcript,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
        )
        response = self.estimate(request, project_metadata=session.metadata)
        assistant_reply_json = response.result.model_dump_json()
        session.history.add("user", enriched_transcript)
        session.history.add("assistant", assistant_reply_json)
        new_metadata, extractor_ops = extract_project_metadata(
            wrapper=self._wrapper,
            current=session.metadata,
            user_message=enriched_transcript,
            assistant_reply=assistant_reply_json,
        )
        session.metadata = new_metadata
        turn_index = session.next_turn()
        session.touch()
        history = session.history
        TurnObserved(
            turn_index=turn_index,
            session_id=session.session_id,
            enriched_transcript_chars=len(enriched_transcript),
            attachments_total_chars=attachments_total_chars,
            messages_in_window=len(history.messages),
            anchors_count=0,
            summary_chars=0,
            tokens_in=extractor_ops.input_tokens,
            tokens_out=extractor_ops.output_tokens,
            cost_usd=round(extractor_ops.cost_usd, 6),
            latency_ms=extractor_ops.latency_ms,
            cache_hit_kind="exact" if response.cached else "none",
            last_resolved_tier=session.last_resolved_tier,
        ).emit()
        return response


class _StubLLMWrapper:
    """Stand-in for LLMWrapper that always returns the same extracted metadata."""

    def __init__(self, metadata: ProjectMetadata) -> None:
        self._metadata = metadata
        self.calls: list[dict] = []

    def complete_structured(
        self,
        *,
        system_prompt: str,
        user_message: str,
        response_model,
        **_: object,
    ) -> StructuredCompletion[ProjectMetadata]:
        self.calls.append({"system_prompt": system_prompt, "user_message": user_message})
        return StructuredCompletion(
            result=self._metadata.model_copy(),
            model="stub-model",
            provider="stub",
            latency_ms=0,
        )


@pytest.fixture
def session_store() -> SessionStore:
    """A fresh, isolated SessionStore per test (the production dep is lru_cached)."""
    return SessionStore(max_turns=4)


@pytest.fixture
def stub_wrapper() -> _StubLLMWrapper:
    return _StubLLMWrapper(
        metadata=ProjectMetadata(
            project_name="Inventory SaaS",
            assumed_team_size=4,
            mentioned_technologies=["fastapi", "postgresql", "stripe"],
            agreed_scope="Inventory SaaS with auth, dashboard and Stripe billing.",
        )
    )


@pytest.fixture
def stub_service(stub_wrapper: _StubLLMWrapper) -> _StubEstimationService:
    return _StubEstimationService(wrapper=stub_wrapper)


@pytest.fixture
def client(
    session_store: SessionStore,
    stub_service: _StubEstimationService,
    stub_wrapper: _StubLLMWrapper,
) -> Iterator[TestClient]:
    app.dependency_overrides[get_session_store] = lambda: session_store
    app.dependency_overrides[get_estimation_service] = lambda: stub_service
    app.dependency_overrides[get_llm_wrapper] = lambda: stub_wrapper
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_session_store, None)
        app.dependency_overrides.pop(get_estimation_service, None)
        app.dependency_overrides.pop(get_llm_wrapper, None)


def test_create_session_returns_id_and_registers_it(
    client: TestClient, session_store: SessionStore
) -> None:
    response = client.post("/api/v1/sessions")

    assert response.status_code == 201
    session_id = response.json()["session_id"]
    assert session_id  # non-empty
    assert session_id in session_store


def test_get_session_returns_404_for_unknown_id(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/does-not-exist")

    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


def test_estimate_in_session_updates_metadata_and_history(
    client: TestClient,
    stub_service: _StubEstimationService,
    stub_wrapper: _StubLLMWrapper,
) -> None:
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    transcript = "x" * 60  # satisfies min_length=20

    estimate_response = client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data={"transcript": transcript},
    )
    assert estimate_response.status_code == 200, estimate_response.text
    body = estimate_response.json()
    assert body["prompt_version"] == "v1"
    assert body["result"]["total_cost_eur"] == 20_000

    # The estimation service saw an empty metadata on the first turn (no memory yet).
    assert len(stub_service.calls) == 1
    assert stub_service.calls[0]["project_metadata"] == ProjectMetadata()

    # The extractor ran with the user transcript and the assistant JSON reply.
    assert len(stub_wrapper.calls) == 1
    assert transcript in stub_wrapper.calls[0]["user_message"]

    # GET /sessions/{id} now reflects both the updated memory and the recorded history.
    snapshot = client.get(f"/api/v1/sessions/{session_id}").json()
    assert snapshot["project_metadata"]["project_name"] == "Inventory SaaS"
    assert snapshot["project_metadata"]["mentioned_technologies"] == [
        "fastapi",
        "postgresql",
        "stripe",
    ]
    roles = [m["role"] for m in snapshot["messages"]]
    assert roles == ["user", "assistant"]
    assert snapshot["messages"][0]["content"] == transcript


def test_second_turn_sees_accumulated_metadata_from_first_turn(
    client: TestClient, stub_service: _StubEstimationService
) -> None:
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    first = client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data={"transcript": "First turn: " + "a" * 60},
    )
    assert first.status_code == 200, first.text

    second = client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data={"transcript": "Second turn: " + "b" * 60},
    )
    assert second.status_code == 200, second.text

    # The second call must have received the metadata the extractor wrote after turn 1,
    # proving that memory (project_metadata) is threaded turn-to-turn — not derived from
    # the sliding-window history alone.
    assert len(stub_service.calls) == 2
    second_call_metadata = stub_service.calls[1]["project_metadata"]
    assert second_call_metadata is not None
    assert second_call_metadata.project_name == "Inventory SaaS"
    assert second_call_metadata.assumed_team_size == 4

    # And the history grew to two full user/assistant pairs.
    snapshot = client.get(f"/api/v1/sessions/{session_id}").json()
    roles = [m["role"] for m in snapshot["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_get_session_exposes_debug_counters(
    client: TestClient, session_store: SessionStore
) -> None:
    """The snapshot must surface the observability fields: message_count
    computed from the sliding window, plus stable placeholders for the
    subsystems (compression, tier resolver) not yet wired in. Without these
    fields the UI debug panel has nothing to render."""
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    fresh = client.get(f"/api/v1/sessions/{session_id}").json()
    assert fresh["message_count"] == 0
    assert fresh["anchors_count"] == 0
    assert fresh["summary_chars"] == 0
    assert fresh["last_resolved_tier"] is None
    assert fresh["last_tier_rule"] is None

    client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data={"transcript": "x" * 60},
    )

    after_turn = client.get(f"/api/v1/sessions/{session_id}").json()
    # One user + one assistant message appended on the turn.
    assert after_turn["message_count"] == 2
    assert after_turn["anchors_count"] == 0
    assert after_turn["summary_chars"] == 0

    # last_resolved_tier / last_tier_rule track tier-resolver writes; the
    # resolver is not wired yet, so they remain None.
    session = session_store.get(session_id)
    assert session is not None
    session.last_resolved_tier = "developer"
    session.last_tier_rule = "default"
    populated = client.get(f"/api/v1/sessions/{session_id}").json()
    assert populated["last_resolved_tier"] == "developer"
    assert populated["last_tier_rule"] == "default"


def test_turn_observed_event_carries_all_required_fields(
    client: TestClient, session_store: SessionStore
) -> None:
    """A CSV extraction over the logs needs ONE event per turn that already
    carries every observable field. ``turn_observed`` must fire exactly once
    per session estimate and include every contract field — turn index,
    transcript and attachment sizes, history counters, token/cost rollup,
    cache kind, and the placeholder tier."""
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    session_id = client.post("/api/v1/sessions").json()["session_id"]
    session_store.get(session_id).last_resolved_tier = "developer"

    transcript = "Build a billing dashboard. " * 5
    with capture_logs() as logs:
        client.post(
            f"/api/v1/sessions/{session_id}/estimate",
            data={"transcript": transcript},
        )

    turn_events = [e for e in logs if e.get("event") == "turn_observed"]
    assert len(turn_events) == 1, f"expected one turn_observed event, got {len(turn_events)}"
    ev = turn_events[0]
    assert ev["turn_index"] == 1
    assert ev["session_id"] == session_id
    assert ev["enriched_transcript_chars"] == len(transcript)
    assert ev["attachments_total_chars"] == 0
    assert ev["messages_in_window"] == 2  # user + assistant
    assert ev["anchors_count"] == 0
    assert ev["summary_chars"] == 0
    assert ev["tokens_in"] >= 0
    assert ev["tokens_out"] >= 0
    assert ev["cost_usd"] >= 0
    assert ev["latency_ms"] >= 0
    assert ev["cache_hit_kind"] in {"none", "exact", "semantic"}
    assert ev["last_resolved_tier"] == "developer"

    # A second turn increments the index and grows the window.
    with capture_logs() as logs2:
        client.post(
            f"/api/v1/sessions/{session_id}/estimate",
            data={"transcript": transcript + " second turn"},
        )
    second = next(e for e in logs2 if e.get("event") == "turn_observed")
    assert second["turn_index"] == 2
    assert second["messages_in_window"] == 4


def test_history_evicts_oldest_pair_when_max_turns_exceeded() -> None:
    """With max_turns=2 the window holds 4 non-system messages (2 pairs).

    Adding a 3rd full turn pushes the count to 6 non-system messages, which
    exceeds the cap, so the oldest user+assistant pair must be dropped — but
    the pinned system prompt at index 0 must survive untouched.
    """
    history = ConversationHistory(max_turns=2)
    history.set_system("you are an estimator")

    history.add("user", "u1")
    history.add("assistant", "a1")
    history.add("user", "u2")
    history.add("assistant", "a2")

    # Still within the cap — nothing evicted yet.
    assert [(m.role, m.content) for m in history.messages] == [
        ("system", "you are an estimator"),
        ("user", "u1"),
        ("assistant", "a1"),
        ("user", "u2"),
        ("assistant", "a2"),
    ]

    history.add("user", "u3")
    history.add("assistant", "a3")

    # Oldest pair (u1/a1) is gone, system prompt survives, newest pair is kept.
    assert [(m.role, m.content) for m in history.messages] == [
        ("system", "you are an estimator"),
        ("user", "u2"),
        ("assistant", "a2"),
        ("user", "u3"),
        ("assistant", "a3"),
    ]
