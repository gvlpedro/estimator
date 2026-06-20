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
    EstimationRequest,
    EstimationResponse,
    EstimationResult,
    Phase,
)
from app.services.llm_wrapper import StructuredCompletion
from app.services.sessions import ConversationHistory, ProjectMetadata, SessionStore


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
    """Stand-in for EstimationService that records calls and returns a canned response."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

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
def stub_service() -> _StubEstimationService:
    return _StubEstimationService()


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
