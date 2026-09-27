"""Integration tests for the request-metrics middleware (Session 16).

Mirrors ``tests/test_sessions.py::test_turn_observed_event_carries_all_required_fields``'s
``capture_logs`` pattern. No real LLM call succeeds in this environment (no
real API keys), so these tests exercise the middleware's OTHER job: a
request still gets exactly one ``request_completed`` event, with the real
HTTP status, even when the LLM call underneath fails -- that is the whole
point of counting in middleware rather than per-route.
"""

import structlog
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from app.main import app


def test_health_is_exempt_from_request_completed() -> None:
    structlog.reset_defaults()
    client = TestClient(app)

    with capture_logs() as logs:
        response = client.get("/health")

    assert response.status_code == 200
    assert [e for e in logs if e.get("event") == "request_completed"] == []


def test_estimate_call_emits_exactly_one_request_completed_event() -> None:
    """A request that fails inside the LLM call still counts, with the real
    status code -- this is the whole reason the metric lives in middleware."""
    structlog.reset_defaults()
    client = TestClient(app)

    with capture_logs() as logs:
        response = client.post(
            "/api/v1/estimate",
            json={"description": "Build a billing dashboard with Stripe integration."},
        )

    events = [e for e in logs if e.get("event") == "request_completed"]
    assert len(events) == 1, f"expected one request_completed event, got {len(events)}"
    ev = events[0]
    assert ev["path"] == "/api/v1/estimate"
    assert ev["method"] == "POST"
    assert ev["status"] == response.status_code
    assert ev["latency_ms"] >= 0
    assert ev["input_tokens"] >= 0
    assert ev["output_tokens"] >= 0
    assert ev["total_tokens"] == ev["input_tokens"] + ev["output_tokens"]
    assert ev["llm_calls"] >= 0
    assert ev["cost_usd"] >= 0


def test_unknown_path_still_emits_request_completed() -> None:
    """A 404 -- failing before any router -- must still be counted."""
    structlog.reset_defaults()
    client = TestClient(app)

    with capture_logs() as logs:
        response = client.get("/this-route-does-not-exist")

    events = [e for e in logs if e.get("event") == "request_completed"]
    assert len(events) == 1
    assert events[0]["status"] == response.status_code == 404
    assert events[0]["llm_calls"] == 0
