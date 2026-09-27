"""Unit tests for the self-contained metrics dashboard's log parsing/aggregation."""

import json

from eval.dashboard import parse_metrics_events, render, render_html, summarize


def _event(**overrides):
    base = {
        "event": "request_completed",
        "path": "/estimate",
        "method": "POST",
        "status": 200,
        "latency_ms": 1000.0,
        "input_tokens": 500,
        "output_tokens": 200,
        "total_tokens": 700,
        "llm_calls": 2,
        "cost_usd": 0.0005,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# parse_metrics_events
# ---------------------------------------------------------------------------


def test_parses_plain_json_lines():
    lines = [json.dumps(_event())]

    events = parse_metrics_events(lines)

    assert len(events) == 1
    assert events[0]["path"] == "/estimate"


def test_strips_docker_compose_log_prefix():
    lines = [f"servicio-ia  | {json.dumps(_event())}"]

    events = parse_metrics_events(lines)

    assert len(events) == 1


def test_ignores_non_json_and_blank_lines():
    lines = ["", "2026-09-21 human-readable log line, not JSON", "   \n"]

    events = parse_metrics_events(lines)

    assert events == []


def test_ignores_json_lines_that_are_a_different_event():
    lines = [json.dumps({"event": "agent_turn_started", "agent": "requirements_extractor"})]

    events = parse_metrics_events(lines)

    assert events == []


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def test_summarize_empty_input_reports_zero_requests():
    summary = summarize([])

    assert summary["request_count"] == 0
    assert summary["error_rate"] is None


def test_summarize_computes_latency_mean_and_p95():
    events = [_event(latency_ms=100.0), _event(latency_ms=200.0), _event(latency_ms=300.0)]

    summary = summarize(events)

    assert summary["latency_ms_mean"] == 200.0
    assert summary["latency_ms_p95"] == 300.0


def test_summarize_computes_cost_mean_and_total():
    events = [_event(cost_usd=0.001), _event(cost_usd=0.003)]

    summary = summarize(events)

    assert summary["cost_usd_mean"] == 0.002
    assert summary["cost_usd_total"] == 0.004


def test_summarize_error_rate_counts_only_non_2xx_status():
    # status is now the HTTP status the middleware recorded, not a business
    # outcome -- a 200 carrying "awaiting_human_review" is correct behavior,
    # not an error, and never appears here as one.
    events = [
        _event(status=200),
        _event(status=200),
        _event(status=401),
        _event(status=500),
    ]

    summary = summarize(events)

    assert summary["error_rate"] == 0.5


def test_summarize_missing_status_counts_as_an_error():
    events = [_event(status=200), _event(status=0)]

    summary = summarize(events)

    assert summary["error_rate"] == 0.5


def test_summarize_breaks_down_by_path_busiest_first():
    events = [
        _event(path="/estimate"),
        _event(path="/estimate"),
        _event(path="/search"),
    ]

    summary = summarize(events)

    paths = list(summary["by_path"].keys())
    assert paths[0] == "/estimate"
    assert summary["by_path"]["/estimate"]["request_count"] == 2
    assert summary["by_path"]["/search"]["request_count"] == 1


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_reports_no_events_found():
    text = render(summarize([]))

    assert "No request_completed events" in text


def test_render_includes_all_headline_signals():
    text = render(summarize([_event()]))

    assert "requests observed" in text
    assert "latency mean / p95" in text
    assert "cost mean / total" in text
    assert "error rate" in text


def test_render_includes_by_path_breakdown():
    text = render(summarize([_event(path="/estimate"), _event(path="/search")]))

    assert "/estimate" in text
    assert "/search" in text


# ---------------------------------------------------------------------------
# render_html
# ---------------------------------------------------------------------------


def test_render_html_is_a_self_contained_page():
    page = render_html(summarize([_event()]))

    assert page.startswith("<!doctype html>")
    assert "/estimate" in page
    assert "</html>" in page


def test_render_html_handles_zero_events_without_crashing():
    page = render_html(summarize([]))

    assert page.startswith("<!doctype html>")
