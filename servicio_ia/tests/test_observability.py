"""Unit tests for the per-request metrics primitives (Session 15/16 evaluation deliverable)."""

from app.observability import begin_request, end_request, estimate_cost_usd, record_token_usage


def test_estimate_cost_usd_is_zero_for_no_tokens():
    assert estimate_cost_usd(0, 0) == 0.0


def test_estimate_cost_usd_scales_with_input_and_output_rates():
    # 1000 input tokens + 1000 output tokens = exactly one unit of each rate.
    cost = estimate_cost_usd(1000, 1000)

    assert cost == round(0.00025 + 0.002, 6)


def test_estimate_cost_usd_weighs_output_tokens_more_than_input():
    input_only = estimate_cost_usd(1000, 0)
    output_only = estimate_cost_usd(0, 1000)

    assert output_only > input_only


def test_estimate_cost_usd_defaults_are_env_overridable(monkeypatch):
    monkeypatch.setenv("COST_PER_1K_INPUT_TOKENS", "1.0")
    monkeypatch.setenv("COST_PER_1K_OUTPUT_TOKENS", "1.0")

    # The module reads these at import time, so re-import to pick up the
    # patched environment rather than asserting against the already-imported
    # module-level constants.
    import importlib

    import app.observability as observability_module

    importlib.reload(observability_module)
    try:
        assert observability_module.estimate_cost_usd(1000, 0) == 1.0
    finally:
        importlib.reload(observability_module)


def test_record_token_usage_is_a_noop_with_no_active_request():
    # No begin_request() call first -- nothing to accumulate into, and this
    # must not raise (e.g. a unit test that calls a node directly).
    record_token_usage(agent="x", call="y", input_tokens=10, output_tokens=5)


def test_end_request_totals_every_recorded_call():
    token = begin_request()
    record_token_usage(agent="requirements_extractor", call="extract_requirements", input_tokens=100, output_tokens=40)
    record_token_usage(agent="requirements_extractor", call="classify_components", input_tokens=60, output_tokens=20)

    metrics = end_request(token)

    assert metrics["input_tokens"] == 160
    assert metrics["output_tokens"] == 60
    assert metrics["total_tokens"] == 220
    assert metrics["llm_calls"] == 2
    assert metrics["cost_usd"] == estimate_cost_usd(160, 60)


def test_end_request_with_no_calls_reports_zeroed_metrics():
    token = begin_request()

    metrics = end_request(token)

    assert metrics == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
        "cost_usd": 0.0,
    }


def test_end_request_resets_so_a_later_record_call_is_a_noop_again():
    token = begin_request()
    end_request(token)

    # Accumulator was reset by end_request -- recording now has nothing to
    # land in, same as never having called begin_request() at all.
    record_token_usage(agent="x", call="y", input_tokens=999, output_tokens=999)

    # No assertion needed beyond "this didn't raise and didn't resurrect the
    # closed accumulator" -- a second begin_request()/end_request() pair
    # confirms the slate is clean.
    token2 = begin_request()
    assert end_request(token2)["llm_calls"] == 0
