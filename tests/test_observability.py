"""Unit tests for the per-request token/cost accumulator (Session 16)."""

from app.services.observability import begin_request, end_request, record_token_usage


def test_no_active_request_is_a_noop() -> None:
    """Calling record_token_usage with no begin_request() is not an error."""
    record_token_usage(input_tokens=100, output_tokens=50, cost_usd=0.01)  # must not raise


def test_end_request_with_zero_calls_returns_zeroed_totals() -> None:
    token = begin_request()
    totals = end_request(token)
    assert totals == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
        "cost_usd": 0.0,
    }


def test_accumulates_multiple_calls_within_one_request() -> None:
    token = begin_request()
    record_token_usage(input_tokens=100, output_tokens=50, cost_usd=0.001)
    record_token_usage(input_tokens=200, output_tokens=80, cost_usd=0.002)
    totals = end_request(token)
    assert totals["input_tokens"] == 300
    assert totals["output_tokens"] == 130
    assert totals["total_tokens"] == 430
    assert totals["llm_calls"] == 2
    assert totals["cost_usd"] == 0.003


def test_reset_prevents_leaking_into_the_next_request() -> None:
    token1 = begin_request()
    record_token_usage(input_tokens=999, output_tokens=999, cost_usd=1.0)
    end_request(token1)

    token2 = begin_request()
    totals = end_request(token2)
    assert totals["llm_calls"] == 0
    assert totals["input_tokens"] == 0
