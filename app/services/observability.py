"""Per-request token/cost accumulator feeding the ``request_completed`` metric.

Mirrors ``servicio_ia/app/observability.py``'s design (same ``ContextVar``
pattern), adapted to this service: here the cost per LLM call is already
computed by ``app/services/llm_wrapper.py`` (``LLMWrapper.complete``/
``complete_structured``, via ``_estimate_cost`` and its ``MODEL_COSTS``
table), so this module does not recompute cost -- it only accumulates what
the wrapper already knows, per request.

- ``begin_request()`` opens the accumulator, called by the middleware
  (``app/main.py``) at request start.
- ``record_token_usage()`` appends one LLM call's usage, called from
  ``LLMWrapper`` right where ``LlmCallCompleted``/``LlmStructuredCallCompleted``
  are emitted (a cache hit never reaches those emit sites, so a cached call
  correctly contributes zero here).
- ``end_request()`` closes the accumulator and returns the totals, called by
  the middleware in a ``finally`` block.

A cache hit is not an error and not "no request" -- ``record_token_usage``
being a no-op with no active accumulator only covers callers outside any
request (e.g. a unit test that calls the wrapper directly).
"""

from __future__ import annotations

import contextvars
from typing import Any

# ``default=None`` (not ``[]``) so ``record_token_usage`` can tell "no
# request in flight on this context" apart from "a request with zero calls
# so far" -- a mutable default would also be shared across every context
# that never calls ``begin_request()``.
_token_usage: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "_token_usage", default=None
)


def begin_request() -> contextvars.Token:
    """Open a fresh per-request token-usage accumulator.

    Starlette runs each request in its own ``asyncio.Task`` with its own
    context, so this is task-local: concurrent requests never see each
    other's accumulator, and nothing needs a lock.
    """
    return _token_usage.set([])


def record_token_usage(*, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
    """Record one LLM call's already-computed usage against the current request, if any."""
    bucket = _token_usage.get()
    if bucket is None:
        return
    bucket.append(
        {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        }
    )


def end_request(token: contextvars.Token) -> dict[str, Any]:
    """Close the current request's accumulator and return its totals.

    Resets the ``ContextVar`` with ``token`` so nothing leaks into whatever
    runs next on this context.
    """
    bucket = _token_usage.get() or []
    input_tokens = sum(u["input_tokens"] for u in bucket)
    output_tokens = sum(u["output_tokens"] for u in bucket)
    _token_usage.reset(token)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "llm_calls": len(bucket),
        "cost_usd": round(sum(u["cost_usd"] for u in bucket), 6),
    }
