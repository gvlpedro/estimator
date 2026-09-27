"""Request-level metrics for the AI service (Session 15/16 evaluation deliverable).

A single HTTP middleware (see ``app/main.py``) emits one structured
``request_completed`` log line per request, carrying latency, token usage,
derived cost, and the HTTP status. Doing it in middleware rather than inside
individual routes is what makes it complete: a request that fails before
reaching a router (auth rejection, a routing error) still gets counted,
which is the only way the dashboard's error rate can be trusted.

Token usage, though, is a business-logic detail that varies per route and
per graph node -- the middleware itself has no idea a model was called. This
module bridges the two with a per-request ``ContextVar`` accumulator:

- ``begin_request()`` opens it, called by the middleware at request start.
- ``record_token_usage()`` appends to it, called by whatever code actually
  invokes a model (currently only ``app.lgraph.nodes.requirements_extractor``).
  It is a no-op with no active request (e.g. a unit test that calls a node
  directly) -- there is nothing to accumulate into, and that is not an error.
- ``end_request()`` closes it and returns the totals, called by the
  middleware in a ``finally`` block.

Plain ``structlog`` calls, matching this service's existing convention in
``app/lgraph/nodes.py`` (unlike the root business backend's typed
``LogEvent`` classes in ``app/schemas/log.py`` -- servicio_ia has no such
module and this stays consistent with what's already here).

Cost is illustrative: ``COST_PER_1K_*_TOKENS`` are placeholder USD rates for
``GRAPH_MODEL`` (see ``app.lgraph.nodes``), override them via env vars once
real per-token pricing for the deployed model is known -- this module does
not call any pricing API.
"""

from __future__ import annotations

import contextvars
import os
from typing import Any

COST_PER_1K_INPUT_TOKENS = float(os.getenv("COST_PER_1K_INPUT_TOKENS", "0.00025"))
COST_PER_1K_OUTPUT_TOKENS = float(os.getenv("COST_PER_1K_OUTPUT_TOKENS", "0.002"))

# ``default=None`` (not ``[]``) so ``record_token_usage`` can tell "no request
# in flight on this context" apart from "a request with zero calls so far" --
# a mutable default would also be shared across every context that never
# calls ``begin_request()``, which is its own bug.
_token_usage: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "_token_usage", default=None
)


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Derive a USD cost from token counts using the configured per-1K rates."""
    cost = (input_tokens / 1000) * COST_PER_1K_INPUT_TOKENS + (
        output_tokens / 1000
    ) * COST_PER_1K_OUTPUT_TOKENS
    return round(cost, 6)


def begin_request() -> contextvars.Token:
    """Open a fresh per-request token-usage accumulator.

    Starlette runs each request in its own ``asyncio.Task`` with its own
    context, so this is task-local: concurrent requests never see each
    other's accumulator, and nothing needs a lock.
    """
    return _token_usage.set([])


def record_token_usage(*, agent: str, call: str, input_tokens: int, output_tokens: int) -> None:
    """Record one model call's usage against the current request, if any."""
    bucket = _token_usage.get()
    if bucket is None:
        return
    bucket.append(
        {
            "agent": agent,
            "call": call,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
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
        "cost_usd": estimate_cost_usd(input_tokens, output_tokens),
    }
