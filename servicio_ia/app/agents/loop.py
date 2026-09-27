"""Manual agent loop over the OpenAI Responses API (Session 12).

No agentic framework and no reliance on the Responses API's own built-in
tool-execution behavior: this module drives ``client.responses.create`` by
hand -- read ``response.output``, run whichever ``function_call`` items
appear, feed their results back as ``function_call_output``, call again with
``previous_response_id``. Driving that loop explicitly (rather than letting
the API execute tools internally) is the point of the exercise: every
reasoning -> act -> observe step becomes one entry in the trace.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from openai import BadRequestError, OpenAI

from app.agents.schemas import AgentResult, TraceStep
from app.agents.tools import TOOLS, calculate_estimate

SYSTEM_PROMPT = """\
You are a senior software estimation consultant. You are given the transcript of a
discovery meeting with a client, and your job is to produce a structured effort
estimate (in hours), broken down by component, grounded in the firm's own historical
budgets.

Work in this order, using your own judgment at each step -- the transcript decides
how many components there are and how they are described, not a fixed script:

1. Read the transcript and identify the distinct components of work it describes. A
   component is a piece of scope that would reasonably be estimated and budgeted on
   its own (e.g. a backend service, a mobile app, a third-party integration, a
   reporting dashboard). Do not merge unrelated pieces of scope into one component,
   and do not split one coherent piece of scope into several.
2. For EACH component, call search_budgets with a query describing that component
   specifically -- sector, technology and what it does, if the transcript states
   them. One call per component; never bundle several components into a single
   query.
3. If a search returns no useful reference (empty or clearly unrelated results), try
   again with a broader or differently-worded query before giving up on that
   component.
4. Once you have searched for every component, call calculate_estimate exactly once
   with all components together, using the "estimated_hours" figures from your
   search results as each component's reference_amounts.
5. Present the final estimate to the user: the per-component breakdown, the total,
   and a short note on any component that came back unbudgeted (no historical
   reference found) so the client knows where the estimate is weakest.

You have two tools available: search_budgets (retrieves relevant historical budget
items for one component) and calculate_estimate (deterministically computes the
hours breakdown and total from the references you have gathered -- it does no
searching and no judgment of its own). Use them as described above; do not invent
numbers that did not come from search_budgets or calculate_estimate.
"""

DEFAULT_MAX_ITERATIONS = 12


@dataclass
class ToolContext:
    """Runtime dependency search_budgets needs -- swapped for --stub vs the real DB."""

    search_budgets: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def _create_response(client: OpenAI, *, model: str, effort: str, **kwargs: Any):
    """Request a response, asking for a reasoning summary; degrade gracefully.

    Reasoning summaries need org verification on some accounts -- a rejected
    ``summary`` field would otherwise take down the whole run over what is,
    for this exercise, a nice-to-have in the trace.
    """
    try:
        return client.responses.create(
            model=model, reasoning={"effort": effort, "summary": "auto"}, **kwargs
        )
    except BadRequestError:
        return client.responses.create(model=model, reasoning={"effort": effort}, **kwargs)


def _extract_reasoning(output: Any) -> str:
    """Join whatever reasoning-summary text items the API attached to this response.

    Empty more often than not: a model that goes straight into a tool call
    without extra deliberation returns a "reasoning" item with an empty
    ``summary`` list (verified directly against the API, independent of
    ``effort`` level) -- it is not a sign anything is broken. Callers must
    treat "" as "no summary available" and fall back to
    ``_fallback_reasoning``, not as an error.
    """
    parts: list[str] = []
    for item in output:
        if getattr(item, "type", None) == "reasoning":
            for entry in getattr(item, "summary", None) or []:
                text = getattr(entry, "text", None)
                if text:
                    parts.append(text)
    return " ".join(parts).strip()


def _fallback_reasoning(name: str, args: dict[str, Any]) -> str:
    """Describe a tool call from its own arguments when the API gave no reasoning summary.

    Not a guess at the model's internal deliberation -- just a plain,
    transparent restatement of what the call itself says it is doing, so the
    trace's reasoning column is never blank.
    """
    if name == "search_budgets":
        note = f'Searching historical budgets for: "{args["query"]}".'
        active_filters = {k: v for k, v in (args.get("filters") or {}).items() if v}
        if active_filters:
            note += f" Narrowed by {active_filters}."
        return note
    if name == "calculate_estimate":
        count = len(args.get("components", []))
        return f"Computing the hours breakdown and total for {count} component(s) from the gathered references."
    return f"Calling {name}."


def _format_action(name: str, args: dict[str, Any]) -> str:
    rendered = ", ".join(f"{key}={value!r}" for key, value in args.items())
    return f"{name}({rendered})"


async def _execute_tool(name: str, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    if name == "search_budgets":
        return await ctx.search_budgets(args)
    if name == "calculate_estimate":
        return calculate_estimate(args)
    raise ValueError(f"Unknown tool call from the model: {name!r}")


async def run_agent(
    transcript: str,
    *,
    model: str,
    effort: str,
    tool_ctx: ToolContext,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> AgentResult:
    """Drive the reason -> act -> observe loop to completion.

    Raises RuntimeError if the model is still requesting tool calls after
    ``max_iterations`` rounds -- a safeguard against a runaway loop, not a
    normal outcome.
    """
    client = OpenAI()
    trace: list[TraceStep] = []
    last_estimate: dict[str, Any] | None = None

    response = _create_response(
        client,
        model=model,
        effort=effort,
        instructions=SYSTEM_PROMPT,
        input=transcript,
        tools=TOOLS,
    )

    rounds = 0
    while True:
        calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
        if not calls:
            break

        rounds += 1
        if rounds > max_iterations:
            raise RuntimeError(
                f"Agent did not converge within {max_iterations} rounds "
                "(still requesting tool calls) -- treat as a safeguard trip, not a "
                "normal outcome."
            )

        reasoning_text = _extract_reasoning(response.output)
        outputs: list[dict[str, Any]] = []
        for call in calls:
            args = json.loads(call.arguments)
            result = await _execute_tool(call.name, args, tool_ctx)
            if call.name == "calculate_estimate":
                last_estimate = result

            trace.append(
                TraceStep(
                    step=len(trace) + 1,
                    reasoning=reasoning_text or _fallback_reasoning(call.name, args),
                    action=_format_action(call.name, args),
                    observation=result.get("summary", json.dumps(result, ensure_ascii=False)),
                )
            )
            outputs.append(
                {"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(result)}
            )

        response = _create_response(
            client,
            model=model,
            effort=effort,
            previous_response_id=response.id,
            input=outputs,
            tools=TOOLS,
        )

    return AgentResult(trace=trace, estimate=last_estimate, final_response=response.output_text)
