"""Least-privilege tool access + audit trail for the multi-agent graph.

Every business-tool call an agent makes goes through :func:`call_tool`
instead of calling the tool function directly, so two things are always
true: (1) an agent can only ever invoke a tool declared in its own privilege
set -- ``requirements_extractor`` has none, by design, since it only reasons
over text; (2) a full run is reconstructable from the structlog output alone
(which agent, which tool, denied or not, succeeded or failed).
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import structlog

log = structlog.get_logger()

# The declared privilege set per agent. Not a suggestion: call_tool refuses
# anything not listed here, regardless of what the caller passes in.
AGENT_TOOL_PRIVILEGES: dict[str, frozenset[str]] = {
    "requirements_extractor": frozenset(),
    "budget_searcher": frozenset({"search_budgets"}),
    "estimate_generator": frozenset({"calculate_estimate"}),
    "coherence_validator": frozenset({"validate_estimate"}),
    "supervisor": frozenset(),
    "human_review_gate": frozenset(),
    "finalize": frozenset(),
}


class ToolAccessDenied(Exception):
    """Raised when an agent calls a tool outside its declared privilege set."""

    def __init__(self, agent: str, tool_name: str) -> None:
        super().__init__(f"agent {agent!r} has no privilege to call tool {tool_name!r}")
        self.agent = agent
        self.tool_name = tool_name


async def call_tool(
    agent: str, tool_name: str, tool_fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Check privilege, log the attempt, call ``tool_fn`` (sync or async), log the outcome."""
    allowed = AGENT_TOOL_PRIVILEGES.get(agent, frozenset())
    if tool_name not in allowed:
        log.error("agent_tool_denied", agent=agent, tool=tool_name, allowed=sorted(allowed))
        raise ToolAccessDenied(agent, tool_name)

    log.info("agent_tool_call", agent=agent, tool=tool_name)
    try:
        if inspect.iscoroutinefunction(tool_fn):
            result = await tool_fn(*args, **kwargs)
        else:
            result = tool_fn(*args, **kwargs)
    except Exception as exc:
        log.error("agent_tool_failed", agent=agent, tool=tool_name, error=str(exc))
        raise

    log.info("agent_tool_succeeded", agent=agent, tool=tool_name)
    return result
