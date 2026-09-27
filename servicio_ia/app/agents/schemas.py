"""Pydantic models for the Session-12 agent's trace and final result.

Kept separate from ``tools.py`` (the tool schemas + implementations) so the
trace format — the exercise's actual deliverable — has one place to change
without touching tool wiring.
"""

from __future__ import annotations

from pydantic import BaseModel


class TraceStep(BaseModel):
    """One reason -> act -> observe iteration of the manual agent loop."""

    step: int
    reasoning: str
    action: str
    observation: str


class AgentResult(BaseModel):
    """What ``run_agent`` returns: the full trace plus the final estimate."""

    trace: list[TraceStep]
    estimate: dict | None
    final_response: str


def format_trace(trace: list[TraceStep]) -> str:
    """Render the trace in the minimal per-step format the exercise asks for.

        STEP 1
          reasoning:   ...
          action:      ...
          observation: ...
    """
    lines: list[str] = []
    for step in trace:
        lines.append(f"STEP {step.step}")
        lines.append(f"  reasoning:   {step.reasoning}")
        lines.append(f"  action:      {step.action}")
        lines.append(f"  observation: {step.observation}")
        lines.append("")
    return "\n".join(lines).rstrip()
