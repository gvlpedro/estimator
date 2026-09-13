"""HTTP surface for the multi-agent estimation graph.

The core contract (``estimate`` + ``status``) is what the business backend
already depends on -- kept stable. The only new surface is the
human-in-the-loop pair: ``POST /estimate`` can now come back with
``status="awaiting_human_review"`` instead of a final estimate, and
``POST /estimate/{estimation_id}/resume`` is where a reviewer's decision
re-enters the graph from the exact checkpoint it paused at.
"""

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from langgraph.types import Command
from pydantic import BaseModel, Field

router = APIRouter()


class EstimateRequest(BaseModel):
    transcript: str = Field(min_length=1, description="Raw discovery-meeting transcript.")
    estimation_id: str = Field(description="Used as the checkpointer's thread_id.")


class ReviewPayload(BaseModel):
    """What a human reviewer needs to see to make the call."""

    reason: str
    estimate: dict[str, Any] | None = None
    confidence: float | None = None
    validation: dict[str, Any] | None = None


class EstimateResponse(BaseModel):
    estimation_id: str
    estimate: dict[str, Any] | None = None
    status: str | None = None
    review: ReviewPayload | None = None


class ResumeRequest(BaseModel):
    outcome: Literal["approved", "adjusted", "rejected"]
    notes: str | None = None
    adjusted_estimate: dict[str, Any] | None = None


def _response_from_result(estimation_id: str, result: dict[str, Any]) -> EstimateResponse:
    interrupts = result.get("__interrupt__")
    if interrupts:
        return EstimateResponse(
            estimation_id=estimation_id,
            status="awaiting_human_review",
            review=ReviewPayload(**interrupts[0].value),
        )
    return EstimateResponse(
        estimation_id=estimation_id,
        estimate=result.get("estimate"),
        status=result.get("status"),
    )


@router.post("/estimate", response_model=EstimateResponse)
async def create_estimate(payload: EstimateRequest, request: Request) -> EstimateResponse:
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": payload.estimation_id}}
    result = await graph.ainvoke(
        {"transcript": payload.transcript, "estimation_id": payload.estimation_id},
        config,
    )
    return _response_from_result(payload.estimation_id, result)


@router.post("/estimate/{estimation_id}/resume", response_model=EstimateResponse)
async def resume_estimate(
    estimation_id: str, payload: ResumeRequest, request: Request
) -> EstimateResponse:
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": estimation_id}}

    pending = await graph.aget_state(config)
    if not pending.next:
        raise HTTPException(
            status_code=409, detail="This estimation is not awaiting human review."
        )

    result = await graph.ainvoke(Command(resume=payload.model_dump()), config)
    return _response_from_result(estimation_id, result)
