"""HTTP surface for the estimation graph.

The response contract (``estimate`` + ``status``) is what the business
backend depends on -- keep it stable even as the graph gains branches.
"""

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter()


class EstimateRequest(BaseModel):
    transcript: str = Field(min_length=1, description="Raw discovery-meeting transcript.")
    estimation_id: str = Field(description="Used as the checkpointer's thread_id.")


class EstimateResponse(BaseModel):
    estimate: dict[str, Any] | None
    status: str | None


@router.post("/estimate", response_model=EstimateResponse)
async def create_estimate(payload: EstimateRequest, request: Request) -> EstimateResponse:
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": payload.estimation_id}}
    result = await graph.ainvoke(
        {"transcript": payload.transcript, "estimation_id": payload.estimation_id},
        config,
    )
    return EstimateResponse(estimate=result.get("estimate"), status=result.get("status"))
