import asyncio
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from app.dependencies import get_estimation_service, get_llm_wrapper
from app.guardrails import InputGuardrailViolation
from app.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
    StreamEstimationRequest,
)
from app.schemas.log import (
    EstimateStreamFailed,
    EstimationBlockedByInputGuardrail,
    EstimationEndpointError,
    EstimationRequestReceived,
)
from app.services.estimation import EstimationService
from app.services.llm_service import build_system_prompt
from app.services.llm_wrapper import LLMWrapper

router = APIRouter(prefix="/api/v1", tags=["estimations"])

PromptVersion = Literal["v1", "v2"]


@router.post("/estimate", response_model=EstimationResponse)
def create_estimation(
    request: EstimationRequest,
    prompt_version: PromptVersion = Query(
        default="v1",
        description="Which prompt family under app/prompts/estimation/ to use.",
    ),
    service: EstimationService = Depends(get_estimation_service),
) -> EstimationResponse:
    """Run the full structured-estimation pipeline and return the validated result.

    - ``InputGuardrailViolation`` → 400 ``{reason, message}`` so the client can
      render an actionable error (prompt injection, PII, moderation).
    - Anything else from the pipeline → 502 (includes ``InstructorRetryException``
      when the model can't satisfy validators within ``max_retries``).
    """
    EstimationRequestReceived(
        project_type=request.project_type.value,
        detail_level=request.detail_level.value,
        output_format=request.output_format.value,
        description_chars=len(request.description),
        prompt_version=prompt_version,
    ).emit()

    try:
        return service.estimate(request, prompt_version=prompt_version)
    except InputGuardrailViolation as exc:
        EstimationBlockedByInputGuardrail(
            reason=exc.reason,
            message=exc.message,
        ).emit()
        raise HTTPException(
            status_code=400,
            detail={"reason": exc.reason, "message": exc.message},
        ) from exc
    except Exception as exc:
        EstimationEndpointError(
            error=str(exc),
            error_type=type(exc).__name__,
        ).emit()
        raise HTTPException(status_code=502, detail="Upstream LLM call failed") from exc


@router.post("/estimate/stream")
async def create_estimation_stream(
    request: StreamEstimationRequest,
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
) -> EventSourceResponse:
    """Stream a software estimation token by token via Server-Sent Events.

    The streaming path keeps the older transcription-based contract — it is
    used by the legacy demo UI and is not affected by the v1 prompt refactor.
    """
    system_prompt = build_system_prompt()

    async def event_generator() -> AsyncIterator[dict]:
        loop = asyncio.get_running_loop()
        chunks = wrapper.complete_stream(
            system_prompt=system_prompt,
            user_message=request.transcription,
            model_override=request.model,
            max_tokens=request.max_tokens,
        )

        def _next_chunk() -> str | None:
            try:
                return next(chunks)
            except StopIteration:
                return None
            except Exception as exc:  # noqa: BLE001 — surface as SSE error event
                EstimateStreamFailed(error=str(exc), error_type=type(exc).__name__).emit()
                raise

        try:
            while True:
                chunk = await loop.run_in_executor(None, _next_chunk)
                if chunk is None:
                    break
                if chunk:
                    yield {"event": "token", "data": chunk}
            yield {"event": "done", "data": "[DONE]"}
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": str(exc)}

    return EventSourceResponse(event_generator())
