"""Session lifecycle and per-session estimation endpoints.

The session_id returned by :func:`create_session` is the handle the client
must echo on every follow-up request that wants to share conversation history
and accumulated project metadata across pages or HTTP calls.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, UploadFile
from pydantic import BaseModel, Field

from app.dependencies import get_estimation_service, get_llm_wrapper, get_session_store
from app.guardrails import InputGuardrailViolation
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    EstimationResponse,
    OutputFormat,
    ProjectType,
)
from app.schemas.log import (
    SessionCreated,
    SessionEstimateBlockedByInputGuardrail,
    SessionEstimateEndpointError,
    SessionEstimateRequest,
)
from app.services.attachments import (
    AttachmentExtractionError,
    extract_attachment_text,
    merge_transcript_and_attachments,
)
from app.services.estimation import EstimationService
from app.services.llm_wrapper import LLMWrapper
from app.services.metadata_extractor import extract_project_metadata
from app.services.sessions import Message, ProjectMetadata, SessionStore

router = APIRouter(prefix="/api/v1", tags=["sessions"])


class SessionCreatedResponse(BaseModel):
    session_id: str = Field(description="UUID v4 identifying the new conversation session.")


class SessionStateResponse(BaseModel):
    """Snapshot of a session: accumulated metadata + recent history.

    The metadata block reflects what the assistant has gathered so far (memory).
    The messages list reflects the sliding-window history (recent turns) so
    clients can render both panels and make the separation between *memory*
    and *history* visible to the user.
    """

    session_id: str
    project_metadata: ProjectMetadata
    messages: list[Message]


@router.get("/sessions/{session_id}", response_model=SessionStateResponse)
def get_session_state(
    session_id: str = Path(description="Session id obtained from POST /api/v1/sessions."),
    store: SessionStore = Depends(get_session_store),
) -> SessionStateResponse:
    """Return the current project_metadata and history for a session.

    Used by the UI to render the *memory* panel (project_metadata) and the
    *history* panel (sliding-window messages) after each interaction.
    """
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    return SessionStateResponse(
        session_id=session.session_id,
        project_metadata=session.metadata,
        messages=session.history.messages,
    )


@router.post("/sessions", response_model=SessionCreatedResponse, status_code=201)
def create_session(
    store: SessionStore = Depends(get_session_store),
) -> SessionCreatedResponse:
    """Create a new conversation session and return its id.

    The client should keep this ``session_id`` and send it on subsequent
    requests so the server can reuse the same ``ConversationHistory`` and
    ``ProjectMetadata`` across calls.
    """
    session = store.get_or_create()
    SessionCreated(session_id=session.session_id).emit()
    return SessionCreatedResponse(session_id=session.session_id)


@router.post(
    "/sessions/{session_id}/estimate",
    response_model=EstimationResponse,
)
def estimate_in_session(
    session_id: str = Path(description="Session id obtained from POST /api/v1/sessions."),
    transcript: str = Form(
        ...,
        min_length=20,
        max_length=80_000,
        description="Conversation transcript or free-form project description.",
    ),
    attachments: list[UploadFile] | None = File(
        default=None,
        description="Optional supporting documents (PDF, DOCX, MD, TXT) parsed server-side.",
    ),
    project_type: ProjectType = Form(default=ProjectType.WEB_SAAS),
    detail_level: DetailLevel = Form(default=DetailLevel.MEDIUM),
    output_format: OutputFormat = Form(default=OutputFormat.PHASES_TABLE),
    store: SessionStore = Depends(get_session_store),
    service: EstimationService = Depends(get_estimation_service),
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
) -> EstimationResponse:
    """Run an estimation inside an existing session, optionally enriched with PDFs.

    ``multipart/form-data`` contract:

    * ``transcript`` (str, required) — the conversation transcript or
      description text.
    * ``attachments`` (list[UploadFile], optional) — supporting docs (PDF,
      DOCX, MD, TXT). The bytes never reach the LLM provider: each file is
      parsed locally and the extracted text is concatenated to ``transcript``
      with a ``--- attachment: <filename> ---`` separator before the prompt
      is built.

    Errors:

    * 404 if ``session_id`` is unknown.
    * 422 if an attachment cannot be parsed or its type is unsupported.
    * 400 / 502 mirror the standard ``/estimate`` failure modes.
    """
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")

    extracted: list[tuple[str, str]] = []
    for upload in attachments or []:
        if not upload.filename:
            continue
        data = upload.file.read()
        if not data:
            continue
        try:
            markdown = extract_attachment_text(filename=upload.filename, data=data)
        except AttachmentExtractionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        extracted.append((upload.filename, markdown))

    description = merge_transcript_and_attachments(transcript, extracted)

    SessionEstimateRequest(
        session_id=session_id,
        transcript_chars=len(transcript),
        attachments=len(extracted),
        description_chars=len(description),
    ).emit()

    request = EstimationRequest(
        description=description,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
    )

    try:
        response = service.estimate(request, project_metadata=session.metadata)
    except InputGuardrailViolation as exc:
        SessionEstimateBlockedByInputGuardrail(
            session_id=session_id,
            reason=exc.reason,
        ).emit()
        raise HTTPException(
            status_code=400,
            detail={"reason": exc.reason, "message": exc.message},
        ) from exc
    except Exception as exc:
        SessionEstimateEndpointError(
            session_id=session_id,
            error=str(exc),
            error_type=type(exc).__name__,
        ).emit()
        raise HTTPException(status_code=502, detail="Upstream LLM call failed") from exc

    assistant_reply_json = response.result.model_dump_json()
    session.history.add("user", description)
    session.history.add("assistant", assistant_reply_json)

    session.metadata = extract_project_metadata(
        wrapper=wrapper,
        current=session.metadata,
        user_message=description,
        assistant_reply=assistant_reply_json,
    )
    session.touch()

    return response
