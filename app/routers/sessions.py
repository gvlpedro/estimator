"""Session lifecycle and per-session estimation endpoints.

The session_id returned by :func:`create_session` is the handle the client
must echo on every follow-up request that wants to share conversation history
and accumulated project metadata across pages or HTTP calls.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, UploadFile
from pydantic import BaseModel, Field, ValidationError

from app.config import get_settings
from app.dependencies import get_ai_service_client, get_estimation_service, get_session_store
from app.guardrails import InputGuardrailViolation
from app.schemas.estimation import (
    DetailLevel,
    EstimationResponse,
    OutputFormat,
    ProjectType,
)
from app.schemas.log import (
    RetrievalContextAttached,
    RetrievalContextSkipped,
    SessionCreated,
    SessionEstimateBlockedByInputGuardrail,
    SessionEstimateEndpointError,
    SessionEstimateRequest,
)
from app.schemas.search import SearchResult
from app.services.ai_client import AIServiceClient, AIServiceError
from app.services.attachments import (
    AttachmentExtractionError,
    extract_attachment_text,
    merge_transcript_and_attachments,
)
from app.services.estimation import EstimationService
from app.services.sessions import Message, ProjectMetadata, SessionStore

router = APIRouter(prefix="/api/v1", tags=["sessions"])


class SessionCreatedResponse(BaseModel):
    session_id: str = Field(description="UUID v4 identifying the new conversation session.")


class SessionStateResponse(BaseModel):
    """Snapshot of a session: accumulated metadata + recent history + debug counters.

    The metadata block reflects what the assistant has gathered so far (memory).
    The messages list reflects the sliding-window history (recent turns).

    The debug counters expose the observability surface for the conversational
    pipeline: how many messages are kept in the window, how many anchors the
    compression policy has promoted, how large the rolling summary is, and the
    last tier the resolver picked together with the rule that fired. The
    anchor/summary counters stay at 0 and the tier fields at ``None`` until the
    matching subsystems (compression, tier resolver) are wired in — the shape
    is stable so the UI debug panel can render the same fields end-to-end.
    """

    session_id: str
    project_metadata: ProjectMetadata
    messages: list[Message]
    message_count: int
    anchors_count: int = 0
    summary_chars: int = 0
    last_resolved_tier: str | None = None
    last_tier_rule: str | None = None


@router.get("/sessions/{session_id}", response_model=SessionStateResponse)
def get_session_state(
    session_id: str = Path(description="Session id obtained from POST /api/v1/sessions."),
    store: SessionStore = Depends(get_session_store),
) -> SessionStateResponse:
    """Return the current memory, history and debug counters for a session.

    Used by the UI to render the *memory* panel (project_metadata), the
    *history* panel (sliding-window messages) and the *debug* panel
    (message_count, anchors_count, summary_chars, last_resolved_tier,
    last_tier_rule) after each interaction.
    """
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    history = session.history
    anchors = getattr(history, "anchors", []) or []
    summary = getattr(history, "summary", None) or ""
    return SessionStateResponse(
        session_id=session.session_id,
        project_metadata=session.metadata,
        messages=history.messages,
        message_count=len(history.messages),
        anchors_count=len(anchors),
        summary_chars=len(summary),
        last_resolved_tier=session.last_resolved_tier,
        last_tier_rule=session.last_tier_rule,
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
    ai_client: AIServiceClient = Depends(get_ai_service_client),
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

    max_attachment_chars = get_settings().MAX_ATTACHMENT_CHARS
    extracted: list[tuple[str, str]] = []
    for upload in attachments or []:
        if not upload.filename:
            continue
        data = upload.file.read()
        if not data:
            continue
        try:
            markdown = extract_attachment_text(
                filename=upload.filename,
                data=data,
                max_chars=max_attachment_chars,
            )
        except AttachmentExtractionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        extracted.append((upload.filename, markdown))

    description = merge_transcript_and_attachments(transcript, extracted)
    attachments_total_chars = sum(len(text) for _, text in extracted)

    SessionEstimateRequest(
        session_id=session_id,
        transcript_chars=len(transcript),
        attachments=len(extracted),
        description_chars=len(description),
    ).emit()

    def fetch_retrieved_context() -> list[dict] | None:
        """Best-effort retrieval of comparable historical-budget chunks.

        Invoked by the estimation pipeline only on a cache miss, so cache hits
        never pay the AI-service round-trip. The estimation must keep working
        when the AI service is down or returns malformed data, so any failure
        degrades to "no retrieved context" instead of an error. The [:2000]
        cap matches the AI service's SearchRequest.query limit — a longer
        transcript gets silently truncated for retrieval purposes.
        """
        try:
            search_body = ai_client.search_sync(query=description[:2000], k=5)
            results = [
                SearchResult.model_validate(item).model_dump()
                for item in search_body.get("results") or []
            ]
        except (AIServiceError, ValidationError) as exc:
            RetrievalContextSkipped(session_id=session_id, error=str(exc)).emit()
            return None
        if not results:
            RetrievalContextSkipped(session_id=session_id, error="no results").emit()
            return None
        RetrievalContextAttached(
            session_id=session_id,
            chunks=len(results),
            top_distance=results[0]["distance"],
        ).emit()
        return results

    try:
        return service.estimate_conversational(
            session=session,
            enriched_transcript=description,
            attachments_total_chars=attachments_total_chars,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            retrieved_context_provider=fetch_retrieved_context,
        )
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
