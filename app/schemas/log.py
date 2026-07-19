"""Eventos de logging estructurado tipados con Pydantic.

Cada evento que la app emite a structlog tiene aquí su propia clase. Así:

* el nombre del evento y el nivel (``info``/``warning``/``error``) viven en
  un único sitio (``ClassVar``) y no se pueden tipear mal en cada uso;
* los campos del evento están validados — un ``log.info("session_created")``
  sin ``session_id`` deja de compilar mentalmente;
* este fichero es el inventario de qué eventos produce la app, útil para
  alertas/paneles.

Patrón de uso::

    from app.schemas.log import SessionCreated
    SessionCreated(session_id=session.session_id).emit()

Si ya existe un logger atado a contexto, se le pasa::

    SessionCreated(session_id=sid).emit(log)
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

import structlog
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

LogLevel = Literal["debug", "info", "warning", "error"]
GuardrailReason = Literal["moderation", "prompt_injection", "pii"]

MAX_ERROR_CHARS = 400
MAX_INJECTION_MATCH_CHARS = 80


def _truncate(limit: int):
    def _inner(value: str) -> str:
        return value[:limit]

    return _inner


TruncatedError = Annotated[str, BeforeValidator(_truncate(MAX_ERROR_CHARS))]
TruncatedInjectionMatch = Annotated[str, BeforeValidator(_truncate(MAX_INJECTION_MATCH_CHARS))]


class LogEvent(BaseModel):
    """Base para todos los eventos estructurados.

    Las subclases declaran ``event`` y ``level`` como ``ClassVar`` (no son
    campos del modelo, así que no aparecen en ``model_dump`` y no contaminan
    el payload que recibe structlog).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event: ClassVar[str]
    level: ClassVar[LogLevel] = "info"

    def emit(self, logger: structlog.stdlib.BoundLogger | None = None) -> None:
        log = logger if logger is not None else structlog.get_logger()
        # ``exclude_none`` so that opcional fields not set por el emisor no
        # aparezcan como ``foo=None`` en la salida de structlog.
        getattr(log, self.level)(self.event, **self.model_dump(exclude_none=True))


# ---------------------------------------------------------------------------
# Ciclo de vida de la app
# ---------------------------------------------------------------------------


class ApplicationStarted(LogEvent):
    event: ClassVar[str] = "application_started"
    environment: str


class ApplicationShutdown(LogEvent):
    event: ClassVar[str] = "application_shutdown"


# ---------------------------------------------------------------------------
# Dependencias
# ---------------------------------------------------------------------------


class OpenAIClientDisabled(LogEvent):
    event: ClassVar[str] = "openai_client_disabled"
    level: ClassVar[LogLevel] = "warning"
    reason: str


# ---------------------------------------------------------------------------
# Router de sesiones
# ---------------------------------------------------------------------------


class SessionCreated(LogEvent):
    event: ClassVar[str] = "session_created"
    session_id: str


class SessionEstimateRequest(LogEvent):
    event: ClassVar[str] = "session_estimate_request"
    session_id: str
    transcript_chars: int = Field(ge=0)
    attachments: int = Field(ge=0)
    description_chars: int = Field(ge=0)


class RetrievalContextAttached(LogEvent):
    event: ClassVar[str] = "retrieval_context_attached"
    session_id: str
    chunks: int = Field(ge=0)
    top_distance: float | None = None


class RetrievalContextSkipped(LogEvent):
    event: ClassVar[str] = "retrieval_context_skipped"
    session_id: str
    error: str


class SessionEstimateBlockedByInputGuardrail(LogEvent):
    event: ClassVar[str] = "session_estimate_blocked_by_input_guardrail"
    session_id: str
    reason: GuardrailReason


class SessionEstimateEndpointError(LogEvent):
    event: ClassVar[str] = "session_estimate_endpoint_error"
    level: ClassVar[LogLevel] = "error"
    session_id: str
    error: TruncatedError
    error_type: str


# ---------------------------------------------------------------------------
# Router de estimaciones
# ---------------------------------------------------------------------------


class EstimationRequestReceived(LogEvent):
    event: ClassVar[str] = "estimation_request_received"
    project_type: str
    detail_level: str
    output_format: str
    description_chars: int = Field(ge=0)
    prompt_version: str


class EstimationBlockedByInputGuardrail(LogEvent):
    event: ClassVar[str] = "estimation_blocked_by_input_guardrail"
    reason: GuardrailReason
    message: str


class EstimationEndpointError(LogEvent):
    event: ClassVar[str] = "estimation_endpoint_error"
    level: ClassVar[LogLevel] = "error"
    error: TruncatedError
    error_type: str


class EstimateStreamFailed(LogEvent):
    event: ClassVar[str] = "estimate_stream_failed"
    level: ClassVar[LogLevel] = "error"
    error: str
    error_type: str


# ---------------------------------------------------------------------------
# Guardrails de entrada
# ---------------------------------------------------------------------------


class ModerationCallFailed(LogEvent):
    event: ClassVar[str] = "moderation_call_failed"
    level: ClassVar[LogLevel] = "warning"
    error_type: str
    error: str


class ModerationFlagged(LogEvent):
    event: ClassVar[str] = "moderation_flagged"
    categories: list[str]


class PromptInjectionDetected(LogEvent):
    event: ClassVar[str] = "prompt_injection_detected"
    pattern: str
    match: TruncatedInjectionMatch


# ---------------------------------------------------------------------------
# Guardrails de salida
# ---------------------------------------------------------------------------


class EnforceScopeResponseFiltering(LogEvent):
    event: ClassVar[str] = "enforce_scope_response_filtering"
    confidence_pct: int = Field(ge=0, le=100)
    original_summary_chars: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Renderizado de prompts
# ---------------------------------------------------------------------------


class PromptRendered(LogEvent):
    event: ClassVar[str] = "prompt_rendered"
    prompt_version: str
    content_hash: str
    system_chars: int = Field(ge=0)
    user_chars: int = Field(ge=0)
    has_reference_projects: bool
    n_reference_projects: int = Field(ge=0)
    has_project_metadata: bool


# ---------------------------------------------------------------------------
# Cache de Redis
# ---------------------------------------------------------------------------


class CacheGetFailed(LogEvent):
    event: ClassVar[str] = "cache_get_failed"
    level: ClassVar[LogLevel] = "warning"
    error: TruncatedError


class CacheHit(LogEvent):
    event: ClassVar[str] = "cache_hit"
    key_prefix: str


class CacheMiss(LogEvent):
    event: ClassVar[str] = "cache_miss"
    key_prefix: str


class CacheStored(LogEvent):
    event: ClassVar[str] = "cache_stored"
    key_prefix: str
    ttl: int = Field(ge=0)


class CacheSetFailed(LogEvent):
    event: ClassVar[str] = "cache_set_failed"
    level: ClassVar[LogLevel] = "warning"
    error: TruncatedError


# ---------------------------------------------------------------------------
# Extracción de project metadata (segundo LLM call por turno)
# ---------------------------------------------------------------------------


class ProjectMetadataExtractionFailed(LogEvent):
    event: ClassVar[str] = "project_metadata_extraction_failed"
    level: ClassVar[LogLevel] = "warning"
    error: TruncatedError
    error_type: str


class ProjectMetadataUpdated(LogEvent):
    event: ClassVar[str] = "project_metadata_updated"
    had_project_name: bool
    has_project_name: bool
    prev_n_technologies: int = Field(ge=0)
    next_n_technologies: int = Field(ge=0)
    had_agreed_scope: bool
    has_agreed_scope: bool


# ---------------------------------------------------------------------------
# Adjuntos parseados localmente
# ---------------------------------------------------------------------------


class AttachmentExtractionFailed(LogEvent):
    event: ClassVar[str] = "attachment_extraction_failed"
    level: ClassVar[LogLevel] = "warning"
    filename: str
    size_bytes: int = Field(ge=0)
    error: TruncatedError


class AttachmentExtracted(LogEvent):
    event: ClassVar[str] = "attachment_extracted"
    filename: str
    size_bytes: int = Field(ge=0)
    text_chars: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Orquestador EstimationService
# ---------------------------------------------------------------------------


class EstimationCacheHit(LogEvent):
    event: ClassVar[str] = "estimation_cache_hit"
    kind: Literal["exact", "semantic"]
    key_prefix: str


class EstimationGenerated(LogEvent):
    event: ClassVar[str] = "estimation_generated"
    prompt_version: str
    confidence_pct: int = Field(ge=0, le=100)
    total_cost_eur: int = Field(ge=0)
    phases: int = Field(ge=1)
    model: str
    provider: str
    latency_ms: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Per-turn observability aggregate
# ---------------------------------------------------------------------------


CacheHitKind = Literal["none", "exact", "semantic"]


class TurnObserved(LogEvent):
    """Single per-turn rollup of everything observable about one estimation.

    Emitted exactly once, just before returning from
    ``EstimationService.estimate_conversational``. Pre-existing per-call
    events (``cache_hit``, ``llm_structured_call_completed``, etc.) still
    fire, but this aggregate lets a CSV extraction pull one row per turn
    without joining across event types.
    """

    event: ClassVar[str] = "turn_observed"
    turn_index: int = Field(ge=1)
    session_id: str
    enriched_transcript_chars: int = Field(ge=0)
    attachments_total_chars: int = Field(ge=0)
    messages_in_window: int = Field(ge=0)
    anchors_count: int = Field(ge=0)
    summary_chars: int = Field(ge=0)
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    latency_ms: int = Field(ge=0)
    cache_hit_kind: CacheHitKind
    last_resolved_tier: str | None = None


# ---------------------------------------------------------------------------
# Actor-Critic-Boss orchestrator
# ---------------------------------------------------------------------------


class ACBRequestReceived(LogEvent):
    event: ClassVar[str] = "acb_request_received"
    project_type: str
    detail_level: str
    output_format: str
    description_chars: int = Field(ge=0)
    max_iterations: int = Field(ge=1)


class ACBActorDraft(LogEvent):
    event: ClassVar[str] = "acb_actor_draft"
    with_critic_feedback: bool
    issues_in_feedback: int = Field(ge=0)
    confidence_pct: int = Field(ge=0, le=100)
    total_cost_eur: int = Field(ge=0)


class ACBCompleted(LogEvent):
    event: ClassVar[str] = "acb_completed"
    final_decision: str
    iterations_run: int = Field(ge=1)
    confidence_pct: int = Field(ge=0, le=100)
    total_cost_eur: int = Field(ge=0)


# ---------------------------------------------------------------------------
# LLMWrapper (llamadas a litellm)
# ---------------------------------------------------------------------------


class LlmCallStarted(LogEvent):
    event: ClassVar[str] = "llm_call_started"
    mode: Literal["blocking"]
    model: str
    has_thinking: bool


class LlmCallCompleted(LogEvent):
    event: ClassVar[str] = "llm_call_completed"
    model: str
    provider: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    latency_ms: int = Field(ge=0)
    finish_reason: str


class LlmCallFailed(LogEvent):
    """``llm_call_failed`` lo emiten tanto el wrapper (con ``latency_ms``) como
    el orquestador ``llm_service`` (sin él): ``latency_ms`` es opcional para
    cubrir los dos sitios sin duplicar el nombre del evento."""

    event: ClassVar[str] = "llm_call_failed"
    level: ClassVar[LogLevel] = "error"
    error: TruncatedError
    error_type: str
    latency_ms: int | None = Field(default=None, ge=0)


class LlmStructuredCallStarted(LogEvent):
    event: ClassVar[str] = "llm_structured_call_started"
    model: str
    response_model: str


class LlmStructuredCallCompleted(LogEvent):
    event: ClassVar[str] = "llm_structured_call_completed"
    model: str
    provider: str
    latency_ms: int = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)


class LlmStructuredCallFailed(LogEvent):
    event: ClassVar[str] = "llm_structured_call_failed"
    level: ClassVar[LogLevel] = "error"
    error: TruncatedError
    error_type: str
    latency_ms: int = Field(ge=0)


class StreamCacheHit(LogEvent):
    event: ClassVar[str] = "stream_cache_hit"
    chars: int = Field(ge=0)


class LlmStreamStarted(LogEvent):
    event: ClassVar[str] = "llm_stream_started"
    model: str


class LlmStreamCompleted(LogEvent):
    event: ClassVar[str] = "llm_stream_completed"
    latency_ms: int = Field(ge=0)
    chars: int = Field(ge=0)


class LlmStreamFailed(LogEvent):
    event: ClassVar[str] = "llm_stream_failed"
    level: ClassVar[LogLevel] = "error"
    error_type: str
    error: str
    latency_ms: int = Field(ge=0)


class ThinkingBudgetIgnoredForProvider(LogEvent):
    event: ClassVar[str] = "thinking_budget_ignored_for_provider"
    level: ClassVar[LogLevel] = "warning"
    provider: str
    model: str


# ---------------------------------------------------------------------------
# LLM service (orquestación de preprocesado y generación)
# ---------------------------------------------------------------------------


class ExtractingRequirements(LogEvent):
    event: ClassVar[str] = "extracting_requirements"
    model_override: str | None = None


class GeneratingEstimation(LogEvent):
    event: ClassVar[str] = "generating_estimation"
    model_override: str | None = None
    preprocessing: str
    example_format: str
    num_examples: int = Field(ge=0)
    use_examples: bool
    max_tokens: int = Field(ge=1)
    thinking_budget: int | None = Field(default=None, ge=0)


# ---------------------------------------------------------------------------
# Demo ejecutable
#
# ``uv run python -m app.schemas.log`` muestra cómo se usa el schema en
# cuatro escenarios. No forma parte de la app: sólo sirve de documentación
# viva — si cambias el contrato de ``LogEvent`` y rompes el demo, lo notas.
# ---------------------------------------------------------------------------


def _demo() -> None:
    from pydantic import ValidationError

    # Renderer humano para que la salida sea legible en consola.
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )

    print("\n--- 1) Evento info: nivel y nombre vienen de la clase ---")
    # No hace falta recordar "session_created" ni "info": la clase los fija.
    SessionCreated(session_id="sess-42").emit()

    print("\n--- 2) Eventos warning y error: cambia el nivel solo ---")
    # ``level`` es un ClassVar; estos dos salen como warning y error sin que
    # el call site tenga que elegir el método de structlog.
    OpenAIClientDisabled(reason="no_api_key").emit()
    EstimationEndpointError(
        error="upstream timeout after 30s",
        error_type="TimeoutError",
    ).emit()

    print("\n--- 3) Truncado automático de errores largos ---")
    # El campo ``error`` está anotado como ``TruncatedError`` (400 chars).
    # El emisor pasa el ``str(exc)`` tal cual y el schema recorta.
    long_msg = "boom! " * 2
    ev = SessionEstimateEndpointError(
        session_id="sess-42",
        error=long_msg,
        error_type="RuntimeError",
    )
    print(f"chars originales={len(long_msg)} -> tras validar={len(ev.error)}")
    ev.emit()

    print("\n--- 4) Validación: olvidar un campo falla en tiempo de construcción ---")
    # Si un call site se equivoca, el error salta aquí (en el emisor),
    # no en producción cuando ya nadie sabe qué evento se intentó loguear.
    try:
        SessionEstimateRequest(session_id="sess-42")  # faltan los _chars
    except ValidationError as exc:
        missing = [".".join(map(str, e["loc"])) for e in exc.errors()]
        print(f"ValidationError: faltan campos -> {missing}")

    print("\n--- 5) Logger atado a contexto (bind) ---")
    # ``emit(logger=...)`` reutiliza un BoundLogger con contexto compartido,
    # así que todos los eventos heredan el ``request_id`` sin repetirlo.
    request_log = structlog.get_logger().bind(request_id="req-abc-123")
    PromptRendered(
        prompt_version="v1",
        content_hash="deadbeefcafe",
        system_chars=4321,
        user_chars=512,
        has_reference_projects=False,
        n_reference_projects=0,
        has_project_metadata=True,
    ).emit(request_log)
    EstimationGenerated(
        prompt_version="v1",
        confidence_pct=78,
        total_cost_eur=45_000,
        phases=4,
        model="gpt-4o-mini",
        provider="openai",
        latency_ms=2480,
    ).emit(request_log)


if __name__ == "__main__":
    _demo()
