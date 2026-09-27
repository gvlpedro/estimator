import structlog
import time
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routers import estimations, search, sessions
from app.schemas.log import ApplicationShutdown, ApplicationStarted, RequestCompleted
from app.services.observability import begin_request, end_request


def configure_logging() -> None:
    """Set up structlog: JSON in production, human-readable in development."""
    settings = get_settings()

    if settings.APP_ENV == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    configure_logging()
    settings = get_settings()
    ApplicationStarted(environment=settings.APP_ENV).emit()
    yield
    ApplicationShutdown().emit()


app = FastAPI(
    title="Software Estimation CAG Service",
    description="AI-powered software estimation service using Cache Augmented Generation architecture",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Excluded from the per-request metric event: the liveness probe, and only
# it. Docker calls /health every 30s, so at that rate it would be most of
# the rows and "the service's p95 latency" would really be the probe's.
_METRICS_EXEMPT_PATHS = frozenset({"/health"})


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Emit one ``request_completed`` event per HTTP request (Session 16).

    Registered as middleware rather than instrumented per-router: a request
    that fails before reaching any router (guardrail rejection before the
    handler, a 404) is still counted, which is the only way the dashboard's
    error rate can be trusted. Token usage comes from
    ``app.services.observability``'s per-request accumulator, fed by
    ``LLMWrapper`` (see ``app/services/llm_wrapper.py``) whichever call this
    request happened to make -- ``/api/v1/estimate``, the session-based
    ``/api/v1/sessions/{id}/estimate``, or none at all (a cache hit, or a
    request that never reached the LLM).
    """
    path = request.url.path
    measured = path not in _METRICS_EXEMPT_PATHS
    token = begin_request() if measured else None
    started = time.perf_counter()
    status = 500  # if call_next raises, the client saw (or would have seen) a 500.
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        if token is not None:
            metrics = end_request(token)
            RequestCompleted(
                path=path,
                method=request.method,
                status=status,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                **metrics,
            ).emit()


app.include_router(estimations.router)
app.include_router(search.router)
app.include_router(sessions.router)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/health")
async def health_check() -> dict:
    """Return service health status."""
    settings = get_settings()
    return {
        "status": "healthy",
        "version": "0.1.0",
        "environment": settings.APP_ENV,
    }
