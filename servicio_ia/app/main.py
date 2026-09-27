"""FastAPI entrypoint for the AI service (servicio_ia)."""

import os
import time
from contextlib import asynccontextmanager

import logfire
import structlog
from fastapi import Depends, FastAPI, Request
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.embedding_pipeline.router import router as embeddings_router
from app.embedding_pipeline.router import search_router
from app.ingest.router import router as ingest_router
from app.lgraph.build import build_graph
from app.lgraph.checkpointer import get_checkpointer_dsn
from app.lgraph.router import router as estimate_router
from app.observability import begin_request, end_request
from app.security import verify_service_token

log = structlog.get_logger()


def configure_logging() -> None:
    """Set up structlog: JSON in production, human-readable in development."""
    if os.getenv("APP_ENV", "development") == "production":
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
    """Application startup and shutdown lifecycle.

    The checkpointer owns the Postgres connection pool for the graph's
    lifetime: opened here (with its tables created on first run via
    ``setup()``), torn down on shutdown alongside everything else.
    """
    configure_logging()
    # "if-token-present": no LOGFIRE_TOKEN -> local-only, no crash on startup;
    # set LOGFIRE_TOKEN in .env to actually ship spans to a Logfire project.
    logfire.configure(service_name="servicio_ia", send_to_logfire="if-token-present")
    logfire.instrument_fastapi(app)
    logfire.instrument_asyncpg()
    logfire.instrument_httpx()  # captures the OpenAI Responses API calls (SDK uses httpx)

    structlog.get_logger().info("application_started", service="servicio_ia")

    async with AsyncPostgresSaver.from_conn_string(get_checkpointer_dsn()) as checkpointer:
        await checkpointer.setup()
        app.state.graph = build_graph(checkpointer)
        yield

    structlog.get_logger().info("application_shutdown", service="servicio_ia")


app = FastAPI(
    title="Estimator AI Service",
    description="Embedding pipeline for historical budgets (RAG groundwork)",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ingest/encode stay open: they're hit by host-side dev tooling
# (scripts/seed_budgets.py, query_examples.py), never by the business
# backend. search and estimate ARE the business backend's call path for a
# live estimation, so they require the shared service token.
# Excluded from the per-request metric event: the liveness probe, and only
# it. Docker calls /health every 30s, so at that rate it would be most of
# the rows and "the service's p95 latency" would really be the probe's.
_METRICS_EXEMPT_PATHS = frozenset({"/health"})


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Emit one ``request_completed`` event per HTTP request (Session 16).

    Registered as middleware rather than instrumented per-router: a request
    that fails before reaching any router (auth rejection, a 404) is still
    counted, which is the only way the dashboard's error rate can be
    trusted. Token usage comes from ``app.observability``'s per-request
    accumulator, fed by whichever graph node happened to call a model.
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
            log.info(
                "request_completed",
                path=path,
                method=request.method,
                status=status,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                **metrics,
            )


app.include_router(ingest_router, prefix="/embeddings")
app.include_router(embeddings_router, prefix="/embeddings")
app.include_router(search_router, dependencies=[Depends(verify_service_token)])
app.include_router(estimate_router, dependencies=[Depends(verify_service_token)])


@app.get("/health")
async def health_check() -> dict:
    """Return service health status."""
    return {"status": "healthy", "service": "servicio_ia", "version": "0.1.0"}
