"""FastAPI entrypoint for the AI service (servicio_ia)."""

import os
from contextlib import asynccontextmanager

import logfire
import structlog
from fastapi import Depends, FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.embedding_pipeline.router import router as embeddings_router
from app.embedding_pipeline.router import search_router
from app.ingest.router import router as ingest_router
from app.lgraph.build import build_graph
from app.lgraph.checkpointer import get_checkpointer_dsn
from app.lgraph.router import router as estimate_router
from app.security import verify_service_token


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
app.include_router(ingest_router, prefix="/embeddings")
app.include_router(embeddings_router, prefix="/embeddings")
app.include_router(search_router, dependencies=[Depends(verify_service_token)])
app.include_router(estimate_router, dependencies=[Depends(verify_service_token)])


@app.get("/health")
async def health_check() -> dict:
    """Return service health status."""
    return {"status": "healthy", "service": "servicio_ia", "version": "0.1.0"}
