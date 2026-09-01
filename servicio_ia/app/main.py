"""FastAPI entrypoint for the AI service (servicio_ia)."""

import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.embedding_pipeline.router import router as embeddings_router
from app.embedding_pipeline.router import search_router
from app.ingest.router import router as ingest_router


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
    """Application startup and shutdown lifecycle."""
    configure_logging()
    structlog.get_logger().info("application_started", service="servicio_ia")
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

app.include_router(ingest_router, prefix="/embeddings")
app.include_router(embeddings_router, prefix="/embeddings")
app.include_router(search_router)


@app.get("/health")
async def health_check() -> dict:
    """Return service health status."""
    return {"status": "healthy", "service": "servicio_ia", "version": "0.1.0"}
