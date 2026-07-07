"""FastAPI router exposing the embedding pipeline endpoints."""

from functools import lru_cache

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimate_cost_usd
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats

log = structlog.get_logger()

# Prefix ("/embeddings") is applied where the router is registered in main.py.
router = APIRouter(tags=["embeddings"])


@lru_cache
def get_chunker() -> JSONStructuralChunker:
    return JSONStructuralChunker()


@lru_cache
def get_embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder()


@router.post("/ingest", response_model=IngestResponse)
def ingest_budgets(
    request: IngestRequest,
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> IngestResponse:
    """Chunk the budgets, embed every chunk, and return vectors plus run stats.

    - Malformed payloads never reach this handler: FastAPI returns 422 from
      Pydantic validation (including the Budget consistency validators).
    - Any uncontrolled embeddings API error → 500 with a generic message;
      the full detail stays in the logs, not in the client response.
    """
    log.info("embedding_ingest_received", budgets=len(request.budgets))

    chunks = chunker.chunk(request.budgets)
    try:
        embedded = embedder.embed_many(chunks)
    except Exception:
        log.exception("embedding_ingest_failed", budgets=len(request.budgets), chunks=len(chunks))
        raise HTTPException(status_code=500, detail="Embedding generation failed") from None

    total_tokens = sum(chunk.token_count for chunk in embedded)
    stats = IngestStats(
        total_budgets=len(request.budgets),
        total_chunks=len(embedded),
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_cost_usd(total_tokens),
    )
    log.info("embedding_ingest_completed", **stats.model_dump())
    return IngestResponse(chunks=embedded, stats=stats)
