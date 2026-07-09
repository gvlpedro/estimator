"""FastAPI router exposing the embedding pipeline endpoints."""

import time
from functools import lru_cache

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import models
from app.db import get_session
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import (
    IngestConflict,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

log = structlog.get_logger()

# Prefix ("/embeddings") is applied where the router is registered in main.py.
router = APIRouter(tags=["embeddings"])

# Registered WITHOUT prefix: the contract exposes POST /search at the root.
search_router = APIRouter(tags=["search"])

CHUNK_TYPE_BUDGET_COMPONENT = "budget_component"


@lru_cache
def get_chunker() -> JSONStructuralChunker:
    return JSONStructuralChunker()


@lru_cache
def get_embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={409: {"model": IngestConflict, "description": "Document already ingested"}},
)
async def ingest_document(
    request: IngestRequest,
    session: AsyncSession = Depends(get_session),
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder = Depends(get_embedder),
):
    """Chunk one budget, embed the chunks and persist document + chunks.

    Everything runs inside a single transaction: if the embedder fails, the
    document row is rolled back too — no orphan documents without chunks.

    - Malformed payloads never reach this handler: FastAPI returns 422 from
      Pydantic validation (including the Budget consistency validators).
    - Any uncontrolled embeddings API error → 500 with a generic message;
      the full detail stays in the logs, not in the client response.
    """
    started = time.perf_counter()
    log.info("embedding_ingest_received", source_path=request.source_path)

    existing_id = await session.scalar(
        select(models.Document.id).where(models.Document.source_path == request.source_path)
    )
    if existing_id is not None:
        log.info("embedding_ingest_duplicate", source_path=request.source_path,
                 document_id=existing_id)
        # Plain JSONResponse instead of HTTPException(detail=...): the contract
        # puts document_id at the top level, not nested under "detail".
        return JSONResponse(
            status_code=409,
            content={"detail": "Document already ingested", "document_id": existing_id},
        )

    budget = request.content
    document = models.Document(
        source_path=request.source_path,
        document_type=request.document_type,
        meta={
            "budget_id": budget.budget_id,
            "client_sector": budget.client_metadata.sector,
            "main_technology": budget.main_technology,
            "year": budget.year,
            "total_estimated_hours": budget.total_estimated_hours,
        },
    )
    session.add(document)
    # Flush sends the INSERT and assigns document.id without committing:
    # the row stays invisible to other transactions until the final commit.
    await session.flush()

    chunks = chunker.chunk([budget])
    try:
        # The OpenAI client is synchronous; a thread keeps the event loop
        # free while the (batched) embeddings call is in flight.
        embedded = await run_in_threadpool(embedder.embed_many, chunks)
    except Exception:
        log.exception("embedding_ingest_failed", source_path=request.source_path,
                      chunks=len(chunks))
        raise HTTPException(status_code=500, detail="Embedding generation failed") from None

    session.add_all(
        models.Chunk(
            document_id=document.id,
            chunk_type=CHUNK_TYPE_BUDGET_COMPONENT,
            content=chunk.text,
            embedding=chunk.embedding,
            meta={**chunk.metadata, "chunk_id": chunk.chunk_id,
                  "token_count": chunk.token_count},
        )
        for chunk in embedded
    )
    await session.commit()

    response = IngestResponse(
        document_id=document.id,
        chunks_created=len(embedded),
        embedding_dimension=len(embedded[0].embedding),
        ingestion_time_ms=round((time.perf_counter() - started) * 1000),
    )
    log.info("embedding_ingest_completed", **response.model_dump())
    return response


@search_router.post("/search", response_model=SearchResponse)
async def search_chunks(
    request: SearchRequest,
    session: AsyncSession = Depends(get_session),
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> SearchResponse:
    """Return the k chunks nearest to the query by cosine distance.

    The query is embedded with the same model used at ingest time
    (text-embedding-3-small) — mixing embedding models across ingest and
    search makes distances meaningless.

    cosine_distance (operator <=>) over inner_product: OpenAI embeddings are
    normalized so both rank identically, but the future HNSW index will use
    vector_cosine_ops — if the query operator and the index operator class
    don't match, Postgres silently falls back to a sequential scan.
    """
    started = time.perf_counter()
    log.info("search_received", k=request.k)

    try:
        query_vector = await run_in_threadpool(embedder.embed_one, request.query)
    except Exception:
        log.exception("search_failed", k=request.k)
        raise HTTPException(status_code=500, detail="Query embedding failed") from None

    distance = models.Chunk.embedding.cosine_distance(query_vector)
    stmt = (
        select(
            models.Chunk.id,
            models.Chunk.document_id,
            models.Chunk.chunk_type,
            models.Chunk.content,
            models.Chunk.meta,
            distance.label("distance"),
        )
        # The schema allows embedding to be NULL (async ingest, future
        # sessions); a NULL distance would be unrankable garbage in the top-k.
        .where(models.Chunk.embedding.is_not(None))
        .order_by(distance)
        .limit(request.k)
    )
    rows = (await session.execute(stmt)).all()

    response = SearchResponse(
        query=request.query,
        k=request.k,
        search_time_ms=round((time.perf_counter() - started) * 1000),
        results=[
            SearchResult(
                chunk_id=row.id,
                document_id=row.document_id,
                chunk_type=row.chunk_type,
                content=row.content,
                distance=round(row.distance, 4),
                metadata=row.meta,
            )
            for row in rows
        ],
    )
    log.info(
        "search_completed",
        k=request.k,
        results=len(response.results),
        search_time_ms=response.search_time_ms,
    )
    return response
