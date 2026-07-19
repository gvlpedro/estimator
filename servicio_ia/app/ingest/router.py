"""FastAPI router exposing the ingest endpoint."""

import time

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_chunker, get_embedder
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.ingest.chunker import JSONStructuralChunker
from app.ingest.schemas import IngestConflict, IngestRequest, IngestResponse
from app.storage import repository
from app.storage.db import get_session

log = structlog.get_logger()

# Prefix ("/embeddings") is applied where the router is registered in main.py.
router = APIRouter(tags=["ingest"])

CHUNK_TYPE_BUDGET_COMPONENT = "budget_component"


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

    Embedding happens BEFORE any row is staged: an embedder failure leaves
    the database untouched, and document + chunks land in a single
    transaction — no orphan documents without chunks.

    - Malformed payloads never reach this handler: FastAPI returns 422 from
      Pydantic validation (including the Budget consistency validators).
    - Any uncontrolled embeddings API error → 500 with a generic message;
      the full detail stays in the logs, not in the client response.
    """
    started = time.perf_counter()
    log.info("embedding_ingest_received", source_path=request.source_path)

    existing_id = await repository.find_document_id(session, request.source_path)
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
    chunks = chunker.chunk([budget])
    try:
        # The OpenAI client is synchronous; a thread keeps the event loop
        # free while the (batched) embeddings call is in flight.
        embedded = await run_in_threadpool(embedder.embed_many, chunks)
    except Exception:
        log.exception("embedding_ingest_failed", source_path=request.source_path,
                      chunks=len(chunks))
        raise HTTPException(status_code=500, detail="Embedding generation failed") from None

    document = await repository.add_document_with_chunks(
        session,
        source_path=request.source_path,
        document_type=request.document_type,
        document_meta={
            "budget_id": budget.budget_id,
            "client_sector": budget.client_metadata.sector,
            "main_technology": budget.main_technology,
            "year": budget.year,
            "total_estimated_hours": budget.total_estimated_hours,
        },
        chunk_type=CHUNK_TYPE_BUDGET_COMPONENT,
        chunks=[
            (
                chunk.text,
                chunk.embedding,
                {**chunk.metadata, "chunk_id": chunk.chunk_id,
                 "token_count": chunk.token_count},
            )
            for chunk in embedded
        ],
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
