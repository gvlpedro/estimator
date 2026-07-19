"""FastAPI routers exposing the embedding endpoints: encode and search."""

import time

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_embedder
from app.embedding_pipeline.embedder import EMBEDDING_MODEL, OpenAIEmbedder
from app.embedding_pipeline.schemas import (
    EncodeRequest,
    EncodeResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.storage import repository
from app.storage.db import get_session

log = structlog.get_logger()

# Prefix ("/embeddings") is applied where the router is registered in main.py.
router = APIRouter(tags=["embeddings"])

# Registered WITHOUT prefix: the contract exposes POST /search at the root.
search_router = APIRouter(tags=["search"])


@router.post("/encode", response_model=EncodeResponse)
async def encode_texts(
    request: EncodeRequest,
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> EncodeResponse:
    """Embed raw texts and return the vectors, one per input in order.

    Unlike /embeddings/ingest, nothing is persisted: this endpoint exposes
    the encoder itself, so other services (and the live session exercises)
    can obtain vectors without going through the document pipeline.
    """
    started = time.perf_counter()
    log.info("encode_received", texts=len(request.texts))

    try:
        # The OpenAI client is synchronous; a thread keeps the event loop
        # free while the (batched) embeddings call is in flight.
        vectors = await run_in_threadpool(embedder.embed_texts, request.texts)
    except Exception:
        log.exception("encode_failed", texts=len(request.texts))
        raise HTTPException(status_code=500, detail="Embedding generation failed") from None

    response = EncodeResponse(
        model=EMBEDDING_MODEL,
        embedding_dimension=len(vectors[0]),
        embeddings=vectors,
        encode_time_ms=round((time.perf_counter() - started) * 1000),
    )
    log.info(
        "encode_completed",
        texts=len(request.texts),
        embedding_dimension=response.embedding_dimension,
        encode_time_ms=response.encode_time_ms,
    )
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
    search makes distances meaningless. The vector ranking itself lives in
    ``app.storage.repository.nearest_chunks``.
    """
    started = time.perf_counter()
    log.info("search_received", k=request.k)

    try:
        query_vector = await run_in_threadpool(embedder.embed_one, request.query)
    except Exception:
        log.exception("search_failed", k=request.k)
        raise HTTPException(status_code=500, detail="Query embedding failed") from None

    rows = await repository.nearest_chunks(session, query_vector, request.k)

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
