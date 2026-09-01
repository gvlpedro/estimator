"""FastAPI routers exposing the embedding endpoints: encode and search."""

import time

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import Row
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_embedder, get_reranker
from app.embedding_pipeline.embedder import EMBEDDING_MODEL, OpenAIEmbedder
from app.embedding_pipeline.schemas import (
    EncodeRequest,
    EncodeResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.generation.rag.retrieval.cross_encoder import CrossEncoderReranker
from app.generation.rag.retrieval.reranked_search import reranked_search
from app.generation.rag.retrieval.retrieve import retrieve
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


def _to_search_result(row: Row, *, distance: float | None, score: float | None) -> SearchResult:
    """Build one SearchResult from a repository row plus its mode-specific score.

    Row shape differs slightly by branch (`.distance` vs `.rank`), but the
    common columns (id, document_id, chunk_type, content, meta) are identical
    across `nearest_chunks`, `lexical_chunks`, and `hybrid_search` results —
    this helper is the one place that assembles them into the wire schema.
    """
    return SearchResult(
        chunk_id=row.id,
        document_id=row.document_id,
        chunk_type=row.chunk_type,
        content=row.content,
        distance=distance,
        score=score,
        metadata=row.meta,
    )


@search_router.post("/search", response_model=SearchResponse)
async def search_chunks(
    request: SearchRequest,
    session: AsyncSession = Depends(get_session),
    embedder: OpenAIEmbedder = Depends(get_embedder),
    reranker: CrossEncoderReranker = Depends(get_reranker),
) -> SearchResponse:
    """Return the k chunks best matching the query, ranked per `request.mode`.

    - "vector" (default): embed the query and rank by cosine distance —
      unchanged behavior, so existing callers that never set `mode` (the
      business backend's search proxy) keep getting today's response shape.
      The query is embedded with the same model used at ingest time
      (text-embedding-3-small) — mixing embedding models across ingest and
      search makes distances meaningless.
    - "lexical": rank by Postgres full-text `ts_rank`. No embedding call is
      made — that is the point of a lexical branch, it doesn't need one.
    - "hybrid": run both branches and fuse them with Reciprocal Rank Fusion
      (see `app.generation.rag.retrieval.hybrid_search`). Needs the query
      embedded too, since the vector branch runs inside the fusion.

    `request.rerank` (default False) layers recall-then-rerank on top of
    whichever `mode` is selected: retrieve a broad candidate set (see
    `app.generation.rag.retrieval.retrieve.RECALL_K`) via the same
    per-mode ranking, then re-score that shortlist with a cross-encoder and
    keep only the top `k` by that score. This trades one extra ML pass for
    higher precision than the branch's native ranking alone. Lexical mode
    with rerank still needs no query embedding — the cross-encoder scores
    raw (query, chunk text) pairs, not vectors.

    `request.dedupe` (default False) collapses the result to at most one
    chunk per budget, keeping each budget's best-ranked chunk (see
    `app.generation.rag.retrieval.dedupe.dedupe_by_budget`). It composes with
    both `mode` and `rerank`: the underlying fetch widens to `RECALL_K`
    first (whether that's the plain per-mode retrieval or the rerank
    candidate pool) so deduping still leaves room for `k` distinct budgets.

    The per-mode ranking dispatch (used both for the plain top-k here and for
    the wider recall pass under rerank) lives in
    ``app.generation.rag.retrieval.retrieve.retrieve``.
    """
    started = time.perf_counter()
    log.info(
        "search_received",
        k=request.k,
        mode=request.mode,
        rerank=request.rerank,
        dedupe=request.dedupe,
    )

    query_vector: list[float] | None = None
    if request.mode in ("vector", "hybrid"):
        try:
            query_vector = await run_in_threadpool(embedder.embed_one, request.query)
        except Exception:
            log.exception("search_failed", k=request.k, mode=request.mode, rerank=request.rerank)
            raise HTTPException(status_code=500, detail="Query embedding failed") from None

    results: list[SearchResult]
    if request.rerank:
        reranked = await reranked_search(
            session,
            reranker,
            mode=request.mode,
            query=request.query,
            query_vector=query_vector,
            k=request.k,
            dedupe=request.dedupe,
        )
        results = [
            _to_search_result(row, distance=None, score=round(ce_score, 4))
            for row, ce_score in reranked
        ]
    else:
        rows = await retrieve(
            session,
            mode=request.mode,
            query=request.query,
            query_vector=query_vector,
            k=request.k,
            dedupe=request.dedupe,
        )
        if request.mode == "vector":
            results = [
                _to_search_result(row, distance=round(native_score, 4), score=None)
                for row, native_score in rows
            ]
        else:
            results = [
                _to_search_result(row, distance=None, score=round(native_score, 4))
                for row, native_score in rows
            ]

    response = SearchResponse(
        query=request.query,
        k=request.k,
        search_time_ms=round((time.perf_counter() - started) * 1000),
        results=results,
    )
    log.info(
        "search_completed",
        k=request.k,
        mode=request.mode,
        rerank=request.rerank,
        dedupe=request.dedupe,
        results=len(response.results),
        search_time_ms=response.search_time_ms,
    )
    return response
