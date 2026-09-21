"""Repository: every SQL statement of the AI service lives in this module.

Routers depend on these functions instead of building queries inline, so the
storage concerns (duplicate detection, transactional persistence, vector
ranking) stay in one place.
"""

from collections.abc import Sequence
from typing import Any

from sqlalchemy import Row, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage import models


async def find_document_id(session: AsyncSession, source_path: str) -> int | None:
    """Return the id of the document already ingested from source_path, if any."""
    return await session.scalar(
        select(models.Document.id).where(models.Document.source_path == source_path)
    )


async def add_document_with_chunks(
    session: AsyncSession,
    *,
    source_path: str,
    document_type: str,
    document_meta: dict[str, Any],
    chunk_type: str,
    chunks: Sequence[tuple[str, list[float], dict[str, Any]]],
) -> models.Document:
    """Stage one document plus its chunks; the caller owns the commit.

    Each chunk is (content, embedding, metadata). Flush sends the document
    INSERT and assigns its id without committing: the row stays invisible to
    other transactions until the caller's final commit, so an embedding or
    chunk failure rolls the document back too — no orphan documents.
    """
    document = models.Document(
        source_path=source_path,
        document_type=document_type,
        meta=document_meta,
    )
    session.add(document)
    await session.flush()

    session.add_all(
        models.Chunk(
            document_id=document.id,
            chunk_type=chunk_type,
            content=content,
            embedding=embedding,
            meta=meta,
        )
        for content, embedding, meta in chunks
    )
    return document


async def nearest_chunks(
    session: AsyncSession, embedding: list[float], k: int
) -> Sequence[Row]:
    """Return the k chunks nearest to the embedding by cosine distance.

    cosine_distance (operator <=>) over inner_product: OpenAI embeddings are
    normalized so both rank identically, but the future HNSW index will use
    vector_cosine_ops — if the query operator and the index operator class
    don't match, Postgres silently falls back to a sequential scan.
    """
    distance = models.Chunk.embedding.cosine_distance(embedding)
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
        .limit(k)
    )
    return (await session.execute(stmt)).all()


async def lexical_chunks(session: AsyncSession, query: str, k: int) -> Sequence[Row]:
    """Return the k chunks that best match query via Postgres full-text search.

    websearch_to_tsquery over plainto_tsquery/to_tsquery: it parses the same
    free-text ``query`` string the vector branch already accepts (quotes for
    phrases, "-" to exclude terms, implicit AND between words) instead of
    requiring callers to learn tsquery's own operator syntax — hybrid search
    calls both branches with the same raw string.

    ts_rank weighs a match by lexeme frequency and position, which is why it
    lives on a different, unbounded scale than cosine distance; the two are
    never compared directly, only fused by rank position (see
    generation/rag/retrieval/rrf.py).

    A query that is empty after stopword removal (or has no lexemes at all)
    produces a tsquery that matches nothing: the ``@@`` predicate below
    naturally filters those rows out, so there is no special case needed for
    "no meaningful search terms" — Postgres already returns an empty result.
    """
    tsquery = func.websearch_to_tsquery(models.FULLTEXT_SEARCH_CONFIG, query)
    rank = func.ts_rank(models.Chunk.content_tsv, tsquery).label("rank")
    stmt = (
        select(
            models.Chunk.id,
            models.Chunk.document_id,
            models.Chunk.chunk_type,
            models.Chunk.content,
            models.Chunk.meta,
            rank,
        )
        .where(models.Chunk.content_tsv.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(k)
    )
    return (await session.execute(stmt)).all()
