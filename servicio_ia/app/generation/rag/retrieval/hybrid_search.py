"""Hybrid search: fuse vector and lexical retrieval into one ranking.

Vector search catches *meaning* — a query and a chunk can rank close even
when they share no words, because the embedding captures what the text is
about. It also blurs exact terms: identifiers, acronyms, and tech-stack
names ("Redsys", "Bizum", a specific API name) get smoothed into the same
neighborhood as related-but-different terms, so a chunk that mentions the
exact term a user searched for can rank behind one that is merely topically
similar. Lexical (full-text) search is the mirror image: it catches exact
term matches precisely but has no notion of meaning beyond shared lexemes.
Hybrid search runs both branches and fuses their rankings so results that
either branch alone would rank low — but the other branch is confident
about — still surface near the top.
"""

from sqlalchemy import Row
from sqlalchemy.ext.asyncio import AsyncSession

from app.generation.rag.retrieval.rrf import RRF_K, reciprocal_rank_fusion
from app.storage import repository


async def hybrid_search(
    session: AsyncSession, *, query_vector: list[float], query_text: str, k: int, rrf_k: float = RRF_K
) -> list[tuple[Row, float]]:
    """Run vector and lexical search independently, then fuse by RRF.

    Each branch gets its own independent top-k *before* fusion — the two
    result sets are not intersected first. A chunk one branch ranks highly
    and the other misses entirely is exactly the case hybrid search exists
    to surface, so fusing only the overlap would silently discard it.
    """
    vector_rows = await repository.nearest_chunks(session, query_vector, k)
    lexical_rows = await repository.lexical_chunks(session, query_text, k)

    # First-write-wins on id collision is fine: content and metadata are
    # identical for the same chunk_id regardless of which branch returned it.
    rows_by_id: dict[int, Row] = {}
    for row in (*vector_rows, *lexical_rows):
        rows_by_id.setdefault(row.id, row)

    fused = reciprocal_rank_fusion(
        [
            [row.id for row in vector_rows],
            [row.id for row in lexical_rows],
        ],
        k=rrf_k,
    )

    # Full row data (content, metadata, ...) paired with the fused score —
    # not the branch-local `.distance`/`.rank`, which have different names
    # and scales across the two Row shapes and are not what downstream code
    # should consume once fusion has happened.
    return [(rows_by_id[chunk_id], score) for chunk_id, score in fused[:k]]
