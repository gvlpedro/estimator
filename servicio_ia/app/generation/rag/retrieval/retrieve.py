"""Shared per-mode retrieval dispatch, reused at different top-k values.

The search router already needs this exact three-way dispatch on
``SearchMode`` (vector / lexical / hybrid) to serve its own top-k request.
Recall-then-rerank (see ``reranked_search.py``) needs the identical dispatch
again, but at a much wider top-k, to gather the candidate pool the
cross-encoder then re-scores. Factoring it out here means both callers share
one implementation of "how do we rank chunks under this mode" instead of the
three-way branch being duplicated in two places and drifting apart.

The "native" score paired with each row differs in meaning and direction per
mode: cosine distance (vector, lower = closer), raw ``ts_rank`` (lexical,
higher = more relevant), or RRF fused score (hybrid, higher = more relevant).
Callers that care about that difference (the router, when not reranking) must
branch on ``mode`` themselves to interpret it correctly. Callers that don't
care (recall-then-rerank, which discards the native score and re-scores with
the cross-encoder instead) can just call this function and ignore the
per-mode scale entirely.

``RECALL_K`` lives here (rather than in ``reranked_search.py``, its original
home) because widening the fetch before truncating is no longer unique to
reranking: deduping by budget (see ``dedupe.py``) needs the same trick for
the same reason — collapsing an already-k-sized fetch down to one chunk per
budget can leave fewer than k results if duplicates dominated that fetch, so
both callers that need "room to discard candidates and still hit k" share
one widened-fetch constant.
"""

from sqlalchemy import Row
from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding_pipeline.schemas import SearchMode
from app.generation.rag.retrieval.dedupe import dedupe_by_budget
from app.generation.rag.retrieval.hybrid_search import hybrid_search
from app.storage import repository

# Matches SearchRequest.k's existing le=50 bound — recall must never
# under-fetch relative to the largest k a client can legally request.
RECALL_K = 50


async def retrieve(
    session: AsyncSession,
    *,
    mode: SearchMode,
    query: str,
    query_vector: list[float] | None,
    k: int,
    dedupe: bool = False,
) -> list[tuple[Row, float]]:
    """Rank chunks under `mode` and return the top k as (row, native_score) pairs.

    `dedupe` (default False) collapses the result to at most one chunk per
    budget before truncating to `k` (see `dedupe_by_budget`). Deduping an
    already-k-sized fetch could yield fewer than k final results if
    duplicates dominated it, so when `dedupe` is True this fetches
    `RECALL_K` candidates instead of `k` from whichever branch `mode`
    selects, giving `dedupe_by_budget` enough room to still pick k distinct
    budgets. When `dedupe` is False, behavior is unchanged: fetch exactly
    `k`, return as-is.
    """
    fetch_k = RECALL_K if dedupe else k
    if mode == "vector":
        rows = await repository.nearest_chunks(session, query_vector, fetch_k)
        result = [(row, row.distance) for row in rows]
    elif mode == "lexical":
        rows = await repository.lexical_chunks(session, query, fetch_k)
        result = [(row, row.rank) for row in rows]
    else:
        result = await hybrid_search(session, query_vector=query_vector, query_text=query, k=fetch_k)

    if dedupe:
        return dedupe_by_budget(result, k)
    return result
