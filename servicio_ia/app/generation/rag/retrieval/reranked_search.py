"""Recall-then-rerank: broad cheap retrieval, then precise expensive scoring.

Two-stage retrieval trades one expensive pass for a cheap first pass that
optimizes for coverage over precision, followed by a second pass that is
precise but too costly to run over the whole corpus. Stage one is whichever
``retrieve()`` branch `mode` selects (vector cosine distance, lexical
``ts_rank``, or hybrid RRF) widened to ``RECALL_K`` candidates instead of the
caller's requested k — the goal here is not to rank well, just to not miss
relevant chunks. Stage two hands that shortlist to the cross-encoder (see
``cross_encoder.py`` for why cross-encoder scoring is accurate but does not
scale past a shortlist) and keeps only the caller's requested top k of the
re-scored results.
"""

from sqlalchemy import Row
from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding_pipeline.schemas import SearchMode
from app.generation.rag.retrieval.cross_encoder import CrossEncoderReranker
from app.generation.rag.retrieval.dedupe import dedupe_by_budget
from app.generation.rag.retrieval.retrieve import RECALL_K, retrieve


async def reranked_search(
    session: AsyncSession,
    reranker: CrossEncoderReranker,
    *,
    mode: SearchMode,
    query: str,
    query_vector: list[float] | None,
    k: int,
    dedupe: bool = False,
) -> list[tuple[Row, float]]:
    """Retrieve RECALL_K candidates under `mode`, rerank, return the top k.

    `dedupe` (default False) is applied AFTER reranking, not before: the
    candidate fetch below always stays a plain (non-deduped) call, so the
    cross-encoder sees the full raw candidate pool including near-duplicate
    chunks from the same budget — deduping before reranking would throw away
    information the reranker could otherwise use to pick the single
    best-matching chunk per budget. When `dedupe` is True, `dedupe_by_budget`
    runs on the reranker's full RECALL_K-sized re-scored output (not
    pre-truncated to k), so it has enough distinct budgets to still return k
    results. When `dedupe` is False, behavior is unchanged.
    """
    candidates = await retrieve(
        session, mode=mode, query=query, query_vector=query_vector, k=RECALL_K
    )
    # The branch-native score (distance / ts_rank / RRF) is discarded here —
    # the cross-encoder scores the raw (query, content) pairing fresh, and
    # that score is what determines the final order, not whatever got the
    # chunk into the recall set.
    rows = [row for row, _native_score in candidates]
    ranked = reranker.rerank(query, [row.content for row in rows])
    scored = [(rows[index], score) for index, score in ranked]
    if dedupe:
        return dedupe_by_budget(scored, k)
    return scored[:k]
