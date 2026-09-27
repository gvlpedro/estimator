"""Reciprocal Rank Fusion: combine several ranked id lists into one ranking.

Hybrid search runs a vector branch (cosine distance) and a lexical branch
(``ts_rank``) independently, and each branch's score lives on an incomparable
scale: cosine distance is a bounded similarity measure, ``ts_rank`` is an
unbounded tf-idf-ish weight shaped by term frequency and document length.
Blending those raw numbers into one score would need an arbitrary
normalization step with no principled scale to normalize against. RRF
sidesteps the problem entirely by throwing the raw scores away and looking
only at *rank position* within each branch — a value both branches produce
in the same units (1st place, 2nd place, ...) regardless of how that branch
computed its ranking internally.
"""

RRF_K = 60  # from the original RRF paper (Cormack, Clarke & Buettcher, 2009); de facto default across hybrid-search implementations


def reciprocal_rank_fusion(rankings: list[list[int]], k: float = RRF_K) -> list[tuple[int, float]]:
    """Fuse several best-to-worst id rankings into one score-sorted ranking.

    ``score(id) = sum over rankings containing id of 1 / (k + rank)``, rank
    counted from 1. An id absent from a branch simply contributes nothing
    from that branch — no penalty, no zero-fill — which is why RRF composes
    cleanly across branches with very different result-set sizes or corpus
    coverage: a chunk only the lexical branch found is not punished for the
    vector branch's silence on it.

    ``k`` is a module-level constant rather than a parameter callers are
    expected to override per call: tuning it is a corpus-level calibration
    exercise (how much weight low-ranked results get relative to top-ranked
    ones across the whole retrieval system), not a per-request client
    choice, so it is deliberately not exposed over HTTP.

    ``sorted`` is stable, so ids that tie on fused score keep the relative
    order they first appeared in — no extra tie-break logic needed.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + position)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
