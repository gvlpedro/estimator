"""Cross-encoder reranking of a vector-search shortlist.

A cross-encoder scores a (query, document) pair jointly — both texts pass
through the model together, so it can attend to how they relate to each
other — which makes it far more accurate than comparing two independently
computed embeddings. That accuracy is expensive: each pair needs its own
forward pass, so it does not scale to scoring an entire corpus. The
``embedding_pipeline`` bi-encoder already narrows the corpus down to a small
shortlist by cosine similarity; this module re-scores only that shortlist,
where the cross-encoder's cost is affordable and its accuracy improves the
final ranking.
"""

import time

import structlog
from sentence_transformers import CrossEncoder

log = structlog.get_logger()

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Scores and reorders a shortlist of documents against a query."""

    def __init__(self, model: CrossEncoder | None = None) -> None:
        # Loading the real model downloads weights from HuggingFace and needs
        # no API key; injecting one keeps the class testable without network access.
        self._model = model or CrossEncoder(RERANKER_MODEL)

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Score each document against the query, one score per input, same order."""
        if not documents:
            return []
        started = time.perf_counter()
        pairs = [[query, document] for document in documents]
        scores = [float(score) for score in self._model.predict(pairs)]
        latency_ms = (time.perf_counter() - started) * 1000
        log.info("rerank_scored", documents=len(documents), latency_ms=round(latency_ms, 1))
        return scores

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        """(original_index, score) pairs, sorted by score descending.

        ``sorted`` is stable, so documents that tie on score keep their
        original relative order — no extra tie-break logic needed.
        """
        scores = self.score(query, documents)
        return sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
