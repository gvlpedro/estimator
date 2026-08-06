"""Unit tests for CrossEncoderReranker: pairing, ordering and edge cases.

A stub model (``.predict(pairs) -> list[float]``) is injected via the
constructor so these tests exercise the wrapper's own logic without
downloading the real HuggingFace model or touching the network.
"""

from app.generation.rag.retrieval.cross_encoder import CrossEncoderReranker


class StubCrossEncoder:
    """Fake CrossEncoder: returns a fixed score list, records what it was called with."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.predict_calls: list[list[list[str]]] = []

    def predict(self, pairs: list[list[str]]) -> list[float]:
        self.predict_calls.append(pairs)
        return self._scores


def test_score_returns_scores_in_input_order():
    stub = StubCrossEncoder(scores=[0.2, 0.9, 0.5])
    reranker = CrossEncoderReranker(model=stub)

    scores = reranker.score("query", ["doc a", "doc b", "doc c"])

    assert scores == [0.2, 0.9, 0.5]
    assert stub.predict_calls == [[["query", "doc a"], ["query", "doc b"], ["query", "doc c"]]]


def test_score_empty_documents_returns_empty_without_calling_model():
    stub = StubCrossEncoder(scores=[])
    reranker = CrossEncoderReranker(model=stub)

    scores = reranker.score("query", [])

    assert scores == []
    assert stub.predict_calls == []


def test_rerank_sorts_by_score_descending_with_correct_original_indices():
    stub = StubCrossEncoder(scores=[0.1, 0.9, 0.5])
    reranker = CrossEncoderReranker(model=stub)

    ranked = reranker.rerank("query", ["doc a", "doc b", "doc c"])

    assert ranked == [(1, 0.9), (2, 0.5), (0, 0.1)]
