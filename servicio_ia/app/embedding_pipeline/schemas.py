"""Pydantic contracts of the embedding pipeline.

Chunks ready for embedding, embedded chunks, and the request/response
payloads of the encode and search endpoints.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints

# Closed vocabularies produced by the normalization pipeline. A value outside
# these sets means the input was NOT normalized — reject it, don't coerce it.
Sector = Literal["finance", "ecommerce", "healthcare", "industrial"]
Complexity = Literal["low", "medium", "high"]

# "vector": cosine similarity only. "lexical": Postgres full-text (ts_rank)
# only. "hybrid": both branches fused by Reciprocal Rank Fusion. Default is
# "vector" so existing callers (the business backend's search proxy, which
# never sets `mode`) keep getting exactly today's response shape.
SearchMode = Literal["vector", "lexical", "hybrid"]


class Chunk(BaseModel):
    """A text fragment ready to be embedded.

    ``metadata`` carries the filterable fields (sector, technology, year, …)
    so retrieval can pre-filter before computing vector similarity.
    """

    chunk_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_count: int = Field(ge=1)


class EmbeddedChunk(Chunk):
    """A chunk plus the embedding vector returned by the embeddings API."""

    embedding: list[float] = Field(min_length=1)


class EncodeRequest(BaseModel):
    """Payload accepted by the encode endpoint: raw texts to embed.

    Capped at the embedder's batch size: one request maps to one embeddings
    API call, so the endpoint's latency stays predictable.
    """

    # Per-item bounds: the embeddings API rejects empty strings, and 8192
    # tokens is the model's context limit — 32k chars is a generous proxy
    # that fails fast on pathological payloads without counting tokens here.
    texts: list[Annotated[str, StringConstraints(min_length=1, max_length=32_000)]] = Field(
        min_length=1, max_length=100
    )


class EncodeResponse(BaseModel):
    """Encode endpoint output: one vector per input text, in input order."""

    model: str
    embedding_dimension: int = Field(ge=1)
    embeddings: list[list[float]]
    encode_time_ms: int = Field(ge=0)


class SearchRequest(BaseModel):
    """Payload accepted by the search endpoint: free-text query + result count."""

    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)
    # Default preserves current behavior for every existing caller: the
    # business backend's search proxy never sets this field.
    mode: SearchMode = "vector"
    # Recall-then-rerank toggle: when True, retrieval widens to a broad
    # candidate set and a cross-encoder re-scores it to pick the final k (see
    # app.generation.rag.retrieval.reranked_search). Default False preserves
    # current behavior for every existing caller, same reasoning as `mode`.
    rerank: bool = False
    # Per-budget dedupe toggle: when True, the result is collapsed to at most
    # one chunk per budget_id, keeping each budget's best-ranked chunk (see
    # app.generation.rag.retrieval.dedupe.dedupe_by_budget). Composes with
    # both `mode` and `rerank` — when True it widens the underlying fetch
    # first so deduping still leaves room for k distinct budgets. Default
    # False preserves current behavior for every existing caller, same
    # reasoning as `mode` and `rerank`.
    dedupe: bool = False


class SearchResult(BaseModel):
    """One retrieved chunk, scored by whichever branch(es) `mode`/`rerank` selected.

    The two score fields point in OPPOSITE directions — read `mode` and
    `rerank` before reading the number:
    - `distance`: cosine distance from the query embedding, lower = closer.
      Populated only when `mode == "vector"` and `rerank is False`; `None`
      otherwise.
    - `score`: `None` only when `mode == "vector"` and `rerank is False`.
      Otherwise: in "lexical" mode (rerank off) it is the raw `ts_rank`
      weight; in "hybrid" mode (rerank off) it is the RRF fused score;
      whenever `rerank is True` (regardless of `mode`) it is the
      cross-encoder's relevance score, which is what actually determined the
      final order — it supersedes the branch-native score, the two are never
      shown together in one response row. Every case: higher = more relevant.
    """

    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float | None = None
    score: float | None = None
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    """Search endpoint output: the echoed query plus the k nearest chunks."""

    query: str
    k: int
    search_time_ms: int
    results: list[SearchResult]
