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


class SearchResult(BaseModel):
    """One retrieved chunk with its cosine distance to the query (lower = closer)."""

    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    """Search endpoint output: the echoed query plus the k nearest chunks."""

    query: str
    k: int
    search_time_ms: int
    results: list[SearchResult]
