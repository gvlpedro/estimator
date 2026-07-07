"""Embedding generation via the OpenAI embeddings API.

Batched calls to ``text-embedding-3-small`` at its default dimension (1536).
No numpy or scikit-learn: vector math (cosine similarity) is implemented by
hand with the stdlib in ``scripts/compare.py``.
"""

import time

import structlog
from openai import OpenAI, RateLimitError
from openai.types import CreateEmbeddingResponse

from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

log = structlog.get_logger()

EMBEDDING_MODEL = "text-embedding-3-small"

# OpenAI list price for text-embedding-3-small input tokens as of 2026-07.
# Prices change over time — re-check before trusting cost estimates.
PRICE_USD_PER_MILLION_INPUT_TOKENS = 0.02

# The embeddings endpoint accepts lists: batching avoids hammering the API
# with one serialized request per chunk.
DEFAULT_BATCH_SIZE = 100

MAX_RETRIES = 3


def estimate_cost_usd(total_tokens: int) -> float:
    """Estimated input cost in USD for the given token count."""
    return total_tokens / 1_000_000 * PRICE_USD_PER_MILLION_INPUT_TOKENS


class OpenAIEmbedder:
    """Embeds chunks with ``text-embedding-3-small`` in batched API calls."""

    def __init__(self, client: OpenAI | None = None, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        # The default client reads OPENAI_API_KEY from the environment;
        # injecting a client keeps the class testable without network access.
        self._client = client or OpenAI()
        self._batch_size = batch_size

    def embed_one(self, text: str) -> list[float]:
        response = self._create_with_retry([text])
        return response.data[0].embedding

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        embedded: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            started = time.perf_counter()
            response = self._create_with_retry([chunk.text for chunk in batch])
            latency_ms = (time.perf_counter() - started) * 1000

            # The API preserves input order, but each item carries its index —
            # sorting makes the chunk/vector pairing explicit instead of assumed.
            vectors = sorted(response.data, key=lambda item: item.index)
            embedded.extend(
                EmbeddedChunk(**chunk.model_dump(), embedding=item.embedding)
                for chunk, item in zip(batch, vectors, strict=True)
            )
            log.info(
                "embedding_batch_processed",
                batch_index=start // self._batch_size,
                chunks=len(batch),
                total_tokens=response.usage.total_tokens,
                latency_ms=round(latency_ms, 1),
            )
        return embedded

    def _create_with_retry(self, texts: list[str]) -> CreateEmbeddingResponse:
        # Only rate limiting is retried (1s, 2s, 4s); any other error means
        # something is actually wrong and must propagate to the caller.
        for attempt in range(MAX_RETRIES + 1):
            try:
                return self._client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            except RateLimitError:
                if attempt == MAX_RETRIES:
                    raise
                wait_seconds = 2**attempt
                log.warning(
                    "embedding_rate_limited",
                    attempt=attempt + 1,
                    max_retries=MAX_RETRIES,
                    wait_seconds=wait_seconds,
                )
                time.sleep(wait_seconds)
        raise AssertionError("unreachable")  # pragma: no cover
