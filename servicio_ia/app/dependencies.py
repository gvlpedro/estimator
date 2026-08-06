"""Shared FastAPI dependency providers.

Both routers (ingest and embeddings/search) need the same embedder instance;
lru_cache makes these process-wide singletons without module-level side
effects at import time (the OpenAI client reads its key lazily).
"""

from functools import lru_cache

from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.generation.rag.retrieval.cross_encoder import CrossEncoderReranker
from app.ingest.chunker import JSONStructuralChunker


@lru_cache
def get_chunker() -> JSONStructuralChunker:
    return JSONStructuralChunker()


@lru_cache
def get_embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder()


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
