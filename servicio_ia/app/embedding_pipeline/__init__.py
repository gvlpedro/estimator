"""Embedding pipeline: text in, vectors out.

Owns the OpenAI embedder, the chunk/vector contracts and the HTTP endpoints
``POST /embeddings/encode`` (raw texts → vectors) and ``POST /search``
(semantic search over the persisted corpus).
"""
