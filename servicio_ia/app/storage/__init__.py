"""Persistence layer: async engine/session, ORM models and repository.

The vector store schema (documents + chunks with pgvector embeddings) and
every SQL statement live here — routers never build queries themselves.
"""
