"""Ingestion module: normalized budgets in, persisted embedded chunks out.

Owns the document contracts (Budget), the chunking strategy and the
``POST /embeddings/ingest`` endpoint that orchestrates chunk → embed → persist.
"""
