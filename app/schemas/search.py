"""Schemas for the semantic-search proxy endpoint.

They mirror the AI service's /search contract
(servicio_ia/app/embedding_pipeline/schemas.py) so the backend re-validates
what it forwards instead of blindly proxying JSON.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search query forwarded to the AI service."""

    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    """One historical budget chunk returned by the AI service."""

    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float
    metadata: dict


class SearchResponse(BaseModel):
    """Ranked chunks for a query, as returned by the AI service."""

    query: str
    k: int
    search_time_ms: int
    results: list[SearchResult]
