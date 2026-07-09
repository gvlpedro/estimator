"""Pydantic models for the embedding pipeline.

Defines the data contracts shared across the pipeline: normalized historical
budgets (as found in ``data/budgets_sample.json``), chunks ready for
embedding, and the request/response payloads of the ingest endpoint.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Closed vocabularies produced by the normalization pipeline. A value outside
# these sets means the input was NOT normalized — reject it, don't coerce it.
Sector = Literal["finance", "ecommerce", "healthcare", "industrial"]
Complexity = Literal["low", "medium", "high"]


class ClientMetadata(BaseModel):
    """Client identification attached to a historical budget."""

    name: str = Field(min_length=1, max_length=120)
    sector: Sector
    country: str = Field(pattern=r"^[A-Z]{2}$", description="Two-letter country code.")


class BudgetComponent(BaseModel):
    """One work component inside a historical budget."""

    component_id: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=10, max_length=1000)
    tech_stack: list[str] = Field(min_length=1)
    estimated_hours: int = Field(ge=1, le=10_000)
    complexity: Complexity
    dependencies: list[str] = Field(default_factory=list)


class Budget(BaseModel):
    """A complete normalized historical budget."""

    budget_id: str = Field(min_length=1, max_length=32)
    client_metadata: ClientMetadata
    project_summary: str = Field(min_length=10, max_length=600)
    main_technology: str = Field(min_length=1, max_length=64)
    year: int = Field(ge=2000, le=2100)
    total_estimated_hours: int = Field(ge=1, le=100_000)
    components: list[BudgetComponent] = Field(min_length=1)

    @model_validator(mode="after")
    def totals_and_dependencies_are_consistent(self) -> "Budget":
        # Normalized data guarantees both invariants; a violation here means
        # the source file is corrupt and must be fixed upstream, not silently
        # reconciled — unlike LLM output, historical data is ground truth.
        component_sum = sum(component.estimated_hours for component in self.components)
        if component_sum != self.total_estimated_hours:
            raise ValueError(
                f"total_estimated_hours ({self.total_estimated_hours}) does not match "
                f"the sum of component hours ({component_sum}) in {self.budget_id}"
            )
        known_ids = {component.component_id for component in self.components}
        for component in self.components:
            unknown = set(component.dependencies) - known_ids
            if unknown:
                raise ValueError(
                    f"component {component.component_id} in {self.budget_id} depends on "
                    f"unknown component ids: {sorted(unknown)}"
                )
        return self


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


class IngestRequest(BaseModel):
    """Payload accepted by the ingest endpoint: one document to persist.

    ``content`` is the full normalized budget JSON; validating it as a
    ``Budget`` keeps the 422-on-malformed-input behavior of the old contract.
    """

    source_path: str = Field(min_length=1, max_length=500)
    document_type: str = Field(min_length=1, max_length=50)
    content: Budget


class IngestResponse(BaseModel):
    """Ingest endpoint output: identifiers and metrics, never the vectors."""

    document_id: int
    chunks_created: int = Field(ge=1)
    embedding_dimension: int = Field(ge=1)
    ingestion_time_ms: int = Field(ge=0)


class IngestConflict(BaseModel):
    """409 body when the source_path was already ingested."""

    detail: str
    document_id: int


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
