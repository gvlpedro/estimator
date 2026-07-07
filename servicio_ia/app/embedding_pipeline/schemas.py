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
    """Payload accepted by the ingest endpoint: budgets to chunk and embed."""

    budgets: list[Budget] = Field(min_length=1)


class IngestStats(BaseModel):
    """Aggregate metrics of one ingest run."""

    total_budgets: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)


class IngestResponse(BaseModel):
    """Ingest endpoint output: embedded chunks plus run statistics."""

    chunks: list[EmbeddedChunk]
    stats: IngestStats
