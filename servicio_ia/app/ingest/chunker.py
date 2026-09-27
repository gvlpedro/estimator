"""Chunking strategy for historical budgets.

Structural chunking: the JSON already carries the semantic boundaries, so one
budget component becomes exactly one chunk. No overlap, no fixed-size
splitting — an exceptionally long description is a data-quality signal to
surface, not something to paper over with a splitter.
"""

import tiktoken

from app.embedding_pipeline.schemas import Chunk
from app.ingest.schemas import Budget

EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_TEMPLATE = """\
[Project: {project_summary}]
[Client sector: {sector} | Year: {year} | Main tech: {main_technology}]

Component: {name}
Description: {description}
Tech stack: {tech_stack}
Complexity: {complexity}
Estimated hours: {estimated_hours}"""


class JSONStructuralChunker:
    """Turns normalized budgets into one chunk per component.

    Each chunk text prepends the parent budget context (contextual chunk
    header): without it, a component like "Authentication backend" embeds
    identically for a fintech and for an e-commerce client, and retrieval
    loses the sector/stack signal that makes historical budgets comparable.
    """

    def __init__(self) -> None:
        self._encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for budget in budgets:
            for component in budget.components:
                text = CHUNK_TEMPLATE.format(
                    project_summary=budget.project_summary,
                    sector=budget.client_metadata.sector,
                    year=budget.year,
                    main_technology=budget.main_technology,
                    name=component.name,
                    description=component.description,
                    tech_stack=", ".join(component.tech_stack),
                    complexity=component.complexity,
                    estimated_hours=component.estimated_hours,
                )
                chunks.append(
                    Chunk(
                        chunk_id=f"{budget.budget_id}::{component.component_id}",
                        text=text,
                        # Filterable fields that travel with the chunk but are
                        # NOT embedded: retrieval can pre-filter on them before
                        # computing vector similarity.
                        metadata={
                            "budget_id": budget.budget_id,
                            "component_id": component.component_id,
                            "client_sector": budget.client_metadata.sector,
                            "main_technology": budget.main_technology,
                            "year": budget.year,
                            "complexity": component.complexity,
                            "estimated_hours": component.estimated_hours,
                        },
                        token_count=len(self._encoding.encode(text)),
                    )
                )
        return chunks
