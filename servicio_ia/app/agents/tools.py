"""The two agent tools: flat JSON Schema (Responses API, ``strict: true``) + implementations.

``search_budgets`` wraps the S9-S10 hybrid retrieval + cross-encoder
reranking pipeline (``app.generation.rag.retrieval``) — it does not
reimplement ranking, only maps the agent's arguments onto that pipeline and
reshapes its output for the agent. ``calculate_estimate`` is pure Python: no
LLM call, no I/O, no network.

Tool descriptions are the only thing the model reads to decide when to call
each tool — it never sees this file — so they carry the calling convention
explicitly (one ``search_budgets`` call per component, one final
``calculate_estimate`` call over all of them) rather than leaving that to
the system prompt alone.
"""

from __future__ import annotations

import asyncio
import statistics
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.generation.rag.retrieval.cross_encoder import CrossEncoderReranker
from app.generation.rag.retrieval.reranked_search import reranked_search
from app.generation.rag.retrieval.retrieve import RECALL_K

# A flat contingency buffer added to every component's central estimate.
# Kept transparent on purpose -- no hidden multipliers.
CONTINGENCY_FACTOR = 0.15

# Historical items surfaced per search_budgets call. Matches the /search
# endpoint's own default k.
SEARCH_TOP_K = 5

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "search_budgets",
        "description": (
            "Retrieve historical project budgets whose components are similar to a "
            "described piece of work, ranked by relevance (hybrid vector + lexical "
            "retrieval, cross-encoder reranked, deduplicated to one item per budget). "
            "Call this ONCE PER DISTINCT COMPONENT of the estimate -- never bundle "
            "several unrelated components into one query, since that dilutes the "
            "semantic match and returns irrelevant results. Each returned item "
            "carries an 'estimated_hours' figure: use those as the reference_amounts "
            "you later pass to calculate_estimate for that same component."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A natural-language description of ONE component, phrased the "
                        "way a similar past project would be described: what it does, "
                        "its client sector if known, and its key technologies. "
                        "Example: 'OAuth2 authentication backend with JWT and refresh "
                        "tokens for a fintech client'. Do not paste the raw transcript "
                        "and do not describe more than one component per call."
                    ),
                },
                "filters": {
                    "type": ["object", "null"],
                    "description": (
                        "Optional narrowing filters. Pass null when no filter is "
                        "needed -- relevance ranking alone is usually enough."
                    ),
                    "properties": {
                        "sector": {
                            "type": ["string", "null"],
                            "description": (
                                "Restrict results to historical budgets from this "
                                "client sector (e.g. 'finance', 'logistics', "
                                "'healthcare', 'ecommerce'). Null for no restriction."
                            ),
                        },
                        "technology": {
                            "type": ["string", "null"],
                            "description": (
                                "Restrict results to historical budgets whose main "
                                "technology matches this (e.g. 'Python', 'React "
                                "Native'). Null for no restriction."
                            ),
                        },
                    },
                    "required": ["sector", "technology"],
                    "additionalProperties": False,
                },
            },
            "required": ["query", "filters"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "calculate_estimate",
        "description": (
            "Deterministically compute the effort estimate, in hours, for one or "
            "more components from the reference hour amounts already found via "
            "search_budgets. Pure arithmetic (median of the references plus a fixed "
            "contingency buffer) -- it does not search, does not call any model, and "
            "will not invent a number for a component you did not search for. Call it "
            "ONCE, after searching for every component, passing all of them together "
            "so the result includes both the per-component breakdown and the grand "
            "total. A component with an empty reference_amounts list is costed at 0 "
            "hours and flagged 'unbudgeted' -- if you see that, consider searching "
            "again with a broader query before finalizing the estimate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "description": "One entry per component identified in the transcript.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": (
                                    "Short human-readable component name, e.g. 'ERP "
                                    "integration (SAP)'."
                                ),
                            },
                            "reference_amounts": {
                                "type": "array",
                                "description": (
                                    "The 'estimated_hours' values from this "
                                    "component's search_budgets results. Empty array "
                                    "if no historical reference was found."
                                ),
                                "items": {"type": "number"},
                            },
                        },
                        "required": ["name", "reference_amounts"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["components"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def calculate_estimate(args: dict[str, Any]) -> dict[str, Any]:
    """Cost each component from its historical reference amounts, then total.

    Median over mean: one unusually cheap or expensive past project (a
    discount, a fixed-price loss-leader) should not drag the whole estimate
    with it the way it would drag a mean.
    """
    components = args["components"]
    breakdown: list[dict[str, Any]] = []
    total = 0.0

    for component in components:
        name = component["name"]
        refs = component.get("reference_amounts", [])

        if refs:
            central = statistics.median(refs)
            hours = round(central * (1 + CONTINGENCY_FACTOR), 1)
            unbudgeted = False
        else:
            hours = 0.0
            unbudgeted = True

        total += hours
        breakdown.append(
            {
                "name": name,
                "reference_count": len(refs),
                "estimated_hours": hours,
                "unbudgeted": unbudgeted,
            }
        )

    total = round(total, 1)
    unbudgeted_names = [c["name"] for c in breakdown if c["unbudgeted"]]
    summary = f"total={total}h across {len(breakdown)} components"
    if unbudgeted_names:
        summary += f"; unbudgeted: {', '.join(unbudgeted_names)}"

    return {"components": breakdown, "total_hours": total, "summary": summary}


async def search_budgets(
    args: dict[str, Any],
    *,
    session: AsyncSession,
    embedder: OpenAIEmbedder,
    reranker: CrossEncoderReranker,
    top_k: int = SEARCH_TOP_K,
) -> dict[str, Any]:
    """Wrap the S9-S10 retrieval pipeline: hybrid search, reranked, deduped by budget.

    ``sector``/``technology`` filtering happens AFTER retrieval, over the
    widened ``RECALL_K`` candidate pool the reranker already scored: the
    storage layer has no metadata pre-filter, so narrowing before ranking
    isn't available, but narrowing after ranking still gets it right as long
    as the pool is wide enough -- which is exactly what ``RECALL_K`` is for.
    """
    query = args["query"]
    filters = args.get("filters") or {}
    sector = (filters.get("sector") or "").strip().lower()
    technology = (filters.get("technology") or "").strip().lower()

    query_vector = await asyncio.to_thread(embedder.embed_one, query)
    candidates = await reranked_search(
        session,
        reranker,
        mode="hybrid",
        query=query,
        query_vector=query_vector,
        k=RECALL_K,
        dedupe=True,
    )

    items: list[dict[str, Any]] = []
    for row, score in candidates:
        meta = row.meta or {}
        if sector and sector != str(meta.get("client_sector", "")).lower():
            continue
        if technology and technology not in str(meta.get("main_technology", "")).lower():
            continue
        items.append(
            {
                "budget_id": meta.get("budget_id"),
                "component_id": meta.get("component_id"),
                "estimated_hours": meta.get("estimated_hours"),
                "sector": meta.get("client_sector"),
                "technology": meta.get("main_technology"),
                "complexity": meta.get("complexity"),
                "year": meta.get("year"),
                "relevance_score": round(float(score), 4),
                "preview": row.content[:220],
            }
        )
        if len(items) >= top_k:
            break

    budgets = ", ".join(str(item["budget_id"]) for item in items)
    summary = f"{len(items)} historical item(s) found" + (f" (budgets: {budgets})" if items else "")
    return {"query": query, "filters": filters, "results": items, "summary": summary}
