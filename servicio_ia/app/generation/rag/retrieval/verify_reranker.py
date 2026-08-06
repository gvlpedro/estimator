"""Verify the cross-encoder wrapper loads its model and reranks correctly.

Standalone sanity check for ``CrossEncoderReranker`` — no vector search, no
Postgres involved: loads the real ``cross-encoder/ms-marco-MiniLM-L-6-v2``
model and scores one query against three budget-component excerpts pulled
from the seeded historical corpus (``data/budgets_sample.json``), expecting
a known relevance ordering. Same idea as
``embedding_pipeline/SANITY_CHECK.md`` for the bi-encoder, but executable
and exit-code-gated so it can confirm the reranker prerequisite before it
gets wired into ``/search``.

    uv run python -m app.generation.rag.retrieval.verify_reranker   # from servicio_ia/, host
    docker compose exec ai-service python -m app.generation.rag.retrieval.verify_reranker
"""

import sys

from app.generation.rag.retrieval.cross_encoder import RERANKER_MODEL, CrossEncoderReranker

QUERY = "OAuth 2.0 authentication REST API for a fintech client"

# (label, document text) — all three components come from the seeded corpus:
# AUTH-001 and TXN-001 from BUD-2024-014 (same fintech project), PDM-001 from
# BUD-2025-013 (wind-farm predictive maintenance, unrelated domain).
DOCUMENTS = [
    (
        "AUTH-001 (OAuth backend, same fintech project)",
        "Component: OAuth 2.0 authentication backend\n"
        "Description: Implementation of OAuth 2.0 flows (authorization code, refresh "
        "token) with JWT-based session management, multi-tenant token isolation, and "
        "rate limiting per client.\nTech stack: ruby_on_rails, postgresql, redis",
    ),
    (
        "TXN-001 (same project, unrelated capability)",
        "Component: Accounts and transactions API\n"
        "Description: Berlin Group-compliant account information endpoints with "
        "pagination, balance snapshots, transaction enrichment, and response caching "
        "per consent scope.\nTech stack: ruby_on_rails, postgresql, redis",
    ),
    (
        "PDM-001 (different sector, unrelated domain)",
        "Component: Predictive maintenance models\n"
        "Description: Anomaly detection on vibration signatures and gearbox "
        "temperature trends, remaining-useful-life estimation per component, model "
        "retraining pipeline, and drift monitoring.\nTech stack: python, pytorch, mlflow",
    ),
]

# Expected rank order by construction: direct match > same-project noise > out-of-domain.
EXPECTED_ORDER = [0, 1, 2]


def main() -> int:
    print(f"Loading {RERANKER_MODEL} (downloads weights on first run)...")
    reranker = CrossEncoderReranker()

    labels = [label for label, _ in DOCUMENTS]
    documents = [text for _, text in DOCUMENTS]

    print(f'\nQuery: "{QUERY}"\n')
    ranked = reranker.rerank(QUERY, documents)
    for position, (original_index, score) in enumerate(ranked, start=1):
        print(f"  {position}. score={score:+.4f}  {labels[original_index]}")

    ranked_indices = [original_index for original_index, _ in ranked]

    print()
    if ranked_indices == EXPECTED_ORDER:
        print("Sanity check (AUTH-001 > TXN-001 > PDM-001): PASS")
        return 0

    print("Sanity check (AUTH-001 > TXN-001 > PDM-001): FAIL")
    print(f"  expected order {EXPECTED_ORDER}, got {ranked_indices}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
