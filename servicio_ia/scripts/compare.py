"""Sanity check for the embedding pipeline: compare chunk pairs by similarity.

Embeds three hand-picked pairs of budget component chunks and prints their
cosine similarity, computed BY HAND with the stdlib (no numpy): if the model
captures semantics, near-duplicate components must score higher than related
ones, and unrelated ones must score lowest.

Usage (needs OPENAI_API_KEY in the environment):
    uv run --env-file .env python servicio_ia/scripts/compare.py
"""

import json
import math
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from app.embedding_pipeline.chunker import JSONStructuralChunker  # noqa: E402
from app.embedding_pipeline.embedder import OpenAIEmbedder  # noqa: E402
from app.embedding_pipeline.schemas import Budget  # noqa: E402

BUDGETS_PATH = SERVICE_ROOT / "data" / "budgets_sample.json"

# Each pair states its expectation so the printed report is self-explanatory.
# Ordered from most to least similar — the sanity check is that cosine
# similarity preserves this ordering.
PAIRS = [
    (
        "BUD-2024-014::AUTH-001",
        "BUD-2023-004::AUTH-001",
        "same concept, same sector and stack (OAuth 2.0 backends in fintech, Rails)",
    ),
    (
        "BUD-2022-003::PAY-001",
        "BUD-2024-009::CHK-001",
        "related concept, different vertical and stack (payments: grocery/Node vs fashion/Laravel)",
    ),
    (
        "BUD-2024-014::AUTH-001",
        "BUD-2025-013::PDM-001",
        "unrelated (fintech OAuth backend vs wind-farm predictive maintenance ML)",
    ),
]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain-stdlib cosine similarity: dot(a, b) / (|a| * |b|)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def main() -> None:
    budgets = [Budget.model_validate(raw) for raw in json.loads(BUDGETS_PATH.read_text())]
    chunks = {chunk.chunk_id: chunk for chunk in JSONStructuralChunker().chunk(budgets)}

    # Embed each distinct chunk once, in a single batched API call.
    involved_ids = sorted({chunk_id for left, right, _ in PAIRS for chunk_id in (left, right)})
    embedded = OpenAIEmbedder().embed_many([chunks[chunk_id] for chunk_id in involved_ids])
    vectors = {chunk.chunk_id: chunk.embedding for chunk in embedded}

    print(f"\nModel: text-embedding-3-small (dim={len(embedded[0].embedding)})\n")
    scores = []
    for left, right, expectation in PAIRS:
        score = cosine_similarity(vectors[left], vectors[right])
        scores.append(score)
        print(f"{left}  vs  {right}")
        print(f"  expectation: {expectation}")
        print(f"  cosine similarity: {score:.4f}\n")

    ordering_ok = scores == sorted(scores, reverse=True)
    print(f"Sanity check (pair 1 > pair 2 > pair 3): {'PASS' if ordering_ok else 'FAIL'}")
    if not ordering_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
