"""Exercise POST /search with five representative queries and print the results.

Each query probes the corpus from a different angle:

1. Known direct component  — should hit an almost perfect match (sanity check).
2. Semantic rephrasing     — same idea, different vocabulary: meaning vs keywords.
3. Out-of-domain           — nothing in the corpus should match: distances stay high.
4. Ambiguous query         — short and generic: ranking without a dominant match.
5. Highly specific query   — precise tech vocabulary: related-technology discrimination.

Requires the service running with the corpus ingested (see README). Stdlib only:
the embedding happens server-side, so this script needs no API key and no deps.

    uv run python servicio_ia/query_examples.py                # from the host
    docker compose run --rm ai_service python query_examples.py
"""

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.getenv("SERVICIO_IA_BASE_URL", "http://localhost:8001")
K = 5
PREVIEW_CHARS = 120

QUERIES = [
    (
        "Componente directo conocido",
        "REST API development with JWT authentication for financial sector",
    ),
    (
        "Reformulación semántica",
        "secure backend service with token-based access control for banking applications",
    ),
    (
        "Dominio distinto",
        "mobile application for restaurant reservations",
    ),
    (
        "Consulta ambigua",
        "integration with external system",
    ),
    (
        "Consulta muy específica",
        "migration from monolith to microservices architecture using Kubernetes",
    ),
]


def search(query: str) -> dict:
    request = urllib.request.Request(
        f"{BASE_URL}/search",
        data=json.dumps({"query": query, "k": K}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def preview(content: str) -> str:
    """Collapse whitespace and truncate for one-line terminal display."""
    flat = " ".join(content.split())
    return flat[:PREVIEW_CHARS] + ("…" if len(flat) > PREVIEW_CHARS else "")


def main() -> int:
    print(f"POST {BASE_URL}/search — top-{K} por query\n")
    for number, (angle, query) in enumerate(QUERIES, start=1):
        try:
            data = search(query)
        except urllib.error.URLError as exc:
            print(f"ERROR: no se pudo invocar {BASE_URL}/search ({exc.reason}).")
            print("¿Está el servicio levantado y el corpus ingestado? Ver README.")
            return 1

        print(f"=== {number}. {angle} ===")
        print(f'Query: "{query}"  ({data["search_time_ms"]} ms)')
        if not data["results"]:
            print("  (sin resultados — ¿corpus vacío?)")
        for rank, result in enumerate(data["results"], start=1):
            print(
                f"  {rank}. d={result['distance']:.4f}  "
                f"chunk={result['chunk_id']:<4d} {result['chunk_type']}  "
                f"{preview(result['content'])}"
            )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
