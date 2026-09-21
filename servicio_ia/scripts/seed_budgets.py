"""Seed the vector store with the historical budgets corpus.

POSTs every budget in ``data/budgets_sample.json`` to /embeddings/ingest.
Idempotent: a 409 means the budget was already ingested and counts as OK,
so re-running after a partial failure only ingests what is missing.

The embedding happens server-side, so this script needs no API key and no
deps beyond the stdlib — but it DOES need the service up and the database
migrated (see README).

    uv run python servicio_ia/scripts/seed_budgets.py            # from the host
    docker compose run --rm ai_service python scripts/seed_budgets.py
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = os.getenv("SERVICIO_IA_BASE_URL", "http://localhost:8001")
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "budgets_sample.json"
DOCUMENT_TYPE = "historical_budget"


def ingest(budget: dict) -> tuple[int, dict]:
    """POST one budget; return (status_code, response_body)."""
    payload = {
        # ``::<budget_id>`` keeps one source_path per budget even though all
        # budgets live in the same file — source_path is the idempotency key,
        # and the separator matches the manual ingest example in the README.
        "source_path": f"data/budgets_sample.json::{budget['budget_id']}",
        "document_type": DOCUMENT_TYPE,
        "content": budget,
    }
    request = urllib.request.Request(
        f"{BASE_URL}/embeddings/ingest",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def main() -> int:
    budgets = json.loads(DATA_FILE.read_text())
    print(f"Seeding {len(budgets)} historical budgets → {BASE_URL}/embeddings/ingest\n")

    created = skipped = failed = 0
    for budget in budgets:
        budget_id = budget["budget_id"]
        try:
            status, body = ingest(budget)
        except urllib.error.URLError as exc:
            print(f"ERROR: cannot reach {BASE_URL} ({exc.reason}).")
            print("Is the service up and the database migrated? See README.")
            return 1

        if status == 200:
            created += 1
            print(f"  {budget_id}: ingested — document_id={body['document_id']}, "
                  f"chunks={body['chunks_created']}, {body['ingestion_time_ms']} ms")
        elif status == 409:
            skipped += 1
            print(f"  {budget_id}: already ingested (document_id={body['document_id']})")
        else:
            failed += 1
            print(f"  {budget_id}: FAILED with HTTP {status}: {body}")

    print(f"\nDone: {created} ingested, {skipped} already present, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
