"""Golden-set evaluation: precision@5 and latency across the 4 retrieval configs.

Reads ``evals/golden_retrieval.json``: domain queries (project descriptions
to estimate), each hand-annotated with the historical ``budget_id``s that
are genuinely relevant, plus deliberate distractors documented per query
(see each query's ``note``) — budgets that share surface vocabulary but are
the wrong domain, or (for Q6/Q7) queries engineered to target one method's
known weak spot. Runs every query against the four configurations
(vector/hybrid x rerank on/off), each measured twice — with server-side
``dedupe`` off and on — and reports precision@k and latency, mean and
per-query. See the "Golden set y comparativa de configuraciones" section of
the root README.md for the results and the methodology writeup.

Precision counts DISTINCT budgets in the top-k, not raw chunk slots: the
chunking template repeats the same project header across every component of
a budget, so 3-4 near-duplicate chunks from one dominant budget can occupy
most of the top-k under any retrieval method — counting each duplicate as an
independent "hit" inflates precision without reflecting a genuine ranking
decision. See ``dedupe_by_budget`` in
``app/generation/rag/retrieval/dedupe.py`` for the server-side fix this
script also exercises via the ``dedupe`` request flag.

Requires the service running with the corpus ingested (see README). Stdlib
only: the embedding happens server-side.

    uv run python servicio_ia/golden_eval.py                # from the host
    docker compose run --rm ai-service python golden_eval.py
"""

import json
import os
import statistics
import sys
import urllib.request
from pathlib import Path

BASE_URL = os.getenv("SERVICIO_IA_BASE_URL", "http://localhost:8001")
GOLDEN_PATH = Path(__file__).resolve().parent / "evals" / "golden_retrieval.json"

CONFIGS = [
    ("A", "Vector", "No", {"mode": "vector", "rerank": False}),
    ("B", "Hybrid", "No", {"mode": "hybrid", "rerank": False}),
    ("C", "Vector", "Yes", {"mode": "vector", "rerank": True}),
    ("D", "Hybrid", "Yes", {"mode": "hybrid", "rerank": True}),
]


def search(query: str, k: int, params: dict) -> dict:
    payload = {"query": query, "k": k, **params}
    request = urllib.request.Request(
        f"{BASE_URL}/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def distinct_budget_precision(budget_ids: list[str], relevant: set[str], k: int) -> float:
    """Fraction of k slots occupied by a DISTINCT relevant budget.

    Collapsing to first-seen budget_id before counting means a budget
    repeated across several near-duplicate chunks contributes at most once —
    matching what ``dedupe_by_budget`` does server-side, so this scoring is
    consistent whether or not the request itself asked for ``dedupe``.
    """
    distinct = list(dict.fromkeys(budget_ids))
    hits = sum(1 for b in distinct if b in relevant)
    return hits / k


def run_pass(queries: list[dict], k: int, dedupe: bool) -> dict:
    label = "dedupe=true" if dedupe else "dedupe=false"
    print(f"\n{'=' * 10} Pass: {label} {'=' * 10}\n")

    # Warm-up call, not measured.
    search(queries[0]["query"], k, {"mode": "hybrid", "rerank": True, "dedupe": dedupe})

    print(f"{'Cfg':4s} {'Query':12s} {'P@' + str(k):6s} {'ms':8s} {'top-5 budget_ids'}")
    results = {cfg[0]: {"precisions": [], "latencies_ms": [], "per_query": {}} for cfg in CONFIGS}

    # Interleaved per query (A,B,C,D for query 1, then A,B,C,D for query 2, ...)
    # rather than one config's queries in a block: spreads any residual
    # warm-up/network variance evenly across configurations instead of
    # concentrating it in whichever config happens to run first.
    for q in queries:
        relevant = set(q["relevant_budget_ids"])
        for cfg_label, _search_label, _rerank_label, params in CONFIGS:
            data = search(q["query"], k, {**params, "dedupe": dedupe})
            budget_ids = [r["metadata"]["budget_id"] for r in data["results"]]
            precision = distinct_budget_precision(budget_ids, relevant, k)
            results[cfg_label]["precisions"].append(precision)
            results[cfg_label]["latencies_ms"].append(data["search_time_ms"])
            results[cfg_label]["per_query"][q["id"]] = precision
            print(
                f"{cfg_label:4s} {q['id']:12s} {precision:<6.2f} "
                f"{data['search_time_ms']:<8d} {budget_ids}"
            )
        print()
    return results


def print_report(results: dict, queries: list[dict], k: int, dedupe: bool) -> None:
    label = "dedupe=true" if dedupe else "dedupe=false"
    print(f"## Retrieval evaluation — precision@{k} (distinct budgets) and latency — {label}\n")
    print(f"| Config | Search | Reranking | Precision@{k} | Latency (ms) |")
    print("| --- | --- | --- | --- | --- |")
    for cfg_label, search_label, rerank_label, _params in CONFIGS:
        bucket = results[cfg_label]
        mean_p = statistics.fmean(bucket["precisions"])
        mean_l = statistics.fmean(bucket["latencies_ms"])
        print(f"| {cfg_label} | {search_label} | {rerank_label} | {mean_p:.2f} | {mean_l:.1f} |")

    print(f"\n### Per-query precision@{k} — {label}\n")
    header = "| Query | " + " | ".join(cfg[0] for cfg in CONFIGS) + " |"
    print(header)
    print("| --- | " + " | ".join("---" for _ in CONFIGS) + " |")
    for q in queries:
        row = [q["id"]] + [f"{results[cfg[0]]['per_query'][q['id']]:.2f}" for cfg in CONFIGS]
        print("| " + " | ".join(row) + " |")


def main() -> int:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    queries = golden["queries"]
    k = int(golden.get("k", 5))

    print(f"POST {BASE_URL}/search — golden set ({len(queries)} queries x {len(CONFIGS)} configs, k={k})")
    print("Two passes, same distinct-budget precision metric: dedupe=false (today's retrieval) vs dedupe=true.")

    results_no_dedupe = run_pass(queries, k, dedupe=False)
    results_dedupe = run_pass(queries, k, dedupe=True)

    print()
    print_report(results_no_dedupe, queries, k, dedupe=False)
    print()
    print_report(results_dedupe, queries, k, dedupe=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
