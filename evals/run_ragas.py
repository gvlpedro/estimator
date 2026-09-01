"""RAGAS baseline runner for the citation-grounded estimation pipeline.

Calls the estimation pipeline **in-process** (not over HTTP) so it can reach
``EstimationService._run_structured_pipeline``'s ``retrieved_context_provider``
hook directly — the only way to wire retrieval context into a call today,
since the public ``EstimationService.estimate()`` does not expose that
parameter. Both the service and the AI-service client come from
``app.dependencies``, exactly as the FastAPI app constructs them (same
Redis-backed cache, same ``AI_SERVICE_BASE_URL``).

For each of the 5 golden-set queries (``evals/ragas_golden_set.json``):

1. Retrieve ``k`` historical-budget chunks from the AI service (:8001) —
   this is the RAGAS ``contexts`` input.
2. Run the estimation pipeline with ``prompt_version="v3"`` (the current,
   citation-grounded prompt) and that same retrieved context wired in via
   ``retrieved_context_provider``.
3. Deterministically render the structured ``EstimationResult`` into a plain
   text ``answer`` (no LLM rephrasing — RAGAS must judge the actual
   structured output, not a paraphrase of it).
4. Score the (question, answer, contexts, ground_truth) tuple with RAGAS's
   ``faithfulness``, ``answer_relevancy``, ``context_precision`` and
   ``context_recall`` metrics, using an OpenAI chat model as judge and
   ``text-embedding-3-small`` for the embedding-based metrics.

Usage::

    uv run python evals/run_ragas.py
    uv run python evals/run_ragas.py --judge-model gpt-4o --k 8

This costs real money and takes real time: 5 generation calls (real LLM via
Instructor/LiteLLM) + 4 RAGAS metrics per query, several of which make their
own judge-LLM call(s) per query, plus embedding calls for
``answer_relevancy``. Failures are retried with exponential backoff; a
metric that still fails after retries aborts the run with the real error
instead of fabricating a score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import ValidationError
from ragas.embeddings.base import embedding_factory
from ragas.llms.base import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.config import get_settings
from app.dependencies import get_ai_service_client, get_estimation_service
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    EstimationResult,
    OutputFormat,
    ProjectType,
)
from app.schemas.search import SearchResult
from app.services.ai_client import AIServiceClient, AIServiceError
from app.services.estimation import EstimationService

DEFAULT_GOLDEN_SET_PATH = Path(__file__).parent / "ragas_golden_set.json"
DEFAULT_OUTPUT_MD_PATH = Path(__file__).parent / "ragas_results.md"
DEFAULT_OUTPUT_JSON_PATH = Path(__file__).parent / "ragas_results.json"

DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_K = 5
PROMPT_VERSION = "v3"

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

# The golden-set `question` text is a free-text project description with no
# explicit project_type field, so a ProjectType has to be picked by hand per
# query (EstimationRequest.project_type is a required enum). Judgment calls,
# documented here rather than left implicit:
#   Q1 - mobile banking application                  -> MOBILE_APP (explicitly "mobile")
#   Q2 - headless e-commerce storefront + PWA         -> WEB_SAAS (web storefront/checkout service)
#   Q3 - telemedicine platform, hospital EHR/FHIR     -> WEB_SAAS (web platform + backend integration, not mobile-first)
#   Q4 - industrial IoT sensor telemetry ingestion    -> DATA_PIPELINE ("ingesting sensor telemetry" is a pipeline by definition)
#   Q5 - real-time payments gateway                   -> WEB_SAAS (backend gateway/service, not a mobile app or an ingestion pipeline)
PROJECT_TYPE_BY_QUERY_ID: dict[str, ProjectType] = {
    "Q1": ProjectType.MOBILE_APP,
    "Q2": ProjectType.WEB_SAAS,
    "Q3": ProjectType.WEB_SAAS,
    "Q4": ProjectType.DATA_PIPELINE,
    "Q5": ProjectType.WEB_SAAS,
}


def render_answer(result: EstimationResult) -> str:
    """Deterministically render a structured ``EstimationResult`` into plain
    text for RAGAS to evaluate. No LLM call — plain string formatting only,
    so the eval measures the actual structured output, not a second model's
    rephrasing of it.
    """
    lines = [result.summary, ""]
    for idx, phase in enumerate(result.phases, start=1):
        grounding = (
            "grounded in retrieved historical data"
            if phase.grounded
            else "based on general judgment, not grounded in retrieved data"
        )
        lines.append(
            f"Phase {idx}: {phase.name} — {phase.duration_weeks} week(s), "
            f"{phase.cost_eur} EUR. {phase.summary} ({grounding})"
        )
    lines.append("")
    lines.append(
        f"Total duration: {result.total_duration_weeks} week(s). "
        f"Total cost: {result.total_cost_eur} EUR."
    )
    return "\n".join(lines)


def fetch_context(ai_client: AIServiceClient, question: str, k: int) -> list[dict]:
    """Retrieve comparable historical-budget chunks for ``question`` from the
    AI service, matching the [:2000] query-length convention already used by
    ``app/routers/sessions.py::fetch_retrieved_context``.

    Unlike the production endpoint (which degrades to "no context" on any
    failure so a live estimation never breaks), this eval script needs real
    retrieval to produce a meaningful RAGAS baseline: a failure here aborts
    the run with the real error instead of silently scoring an empty-context
    sample.
    """
    try:
        body = ai_client.search_sync(query=question[:2000], k=k)
    except AIServiceError as exc:
        raise SystemExit(f"AI service search failed for query {question[:60]!r}: {exc}") from exc
    try:
        results = [SearchResult.model_validate(item).model_dump() for item in body.get("results") or []]
    except ValidationError as exc:
        raise SystemExit(f"AI service returned malformed search results: {exc}") from exc
    if not results:
        raise SystemExit(
            f"AI service returned zero results for query {question[:60]!r}; "
            "cannot build RAGAS contexts from an empty retrieval"
        )
    return results


def build_samples(
    queries: list[dict],
    service: EstimationService,
    ai_client: AIServiceClient,
    k: int,
) -> list[dict[str, Any]]:
    """Run retrieval + the structured pipeline for every golden-set query,
    returning the (question, answer, contexts, ground_truth) tuples RAGAS
    needs, plus enough metadata to sanity-check results afterwards.
    """
    samples: list[dict[str, Any]] = []
    for entry in queries:
        qid = entry["id"]
        question = entry["question"]
        project_type = PROJECT_TYPE_BY_QUERY_ID[qid]

        print(f"[{qid}] fetching retrieval context (k={k}) ...")
        context_dicts = fetch_context(ai_client, question, k)
        contexts_text = [item["content"] for item in context_dicts]

        request = EstimationRequest(
            description=question,
            project_type=project_type,
            detail_level=DetailLevel.MEDIUM,
            output_format=OutputFormat.PHASES_TABLE,
        )

        print(f"[{qid}] calling estimation pipeline (prompt_version={PROMPT_VERSION}) ...")
        response, _ops = service._run_structured_pipeline(  # noqa: SLF001 - intentional: see module docstring
            request,
            prompt_version=PROMPT_VERSION,
            retrieved_context_provider=lambda ctx=context_dicts: ctx,
        )
        answer = render_answer(response.result)

        samples.append(
            {
                "id": qid,
                "question": question,
                "project_type": project_type.value,
                "contexts": contexts_text,
                "answer": answer,
                "ground_truth": entry["ground_truth"],
                "cached": response.cached,
            }
        )
    return samples


async def _ascore_with_retry(
    label: str,
    coro_factory,
    *,
    retries: int = 4,
    base_delay: float = 2.0,
):
    """Run a RAGAS ``ascore(...)`` coroutine factory with exponential
    backoff. Raises the real last exception (never fabricates a score) once
    retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 - must catch any judge/embedding-call failure to retry it
            last_exc = exc
            if attempt == retries:
                break
            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"  [retry] {label} failed (attempt {attempt}/{retries}): "
                f"{type(exc).__name__}: {exc} — retrying in {delay:.0f}s"
            )
            await asyncio.sleep(delay)
    raise RuntimeError(f"{label} failed after {retries} attempts: {last_exc}") from last_exc


async def score_samples(
    samples: list[dict[str, Any]],
    judge_model: str,
    embedding_model: str,
) -> list[dict[str, Any]]:
    """Score every sample with faithfulness, answer_relevancy,
    context_precision and context_recall using ragas.metrics.collections
    (ragas>=0.4's modern async metric API — see module docstring)."""
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is not configured; cannot run the RAGAS judge/embeddings.")

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    # ragas.llms.base.llm_factory defaults to max_tokens=1024, which is too
    # small for structured judge outputs like faithfulness's per-statement
    # NLI verdicts (ragas/llms/base.py's own docstring recommends 4096+;
    # confirmed the hard way — the default truncated Faithfulness's judge
    # call on Q1 with IncompleteOutputException).
    llm = llm_factory(judge_model, client=client, max_tokens=4096)
    embeddings = embedding_factory("openai", model=embedding_model, client=client)

    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)

    rows: list[dict[str, Any]] = []
    for sample in samples:
        qid = sample["id"]
        print(f"[{qid}] scoring with RAGAS (judge={judge_model}, embeddings={embedding_model}) ...")

        faith_result = await _ascore_with_retry(
            f"{qid}:faithfulness",
            lambda s=sample: faithfulness.ascore(
                user_input=s["question"], response=s["answer"], retrieved_contexts=s["contexts"]
            ),
        )
        ar_result = await _ascore_with_retry(
            f"{qid}:answer_relevancy",
            lambda s=sample: answer_relevancy.ascore(user_input=s["question"], response=s["answer"]),
        )
        cp_result = await _ascore_with_retry(
            f"{qid}:context_precision",
            lambda s=sample: context_precision.ascore(
                user_input=s["question"], reference=s["ground_truth"], retrieved_contexts=s["contexts"]
            ),
        )
        cr_result = await _ascore_with_retry(
            f"{qid}:context_recall",
            lambda s=sample: context_recall.ascore(
                user_input=s["question"], retrieved_contexts=s["contexts"], reference=s["ground_truth"]
            ),
        )

        scores = {
            "faithfulness": float(faith_result.value),
            "answer_relevancy": float(ar_result.value),
            "context_precision": float(cp_result.value),
            "context_recall": float(cr_result.value),
        }
        rows.append({**sample, "scores": scores})
    return rows


def build_markdown_table(rows: list[dict[str, Any]]) -> str:
    header = "| Query | " + " | ".join(METRIC_NAMES) + " |"
    separator = "|" + "---|" * (len(METRIC_NAMES) + 1)
    lines = [header, separator]

    totals = dict.fromkeys(METRIC_NAMES, 0.0)
    for row in rows:
        scores = row["scores"]
        cells = [f"{scores[name]:.2f}" for name in METRIC_NAMES]
        lines.append(f"| {row['id']} | " + " | ".join(cells) + " |")
        for name in METRIC_NAMES:
            totals[name] += scores[name]

    n = len(rows)
    avg_cells = [f"{totals[name] / n:.2f}" for name in METRIC_NAMES]
    lines.append("| **Average** | " + " | ".join(avg_cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET_PATH)
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("RAGAS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        help="OpenAI chat model used as the RAGAS judge (env: RAGAS_JUDGE_MODEL)",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("RAGAS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="OpenAI embedding model for embedding-based metrics (env: RAGAS_EMBEDDING_MODEL)",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of retrieved chunks per query")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON_PATH)
    args = parser.parse_args()

    t0 = time.perf_counter()

    golden_set = json.loads(args.golden_set.read_text(encoding="utf-8"))
    queries = golden_set["queries"]
    print(f"Loaded {len(queries)} queries from {args.golden_set}")

    service = get_estimation_service()
    ai_client = get_ai_service_client()

    samples = build_samples(queries, service, ai_client, args.k)
    rows = asyncio.run(score_samples(samples, args.judge_model, args.embedding_model))

    table = build_markdown_table(rows)
    print("\n" + table + "\n")

    args.output_md.write_text(table + "\n", encoding="utf-8")
    args.output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")
    print(f"Total wall-clock time: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
