"""Golden-set evaluation harness for the estimation quality of ``POST /estimate``.

Adapted from the Session-15 assignment's illustrative pseudocode to the REAL
contract of this service (``servicio_ia/app/lgraph/router.py``), which does
not match that pseudocode:

- The endpoint is ``/estimate`` (not ``/v1/estimate``), and it takes a full
  meeting ``transcript`` + ``estimation_id`` -- not a one-line
  ``task_description`` + ``project_id``.
- The unit of effort in this system is HOURS, not story points
  (``estimate.total_hours`` -- see ``app.agents.tools.calculate_estimate``).
- There is no synchronous "the system says it doesn't know" response. A
  low-confidence or no-precedent estimate instead PAUSES the graph and comes
  back as ``status="awaiting_human_review"`` with a ``review`` payload
  (``app.lgraph.nodes.human_review_gate`` / ``_needs_human_review``). This
  harness treats that as the abstention/safety signal the assignment asks
  for, and further distinguishes genuine no-precedent
  (``review.validation.no_precedent is True``) from "needs review for some
  other reason, but some precedent exists" (``no_precedent is False``) --
  see ``golden_set.json``'s ``case_type`` field and doc/evaluation.md.
- ``expected_sources_include`` from the assignment's example is NOT checked
  here: the router's response never exposes ``budget_matches`` (the list of
  referenced historical budget_ids) -- only the aggregated
  ``estimate.components`` breakdown. Golden cases document the expected
  historical grounding in their ``grounding`` field for a human to read, but
  ``evaluate_case`` cannot verify it from this API alone. Flagged as a
  known gap in doc/evaluation.md.

Stdlib only (matches ``servicio_ia/golden_eval.py``'s convention, since
``httpx`` may not be present in every environment this script runs in):

    uv run python servicio_ia/eval/run_eval.py
    docker compose exec ai-service python eval/run_eval.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = os.getenv("SERVICIO_IA_BASE_URL", "http://localhost:8001")
SERVICE_TOKEN = os.getenv("AI_SERVICE_TOKEN", "")
GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.json"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def load_golden_set(path: Path | str = GOLDEN_PATH) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["cases"]


def call_estimate(case: dict[str, Any]) -> dict[str, Any]:
    """POST the case's transcript to /estimate and time the round trip.

    Uses a fresh ``estimation_id`` per case (derived from the case id) so
    repeated runs don't collide with a previous run's LangGraph checkpoint
    for the same thread_id.
    """
    payload = json.dumps(
        {"transcript": case["transcript"], "estimation_id": f"golden-{case['id']}"}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}/estimate",
        data=payload,
        headers={"Content-Type": "application/json", "X-Service-Token": SERVICE_TOKEN},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read())
            status_code = response.status
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read()) if exc.fp else {"detail": str(exc)}
        status_code = exc.code
    latency_ms = (time.perf_counter() - started) * 1000
    return {"status_code": status_code, "body": body, "latency_ms": latency_ms}


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Score one case against its response. Encodes what "good" means per case_type.

    - "estimation": passes if the predicted total_hours falls in acceptable_range.
    - "abstention": passes if the graph paused for review AND the reason was a
      genuine lack of historical precedent (review.validation.no_precedent).
    - "partial_precedent": passes if the graph paused for review for some
      OTHER reason (unbudgeted component / out-of-range total), i.e. some
      precedent existed but confidence was still too low to trust as-is.
    """
    body = result["body"]
    case_id = case["id"]
    case_type = case["case_type"]
    status = body.get("status")

    if case_type == "abstention":
        no_precedent = (body.get("review") or {}).get("validation", {}).get("no_precedent")
        passed = status == "awaiting_human_review" and no_precedent is True
        return {
            "id": case_id,
            "type": case_type,
            "passed": passed,
            "status": status,
            "no_precedent": no_precedent,
        }

    if case_type == "partial_precedent":
        no_precedent = (body.get("review") or {}).get("validation", {}).get("no_precedent")
        passed = status == "awaiting_human_review" and no_precedent is False
        return {
            "id": case_id,
            "type": case_type,
            "passed": passed,
            "status": status,
            "no_precedent": no_precedent,
        }

    # case_type == "estimation"
    estimate = body.get("estimate") or {}
    predicted = estimate.get("total_hours")
    low, high = case["acceptable_range"]
    within_range = predicted is not None and low <= predicted <= high
    abs_error = abs(predicted - case["expected_hours"]) if predicted is not None else None

    return {
        "id": case_id,
        "type": case_type,
        "passed": within_range,
        "status": status,
        "predicted_hours": predicted,
        "expected_hours": case["expected_hours"],
        "abs_error": abs_error,
    }


def build_report(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case evaluations into the minimum required metrics."""
    estimation_cases = [e for e in evaluations if e["type"] == "estimation"]
    safety_cases = [e for e in evaluations if e["type"] in ("abstention", "partial_precedent")]
    within_range = [e for e in estimation_cases if e["passed"]]
    errors = [e["abs_error"] for e in estimation_cases if e.get("abs_error") is not None]

    return {
        "within_range_rate": (
            len(within_range) / len(estimation_cases) if estimation_cases else None
        ),
        "mean_absolute_error": sum(errors) / len(errors) if errors else None,
        "abstention_correct": all(e["passed"] for e in safety_cases) if safety_cases else None,
        "mean_latency_ms": (
            sum(e["latency_ms"] for e in evaluations) / len(evaluations) if evaluations else None
        ),
        "cases": evaluations,
    }


def run() -> dict[str, Any]:
    cases = load_golden_set()
    evaluations = []
    for case in cases:
        result = call_estimate(case)
        evaluation = evaluate_case(case, result)
        evaluation["latency_ms"] = round(result["latency_ms"], 1)
        evaluations.append(evaluation)
    return build_report(evaluations)


def main() -> int:
    print(
        f"POST {BASE_URL}/estimate -- {len(load_golden_set())} golden case(s). "
        "This calls the real deployed model and spends real tokens.",
        file=sys.stderr,
    )
    report = run()

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / f"run_{int(time.time())}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport written to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
