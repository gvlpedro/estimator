"""CLI runner: feed the golden dataset against the live FastAPI app.

Default transport is the in-process ``TestClient`` — no need to start an
external server. ``--http BASE`` switches to a real HTTP target so you can
point this at a staging deployment.

Usage::

    uv run python -m evals.run --mode actor
    uv run python -m evals.run --mode acb --limit 5 --output /tmp/acb.json
    uv run python -m evals.run --mode actor --http http://localhost:8000

The runner prints a compact table per case + per metric and a final tally.
``--output`` persists the full report (case scores + meta) as JSON for
diffing across runs.

Comparing ``actor`` vs ``acb`` on the same dataset is the whole point: the
actor mode is the single-shot baseline, ``acb`` adds the Critic loop. If
ACB doesn't lift the pass rate (or move the failures to different cases)
you should question whether the extra LLM calls are worth their cost.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.estimation import EstimationResult
from evals.dataset import GoldenCase, load_dataset
from evals.metrics import MetricResult, run_all_metrics

Mode = Literal["actor", "acb"]

_ACTOR_PATH = "/api/v1/estimate"
_ACB_PATH = "/api/v1/estimate/acb"

_GUARDRAIL_OUT_OF_SCOPE_REASONS = {"prompt_injection", "moderation", "pii"}


def _json_body(case: GoldenCase) -> dict[str, str]:
    return {
        "description": case.description,
        "project_type": case.project_type,
        "detail_level": case.detail_level,
        "output_format": case.output_format,
    }


def _endpoint(mode: Mode) -> str:
    return _ACB_PATH if mode == "acb" else _ACTOR_PATH


def _guardrail_payload(reason: str, message: str) -> dict[str, Any]:
    """Synthesize an EstimationResponse-shaped payload for a layer-1 rejection.

    When the input guardrail blocks an adversarial / unsafe description
    before the LLM is called, that *is* the out-of-scope outcome. We map
    it to the same envelope the LLM would have produced (cost=0,
    duration=1, summary prefixed with "Out of scope:") so the existing
    metrics can score it as a pass for cases that expect
    ``expected_out_of_scope: true``.
    """
    summary = f"Out of scope: rejected by input guardrail ({reason}): {message}"[:1200]
    return {
        "result": {
            "summary": summary,
            "confidence_pct": 0,
            "phases": [
                {
                    "name": "n/a",
                    "duration_weeks": 1,
                    "cost_eur": 0,
                    "summary": "Blocked by input guardrail; no estimation produced.",
                }
            ],
            "total_duration_weeks": 1,
            "total_cost_eur": 0,
        },
        "prompt_version": f"guardrail/{reason}",
        "cached": False,
    }


def _maybe_guardrail_payload(case: GoldenCase, response: Any) -> dict[str, Any] | None:
    """Return a synthetic payload if the response is a guardrail rejection that
    counts as the expected out-of-scope outcome for this case; else ``None``.
    """
    if response.status_code != 400 or not case.expected_out_of_scope:
        return None
    try:
        detail = response.json().get("detail") or {}
    except Exception:  # noqa: BLE001
        return None
    reason = detail.get("reason") if isinstance(detail, dict) else None
    if reason not in _GUARDRAIL_OUT_OF_SCOPE_REASONS:
        return None
    return _guardrail_payload(reason, detail.get("message", ""))


def _run_case_in_process(
    client: TestClient, case: GoldenCase, mode: Mode
) -> tuple[dict[str, Any], int]:
    t0 = time.perf_counter()
    response = client.post(_endpoint(mode), json=_json_body(case))
    latency_ms = int((time.perf_counter() - t0) * 1000)
    synthetic = _maybe_guardrail_payload(case, response)
    if synthetic is not None:
        return synthetic, latency_ms
    response.raise_for_status()
    return response.json(), latency_ms


def _run_case_over_http(
    client: httpx.Client, case: GoldenCase, mode: Mode
) -> tuple[dict[str, Any], int]:
    t0 = time.perf_counter()
    response = client.post(_endpoint(mode), json=_json_body(case))
    latency_ms = int((time.perf_counter() - t0) * 1000)
    synthetic = _maybe_guardrail_payload(case, response)
    if synthetic is not None:
        return synthetic, latency_ms
    response.raise_for_status()
    return response.json(), latency_ms


def _evaluate(case: GoldenCase, payload: dict[str, Any]) -> list[MetricResult]:
    result = EstimationResult.model_validate(payload["result"])
    return run_all_metrics(case, result)


def _format_row(case_id: str, latency_ms: int, metrics: list[MetricResult]) -> str:
    parts = [f"{case_id:<32}", f"{latency_ms:>6} ms"]
    for m in metrics:
        mark = "PASS" if m.passed else "FAIL"
        parts.append(f"{m.name}={mark}({m.score:.2f})")
    return " | ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("actor", "acb"), default="actor")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--http",
        default=None,
        help="Base URL for HTTP mode (e.g. http://localhost:8000)",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cases = load_dataset()
    if args.limit is not None:
        cases = cases[: args.limit]

    rows: list[dict[str, Any]] = []
    pass_count = 0
    metric_totals: dict[str, list[bool]] = {}

    print(f"Running {len(cases)} cases against mode={args.mode}")

    if args.http:
        http_client = httpx.Client(base_url=args.http, timeout=180.0)
        runner = lambda case: _run_case_over_http(http_client, case, args.mode)  # noqa: E731
        ctx = http_client
    else:
        test_client = TestClient(app)
        runner = lambda case: _run_case_in_process(test_client, case, args.mode)  # noqa: E731
        ctx = test_client

    try:
        with ctx:
            for case in cases:
                try:
                    payload, latency_ms = runner(case)
                except Exception as exc:  # noqa: BLE001
                    print(f"{case.id}: ERROR {type(exc).__name__}: {str(exc)[:200]}")
                    rows.append({"id": case.id, "error": str(exc)[:400]})
                    continue

                metrics = _evaluate(case, payload)
                print(_format_row(case.id, latency_ms, metrics))

                row_passed = all(m.passed for m in metrics)
                if row_passed:
                    pass_count += 1
                rows.append(
                    {
                        "id": case.id,
                        "latency_ms": latency_ms,
                        "passed": row_passed,
                        "metrics": [m.__dict__ for m in metrics],
                        "result_summary": payload["result"]["summary"][:200],
                        "prompt_version": payload.get("prompt_version"),
                        "acb_final_decision": (
                            payload.get("acb", {}).get("final_decision")
                            if isinstance(payload.get("acb"), dict)
                            else None
                        ),
                        "acb_iterations_run": (
                            payload.get("acb", {}).get("iterations_run")
                            if isinstance(payload.get("acb"), dict)
                            else None
                        ),
                    }
                )
                for m in metrics:
                    metric_totals.setdefault(m.name, []).append(m.passed)
    finally:
        pass

    print("\nSummary")
    print(f"  {pass_count}/{len(cases)} cases passing all metrics")
    for name, results in metric_totals.items():
        print(f"  {name}: {sum(results)}/{len(results)}")

    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "cases_total": len(cases),
                    "cases_passing": pass_count,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nReport written to {args.output}")

    return 0 if pass_count == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
