"""Stress runner: drive scenarios × attachments × repeats against the API,
emit a per-turn CSV and a human-readable ``REPORT.md``.

Usage::

    uv run python -m evals.stress.run \\
        --scenarios growing,pivot,contradiction \\
        --attachment-sizes 0KB,5KB,20KB,50KB,100KB \\
        --turns 6 \\
        --repeats 3 \\
        --latency-budget-ms 8000 \\
        --cost-budget-usd 0.05 \\
        --output evals/stress/results.csv \\
        --report evals/stress/REPORT.md

``--http BASE`` switches transport from the in-process ``TestClient`` to a
real HTTP target. The structlog ``turn_observed`` event only flows through
the in-process transport (we capture it via ``structlog.testing``); in HTTP
mode the runner falls back to a thinner observation built from response
timings and the session snapshot. The CSV documents which transport produced
each row in the ``transport`` column.
"""

from __future__ import annotations

import argparse
import csv
import io
import statistics
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import structlog
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from app.main import app
from evals.stress.attachments import (
    ATTACHMENT_SIZES,
    SyntheticAttachment,
    SyntheticAttachmentSpec,
    generate_attachment,
    spec_by_label,
)
from evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
    SessionSnapshot,
    TurnObservation,
)
from evals.stress.scenarios import (
    FactAssertion,
    StressScenario,
    StressTurn,
    SUPPORTED_TURN_LENGTHS,
    scenarios_for_length,
)


CSV_COLUMNS: tuple[str, ...] = (
    "profile",
    "attachment_label",
    "repeat",
    "turn_index",
    "session_id",
    "transport",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "cache_hit_kind",
    "last_resolved_tier",
    "latency_pass",
    "cost_pass",
    "memory_recall_passes",
    "memory_recall_total",
    "marker_in_summary",
)


# ---------------------------------------------------------------------------
# Per-turn execution
# ---------------------------------------------------------------------------


@contextmanager
def _capture_turn_observed() -> Iterator[list[dict[str, Any]]]:
    """Yield a list that will hold every ``turn_observed`` event emitted while
    the context is active. We reset structlog defaults at the entry so the
    capture processor sees the events even if the app configured a renderer
    that swallows them."""
    structlog.reset_defaults()
    with capture_logs() as logs:
        yield logs


def _build_form_data(turn: StressTurn) -> dict[str, str]:
    return {
        "transcript": turn.transcript,
        "project_type": "web_saas",  # overwritten by caller from scenario
        "detail_level": "medium",
        "output_format": "phases_table",
    }


def _post_turn_in_process(
    client: TestClient,
    *,
    session_id: str,
    turn: StressTurn,
    scenario: StressScenario,
    attachment: SyntheticAttachment | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    form = _build_form_data(turn) | {"project_type": scenario.project_type}
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None
    if attachment is not None:
        files = [
            (
                "attachments",
                (attachment.filename, attachment.data, "application/pdf"),
            )
        ]
    t0 = time.perf_counter()
    with _capture_turn_observed() as logs:
        response = client.post(
            f"/api/v1/sessions/{session_id}/estimate",
            data=form,
            files=files,
        )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    response.raise_for_status()
    return response.json(), logs, latency_ms


def _post_turn_over_http(
    client: httpx.Client,
    *,
    session_id: str,
    turn: StressTurn,
    scenario: StressScenario,
    attachment: SyntheticAttachment | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    form = _build_form_data(turn) | {"project_type": scenario.project_type}
    files = None
    if attachment is not None:
        files = {
            "attachments": (
                attachment.filename,
                attachment.data,
                "application/pdf",
            )
        }
    t0 = time.perf_counter()
    response = client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data=form,
        files=files,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    response.raise_for_status()
    # HTTP mode cannot see in-process structlog events.
    return response.json(), [], latency_ms


def _observation_from_event(
    event: dict[str, Any] | None,
    *,
    session_id: str,
    turn_index: int,
    fallback_latency_ms: int,
) -> TurnObservation:
    """Build a TurnObservation either from the captured event (in-process) or
    from the response roundtrip when the event is unavailable (HTTP mode)."""
    if event is None:
        return TurnObservation(
            turn_index=turn_index,
            session_id=session_id,
            enriched_transcript_chars=0,
            attachments_total_chars=0,
            messages_in_window=0,
            anchors_count=0,
            summary_chars=0,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            latency_ms=fallback_latency_ms,
            cache_hit_kind="none",
            last_resolved_tier=None,
        )
    return TurnObservation(
        turn_index=event["turn_index"],
        session_id=event["session_id"],
        enriched_transcript_chars=event["enriched_transcript_chars"],
        attachments_total_chars=event["attachments_total_chars"],
        messages_in_window=event["messages_in_window"],
        anchors_count=event["anchors_count"],
        summary_chars=event["summary_chars"],
        tokens_in=event["tokens_in"],
        tokens_out=event["tokens_out"],
        cost_usd=event["cost_usd"],
        latency_ms=event["latency_ms"],
        cache_hit_kind=event["cache_hit_kind"],
        last_resolved_tier=event.get("last_resolved_tier"),
    )


def _pick_turn_event(logs: list[dict[str, Any]], turn_index: int) -> dict[str, Any] | None:
    for ev in logs:
        if ev.get("event") == "turn_observed" and ev.get("turn_index") == turn_index:
            return ev
    return None


# ---------------------------------------------------------------------------
# Drift evaluation
# ---------------------------------------------------------------------------


def _snapshot_from_session_response(
    payload: dict[str, Any], last_assistant_summary: str
) -> SessionSnapshot:
    return SessionSnapshot(
        session_id=payload["session_id"],
        summary=payload.get("project_metadata", {}).get("agreed_scope", "") or "",
        anchors=tuple(),
        metadata=payload.get("project_metadata", {}) or {},
        last_assistant_summary=last_assistant_summary,
    )


def _evaluate_recall(
    snapshot: SessionSnapshot,
    assertions: tuple[FactAssertion, ...],
    *,
    at_turn: int,
) -> tuple[int, int]:
    """Return (passes, total). Forbidden facts use inverted semantics."""
    passes = 0
    total = 0
    for a in assertions:
        if a.introduced_at_turn > at_turn:
            continue
        total += 1
        metric = MemoryDriftMetric(
            fact=a.fact,
            where=list(a.where),
            forbidden=(a.kind == "forbidden"),
            label=a.label or a.fact,
        )
        if metric.evaluate(snapshot).passed:
            passes += 1
    return passes, total


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def _run_scenario(
    *,
    transport: str,
    in_process_client: TestClient | None,
    http_client: httpx.Client | None,
    scenario: StressScenario,
    attachment_spec: SyntheticAttachmentSpec,
    repeat_index: int,
    latency_budget_ms: int,
    cost_budget_usd: float,
) -> list[dict[str, Any]]:
    """Run one (scenario, attachment, repeat) tuple, return rows ready for CSV."""
    attachment = generate_attachment(attachment_spec)
    if transport == "in_process":
        assert in_process_client is not None
        create_resp = in_process_client.post("/api/v1/sessions")
    else:
        assert http_client is not None
        create_resp = http_client.post("/api/v1/sessions")
    create_resp.raise_for_status()
    session_id = create_resp.json()["session_id"]

    latency_metric = LatencyBudgetMetric(budget_ms=latency_budget_ms)
    cost_metric = CostBudgetMetric(budget_usd=cost_budget_usd)
    rows: list[dict[str, Any]] = []
    last_assistant_summary = ""

    for turn in scenario.turns:
        if transport == "in_process":
            payload, logs, roundtrip_ms = _post_turn_in_process(
                in_process_client,
                session_id=session_id,
                turn=turn,
                scenario=scenario,
                attachment=attachment,
            )
            event = _pick_turn_event(logs, turn.index)
        else:
            payload, _logs, roundtrip_ms = _post_turn_over_http(
                http_client,
                session_id=session_id,
                turn=turn,
                scenario=scenario,
                attachment=attachment,
            )
            event = None

        observation = _observation_from_event(
            event,
            session_id=session_id,
            turn_index=turn.index,
            fallback_latency_ms=roundtrip_ms,
        )
        last_assistant_summary = payload["result"]["summary"]

        latency_res = latency_metric.evaluate(observation)
        cost_res = cost_metric.evaluate(observation)

        marker_in_summary = (
            attachment is not None
            and attachment.marker_phrase.casefold() in last_assistant_summary.casefold()
        )

        rows.append(
            {
                "profile": scenario.profile,
                "attachment_label": attachment_spec.label,
                "repeat": repeat_index,
                "turn_index": turn.index,
                "session_id": session_id,
                "transport": transport,
                "enriched_transcript_chars": observation.enriched_transcript_chars,
                "attachments_total_chars": observation.attachments_total_chars,
                "messages_in_window": observation.messages_in_window,
                "anchors_count": observation.anchors_count,
                "summary_chars": observation.summary_chars,
                "tokens_in": observation.tokens_in,
                "tokens_out": observation.tokens_out,
                "cost_usd": round(observation.cost_usd, 6),
                "latency_ms": observation.latency_ms,
                "cache_hit_kind": observation.cache_hit_kind,
                "last_resolved_tier": observation.last_resolved_tier or "",
                "latency_pass": int(latency_res.passed),
                "cost_pass": int(cost_res.passed),
                "memory_recall_passes": 0,
                "memory_recall_total": 0,
                "marker_in_summary": int(marker_in_summary),
            }
        )

    # Drift evaluation at the end of the slice — pull the session snapshot
    # once and score every assertion introduced so far.
    if transport == "in_process":
        snap_resp = in_process_client.get(f"/api/v1/sessions/{session_id}")
    else:
        snap_resp = http_client.get(f"/api/v1/sessions/{session_id}")
    snap_resp.raise_for_status()
    snapshot = _snapshot_from_session_response(snap_resp.json(), last_assistant_summary)
    passes, total = _evaluate_recall(snapshot, scenario.assertions, at_turn=scenario.turns[-1].index)
    if rows:
        rows[-1]["memory_recall_passes"] = passes
        rows[-1]["memory_recall_total"] = total
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def _ascii_curve(label_x: str, label_y: str, pairs: list[tuple[float, float]]) -> str:
    """Render a tiny markdown table of (x, y) pairs sorted by x."""
    if not pairs:
        return f"_(no data for {label_y} vs {label_x})_"
    rows = sorted(pairs, key=lambda p: p[0])
    lines = [
        f"| {label_x} | {label_y} |",
        f"| --- | --- |",
    ]
    for x, y in rows:
        lines.append(f"| {x:g} | {y:g} |")
    return "\n".join(lines)


def _generate_report(rows: list[dict[str, Any]], *, csv_path: Path, report_path: Path) -> None:
    if not rows:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            "# Stress report\n\n_(no rows in the input CSV; run the harness first)_\n",
            encoding="utf-8",
        )
        return

    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_profile[row["profile"]].append(row)

    lines: list[str] = ["# Stress report", ""]
    lines.append(f"Source CSV: `{csv_path}`. Rows: {len(rows)}.")
    lines.append("")

    # --- Summary table ---------------------------------------------------
    lines.append("## Summary by profile")
    lines.append("")
    lines.append(
        "| Profile | Rows | P50 latency (ms) | P95 latency (ms) | "
        "Total cost (USD) | Cache hit rate | Mean fact recall |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for profile in sorted(by_profile):
        prows = by_profile[profile]
        latencies = [float(r["latency_ms"]) for r in prows]
        costs = [float(r["cost_usd"]) for r in prows]
        cache_hits = sum(1 for r in prows if r["cache_hit_kind"] in {"exact", "semantic"})
        # Recall: average over the last turn of each scenario run
        final_turn_rows = [r for r in prows if r["memory_recall_total"]]
        if final_turn_rows:
            recalls = [
                r["memory_recall_passes"] / r["memory_recall_total"]
                for r in final_turn_rows
            ]
            mean_recall = statistics.mean(recalls)
        else:
            mean_recall = 0.0
        lines.append(
            f"| {profile} | {len(prows)} | "
            f"{_percentile(latencies, 0.5):.0f} | {_percentile(latencies, 0.95):.0f} | "
            f"{sum(costs):.4f} | {cache_hits}/{len(prows)} | {mean_recall:.2f} |"
        )
    lines.append("")

    # --- Curves ----------------------------------------------------------
    lines.append("## Curves")
    lines.append("")
    lines.append("### Latency vs tokens_in")
    lines.append("")
    pairs = [(float(r["tokens_in"]), float(r["latency_ms"])) for r in rows if r["tokens_in"]]
    lines.append(_ascii_curve("tokens_in", "latency_ms", pairs))
    lines.append("")

    lines.append("### Cumulative cost vs turn (per profile)")
    lines.append("")
    for profile in sorted(by_profile):
        lines.append(f"#### {profile}")
        lines.append("")
        cum: dict[int, float] = defaultdict(float)
        for r in by_profile[profile]:
            cum[r["turn_index"]] += float(r["cost_usd"])
        running = 0.0
        cum_pairs: list[tuple[float, float]] = []
        for turn in sorted(cum):
            running += cum[turn]
            cum_pairs.append((float(turn), round(running, 6)))
        lines.append(_ascii_curve("turn", "cumulative_cost_usd", cum_pairs))
        lines.append("")

    lines.append("### Recall vs turns slice")
    lines.append("")
    recall_pairs: list[tuple[float, float]] = []
    last_turn_rows = [r for r in rows if r["memory_recall_total"]]
    by_n: dict[int, list[float]] = defaultdict(list)
    for r in last_turn_rows:
        by_n[r["turn_index"]].append(r["memory_recall_passes"] / r["memory_recall_total"])
    for n in sorted(by_n):
        recall_pairs.append((float(n), round(statistics.mean(by_n[n]), 3)))
    lines.append(_ascii_curve("turns_in_slice", "mean_recall", recall_pairs))
    lines.append("")

    # --- Reading -----------------------------------------------------------
    lines.append("## Lectura — dónde empieza a romperse el CAG y por qué")
    lines.append("")
    lines.append(_reading_paragraph_one(rows))
    lines.append("")
    lines.append(_reading_paragraph_two(rows))
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _reading_paragraph_one(rows: list[dict[str, Any]]) -> str:
    """Latency / cost break-down based on the actual data."""
    truncation_rows = [
        r for r in rows
        if r["attachment_label"] == "100KB" and r["attachments_total_chars"]
    ]
    cap_truncated = any(
        r["attachments_total_chars"] <= 60_000 for r in truncation_rows
    )
    cap_clause = (
        "El cap de attachment (MAX_ATTACHMENT_CHARS=60.000) se está aplicando "
        "en el escenario 100KB, lo que mantiene acotado el coste por turno "
        "incluso cuando subes un PDF que extraería el doble."
        if cap_truncated
        else "El cap no se observa activo en esta corrida: o no se han generado "
             "filas con attachment 100KB o el extractor está devolviendo menos "
             "caracteres de los que el cap exigiría truncar."
    )
    latency_rows = [float(r["latency_ms"]) for r in rows if r["latency_ms"]]
    p95 = _percentile(latency_rows, 0.95) if latency_rows else 0.0
    return (
        f"La P95 de latencia observada es {p95:.0f} ms; el coste por turno escala con "
        f"el tamaño del attachment hasta el punto en que el cap entra en juego. "
        f"{cap_clause}"
    )


def _reading_paragraph_two(rows: list[dict[str, Any]]) -> str:
    """Memory recall takeaway by profile."""
    last_turns = [r for r in rows if r["memory_recall_total"]]
    by_profile: dict[str, list[float]] = defaultdict(list)
    for r in last_turns:
        by_profile[r["profile"]].append(
            r["memory_recall_passes"] / r["memory_recall_total"]
        )
    if not by_profile:
        return (
            "No hay rows con assertions evaluables (turn_recall_total=0 en todas). "
            "Ejecuta más repeats o slices más largos para llenar la curva de recall."
        )
    parts = [
        f"{profile}={statistics.mean(values):.2f}"
        for profile, values in sorted(by_profile.items())
    ]
    return (
        "Recall medio por perfil al final de cada slice: " + ", ".join(parts) + ". "
        "El perfil que se degrada antes es donde el summary acumulado deja de "
        "echo-back-ear los facts más antiguos — ahí es donde el ventaneo y la "
        "ausencia de anchors te está costando memoria."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_scenarios(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("at least one scenario is required")
    return parts


def _parse_sizes(raw: str) -> list[SyntheticAttachmentSpec]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    specs: list[SyntheticAttachmentSpec] = []
    for label in parts:
        # Accept both "0KB" and "0" — the latter is what the CLI example uses.
        normalised = label if label.upper().endswith("KB") else f"{label}KB"
        specs.append(spec_by_label(normalised))
    return specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=_parse_scenarios, default=["growing", "pivot", "contradiction"])
    parser.add_argument("--attachment-sizes", type=_parse_sizes, default=list(ATTACHMENT_SIZES))
    parser.add_argument("--turns", type=int, choices=SUPPORTED_TURN_LENGTHS, default=6)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--http", default=None, help="Base URL for HTTP transport.")
    parser.add_argument("--latency-budget-ms", type=int, default=15_000)
    parser.add_argument("--cost-budget-usd", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=Path("evals/stress/results.csv"))
    parser.add_argument("--report", type=Path, default=Path("evals/stress/REPORT.md"))
    args = parser.parse_args(argv)

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    if args.http:
        transport = "http"
        in_process_client: TestClient | None = None
        http_client = httpx.Client(base_url=args.http, timeout=300.0)
    else:
        transport = "in_process"
        in_process_client = TestClient(app)
        http_client = None

    rows: list[dict[str, Any]] = []

    print(
        f"Running scenarios={args.scenarios} sizes={[s.label for s in args.attachment_sizes]} "
        f"turns={args.turns} repeats={args.repeats} transport={transport}"
    )

    try:
        ctx_in = in_process_client if in_process_client is not None else _noop_cm()
        ctx_http = http_client if http_client is not None else _noop_cm()
        with ctx_in, ctx_http:
            for profile in args.scenarios:
                scenario = scenarios_for_length(profile, args.turns)
                for spec in args.attachment_sizes:
                    for repeat in range(args.repeats):
                        try:
                            new_rows = _run_scenario(
                                transport=transport,
                                in_process_client=in_process_client,
                                http_client=http_client,
                                scenario=scenario,
                                attachment_spec=spec,
                                repeat_index=repeat,
                                latency_budget_ms=args.latency_budget_ms,
                                cost_budget_usd=args.cost_budget_usd,
                            )
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"  {profile}/{spec.label}/repeat={repeat}: ERROR "
                                f"{type(exc).__name__}: {str(exc)[:200]}"
                            )
                            continue
                        rows.extend(new_rows)
                        print(
                            f"  {profile}/{spec.label}/repeat={repeat}: {len(new_rows)} turns"
                        )
    finally:
        if http_client is not None:
            http_client.close()

    _write_csv(rows, args.output)
    _generate_report(rows, csv_path=args.output, report_path=args.report)

    print(f"\nWrote {len(rows)} rows → {args.output}")
    print(f"Wrote report → {args.report}")
    return 0


@contextmanager
def _noop_cm() -> Iterator[None]:
    yield None


if __name__ == "__main__":
    raise SystemExit(main())
