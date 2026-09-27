#!/usr/bin/env python3
"""Self-contained monitoring panel: reads structured logs, prints/exports
latency, cost and error-rate signals for the AI service.

No external account needed (no Logfire/Langsmith project required) -- reads
the ``request_completed`` events emitted once per HTTP request by the
middleware in ``app/main.py`` (Session 16; before that, ``app/observability``
was called manually inside ``/estimate`` only -- see doc/evaluation.md for
why the middleware is what makes the error rate trustworthy).

The events are only valid JSON when the service logs in JSON mode
(``configure_logging()`` in ``app/main.py`` only picks ``JSONRenderer`` when
``APP_ENV=production`` -- in ``development`` it uses a human-readable
console renderer that this script cannot parse). See doc/evaluation.md.

Usage, piping ``docker compose logs`` straight in (each line is prefixed
with the container name, which this script strips):

    docker compose logs ai-service --no-color | uv run python servicio_ia/eval/dashboard.py

Or against a saved log file, optionally also writing a static HTML page and
the raw aggregates as JSON:

    docker compose logs ai-service --no-color > /tmp/ai-service.log
    uv run python servicio_ia/eval/dashboard.py /tmp/ai-service.log \\
        --html servicio_ia/eval/reports/dashboard.html \\
        --json servicio_ia/eval/reports/dashboard.json
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

EVENT_NAME = "request_completed"


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _strip_docker_prefix(line: str) -> str:
    """``docker compose logs`` prefixes each line with ``<container>  | ``."""
    marker = " | "
    idx = line.find(marker)
    return line[idx + len(marker) :] if idx != -1 else line


def parse_metrics_events(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Extract every well-formed ``request_completed`` JSON event from log lines.

    Silently skips lines that aren't JSON (human-readable log lines, docker
    noise, blank lines) and JSON lines that aren't this specific event --
    this is a best-effort scrape of a live log stream, not a strict parser.
    """
    events = []
    for raw_line in lines:
        line = _strip_docker_prefix(raw_line).strip()
        if not line or not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == EVENT_NAME:
            events.append(payload)
    return events


# --------------------------------------------------------------------------- #
# Aggregating
# --------------------------------------------------------------------------- #


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile -- no numpy dependency for a handful of values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1)))))
    return ordered[index]


def _is_error(event: dict[str, Any]) -> bool:
    """Any non-2xx HTTP status, or a missing one, counts as an error.

    ``status`` is the HTTP status code the middleware recorded (200, 401,
    500...) -- not a business outcome. ``needs_review``/``awaiting_human_review``
    are correct, non-error behavior that still returns 200, so they never
    inflate this rate; a request that raised before producing a response has
    no status at all and counts as an error.
    """
    status = int(event.get("status", 0) or 0)
    return status >= 400 or status == 0


def _row(events: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(e["latency_ms"]) for e in events if "latency_ms" in e]
    costs = [float(e["cost_usd"]) for e in events if "cost_usd" in e]
    errors = [e for e in events if _is_error(e)]
    return {
        "request_count": len(events),
        "errors": len(errors),
        "error_rate": round(len(errors) / len(events), 3) if events else 0.0,
        "latency_ms_mean": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "latency_ms_p95": round(_percentile(latencies, 95), 1) if latencies else 0.0,
        "cost_usd_mean": round(sum(costs) / len(costs), 6) if costs else 0.0,
        "cost_usd_total": round(sum(costs), 6) if costs else 0.0,
        "tokens_total": sum(int(e.get("total_tokens", 0) or 0) for e in events),
        "llm_calls_total": sum(int(e.get("llm_calls", 0) or 0) for e in events),
    }


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate events overall and broken down by path.

    Kept as a plain function (not folded into rendering) so tests can assert
    on numbers directly. ``status`` here is an HTTP status code, not a
    business-outcome string -- see ``_is_error``.
    """
    if not events:
        return {
            "request_count": 0,
            "errors": 0,
            "latency_ms_mean": None,
            "latency_ms_p95": None,
            "cost_usd_mean": None,
            "cost_usd_total": None,
            "error_rate": None,
            "tokens_total": 0,
            "llm_calls_total": 0,
            "by_path": {},
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sparkline": [],
        }

    overall = _row(events)
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_path[str(event.get("path", "?"))].append(event)
    # Busiest first: the row that dominates the overall number is the one
    # worth reading first.
    overall["by_path"] = {
        path: _row(rows) for path, rows in sorted(by_path.items(), key=lambda kv: -len(kv[1]))
    }
    overall["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    overall["sparkline"] = [float(e.get("latency_ms", 0) or 0) for e in events][-60:]
    return overall


# --------------------------------------------------------------------------- #
# Rendering -- terminal
# --------------------------------------------------------------------------- #


def render(summary: dict[str, Any]) -> str:
    if summary["request_count"] == 0:
        return f"No {EVENT_NAME} events found in the given log input."

    lines = [
        "== servicio_ia dashboard ==",
        f"requests observed      : {summary['request_count']}",
        f"latency mean / p95 (ms): {summary['latency_ms_mean']} / {summary['latency_ms_p95']}",
        f"cost mean / total (usd): {summary['cost_usd_mean']} / {summary['cost_usd_total']}",
        f"error rate             : {summary['error_rate']:.1%}",
    ]

    by_path = summary.get("by_path") or {}
    if by_path:
        lines.append("")
        lines.append(
            f"{'path':30s} {'reqs':>5s} {'err':>6s} {'mean ms':>9s} {'p95 ms':>9s} {'$/req':>9s}"
        )
        for path, row in by_path.items():
            lines.append(
                f"{path[:30]:30s} {row['request_count']:5d} {row['error_rate'] * 100:5.0f}% "
                f"{row['latency_ms_mean']:9.1f} {row['latency_ms_p95']:9.1f} "
                f"${row['cost_usd_mean']:8.4f}"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Rendering -- self-contained HTML
# --------------------------------------------------------------------------- #


def _sparkline_svg(values: list[float]) -> str:
    if not values:
        return ""
    width, height = 640, 90
    peak = max(values) or 1.0
    step = width / max(1, len(values))
    bars = "".join(
        f'<rect x="{i * step:.1f}" y="{height - (v / peak) * height:.1f}" '
        f'width="{max(1.0, step - 1.5):.1f}" height="{(v / peak) * height:.1f}" rx="1"/>'
        for i, v in enumerate(values)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
        f'class="spark" role="img" aria-label="Latency of the last {len(values)} requests">'
        f"{bars}</svg>"
    )


def render_html(summary: dict[str, Any]) -> str:
    cards = [
        ("Requests", f"{summary['request_count']}", ""),
        ("Error rate", f"{(summary['error_rate'] or 0) * 100:.1f}%", f"{summary['errors']} non-2xx"),
        (
            "Latency p95",
            f"{(summary['latency_ms_p95'] or 0) / 1000:.2f}s",
            f"mean {(summary['latency_ms_mean'] or 0) / 1000:.2f}s",
        ),
        (
            "Cost / request",
            f"${summary['cost_usd_mean'] or 0:.4f}",
            f"${summary['cost_usd_total'] or 0:.4f} total",
        ),
    ]
    card_html = "".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f'<div class="hint">{html.escape(hint)}</div></div>'
        for label, value, hint in cards
    )
    rows = "".join(
        f"<tr><td class='path'>{html.escape(path)}</td>"
        f"<td>{r['request_count']}</td>"
        f"<td class='{'bad' if r['error_rate'] else 'ok'}'>{r['error_rate'] * 100:.0f}%</td>"
        f"<td>{r['latency_ms_mean'] / 1000:.2f}s</td>"
        f"<td>{r['latency_ms_p95'] / 1000:.2f}s</td>"
        f"<td>${r['cost_usd_mean']:.4f}</td>"
        f"<td>{r['tokens_total']:,}</td></tr>"
        for path, r in (summary.get("by_path") or {}).items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>servicio_ia -- production signals</title>
<style>
  :root {{ --bg:#fbfaf9; --fg:#1a1a19; --muted:#6b6b68; --line:#e5e3e0;
           --card:#fff; --accent:#b45309; --bad:#b91c1c; --ok:#15803d; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#141413; --fg:#f0eee6; --muted:#9a9a95; --line:#2b2b28;
             --card:#1c1c1a; --accent:#d97706; --bad:#f87171; --ok:#4ade80; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:2.5rem 1.5rem; background:var(--bg); color:var(--fg);
         font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",sans-serif; }}
  main {{ max-width:900px; margin:0 auto; }}
  h1 {{ font-size:1.4rem; margin:0 0 .25rem; letter-spacing:-.01em; }}
  .meta {{ color:var(--muted); font-size:.85rem; margin-bottom:2rem; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
            gap:.85rem; margin-bottom:2rem; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:1rem; }}
  .label {{ color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; }}
  .value {{ font-size:1.75rem; font-weight:600; margin:.3rem 0 .1rem;
            font-variant-numeric:tabular-nums; }}
  .hint {{ color:var(--muted); font-size:.78rem; }}
  .spark {{ width:100%; height:90px; fill:var(--accent); opacity:.85;
            background:var(--card); border:1px solid var(--line); border-radius:10px; }}
  h2 {{ font-size:.8rem; text-transform:uppercase; letter-spacing:.06em;
        color:var(--muted); margin:2rem 0 .6rem; }}
  .scroll {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; font-size:.88rem; }}
  th,td {{ text-align:right; padding:.5rem .7rem; border-bottom:1px solid var(--line);
           font-variant-numeric:tabular-nums; white-space:nowrap; }}
  th {{ color:var(--muted); font-weight:500; font-size:.75rem; text-transform:uppercase; }}
  th:first-child, td.path {{ text-align:left; font-family:ui-monospace,monospace; }}
  .bad {{ color:var(--bad); }} .ok {{ color:var(--ok); }}
  footer {{ color:var(--muted); font-size:.78rem; margin-top:2.5rem;
            border-top:1px solid var(--line); padding-top:1rem; }}
</style></head>
<body><main>
  <h1>servicio_ia -- production signals</h1>
  <div class="meta">Generated {html.escape(summary["generated_at"])} · from <code>{EVENT_NAME}</code>
  events in the structured logs</div>
  <div class="cards">{card_html}</div>
  <h2>Latency, last {len(summary["sparkline"])} requests</h2>
  {_sparkline_svg(summary["sparkline"])}
  <h2>By endpoint</h2>
  <div class="scroll"><table>
    <thead><tr><th>path</th><th>reqs</th><th>errors</th><th>mean</th><th>p95</th>
    <th>$/req</th><th>tokens</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
  <footer>The health probe (<code>/health</code>) is excluded: called every 30s, it would
  dominate every number here. See doc/evaluation.md.</footer>
</main></body></html>"""


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="servicio_ia self-contained metrics dashboard")
    parser.add_argument("log_file", nargs="?", default=None, help="Log file (default: stdin)")
    parser.add_argument("--html", default=None, help="Also write a self-contained HTML dashboard")
    parser.add_argument("--json", default=None, help="Also write the aggregates as JSON")
    args = parser.parse_args(argv)

    if args.log_file:
        with open(args.log_file, encoding="utf-8") as f:
            events = parse_metrics_events(f)
    else:
        events = parse_metrics_events(sys.stdin)

    summary = summarize(events)
    print(render(summary))

    if args.html:
        out = Path(args.html)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_html(summary), encoding="utf-8")
        print(f"\ndashboard -> {out}")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"aggregates -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
