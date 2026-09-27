"""Run the Session-14 multi-agent estimation graph end to end, with a full trace.

Drives the supervisor/specialist graph on a transcript and, if the
coherence validator's confidence trips the human-review gate, resumes the
paused run with a simulated decision -- so a single invocation can produce a
trace that covers both Level 1 (routing) and Level 2 (pause + resume).

Needs OPENAI_API_KEY (uv does not load .env by itself -- pass --env-file
.env or export it yourself) and a reachable Postgres (the same one used for
embeddings; see docker-compose.yml). To get a shareable trace URL, set
LOGFIRE_TOKEN in .env to a write token from https://logfire.pydantic.dev --
without it, this still runs fine, just without shipping spans anywhere.

    # Normal run (pauses only if confidence is actually low)
    uv run --env-file .env python servicio_ia/scripts/run_multiagent_s14.py \\
        servicio_ia/s12_agent_resources/sample_transcript_complex.txt \\
        --estimation-id demo-multiagent-001

    # Force the pause to demo Level 2 even when the estimate is confident
    # (CONFIDENCE_THRESHOLD is meant to be configurable -- this just cranks
    # it above the maximum possible confidence of 1.0 for the demo)
    uv run --env-file .env python servicio_ia/scripts/run_multiagent_s14.py \\
        servicio_ia/s12_agent_resources/sample_transcript_complex.txt \\
        --estimation-id demo-multiagent-002 --confidence-threshold 1.1 --decision adjusted
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Make `app` importable regardless of invocation cwd, same convention as
# scripts/run_agent_s12.py and scripts/run_graph_s13.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def _main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("transcript", type=Path, help="Path to the transcript .txt file")
    parser.add_argument(
        "--estimation-id",
        required=True,
        help="Identifies this estimation; used as the checkpointer thread_id.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Override CONFIDENCE_THRESHOLD for this run (e.g. 1.1 forces a pause).",
    )
    parser.add_argument(
        "--decision",
        choices=["approved", "adjusted", "rejected"],
        default="approved",
        help="Decision to resume with if the graph pauses (default: approved).",
    )
    args = parser.parse_args()

    # Must happen before importing app.lgraph.nodes, which reads
    # CONFIDENCE_THRESHOLD at import time.
    if args.confidence_threshold is not None:
        os.environ["CONFIDENCE_THRESHOLD"] = str(args.confidence_threshold)

    import logfire
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.types import Command

    from app.lgraph.build import build_graph
    from app.lgraph.checkpointer import get_checkpointer_dsn

    logfire.configure(
        service_name="servicio_ia.run_multiagent_s14", send_to_logfire="if-token-present"
    )
    logfire.instrument_asyncpg()
    logfire.instrument_httpx()

    transcript = args.transcript.read_text(encoding="utf-8")

    with logfire.span("run_multiagent_s14", estimation_id=args.estimation_id):
        async with AsyncPostgresSaver.from_conn_string(get_checkpointer_dsn()) as checkpointer:
            await checkpointer.setup()
            graph = build_graph(checkpointer)
            config = {"configurable": {"thread_id": args.estimation_id}}

            result = await graph.ainvoke(
                {"transcript": transcript, "estimation_id": args.estimation_id}, config
            )

            if "__interrupt__" in result:
                review = result["__interrupt__"][0].value
                print(
                    f"PAUSED: {review['reason']} "
                    f"(confidence={review['confidence']}) -- resuming with "
                    f"decision={args.decision!r}"
                )
                result = await graph.ainvoke(
                    Command(
                        resume={
                            "outcome": args.decision,
                            "notes": "resumed by run_multiagent_s14.py",
                        }
                    ),
                    config,
                )

    print(f"status: {result['status']}")
    print(f"agents: {[c['agent'] for c in result.get('agent_contributions', [])]}")
    if result.get("human_decision"):
        print(f"human_decision: {result['human_decision']}")
    print(f"estimate: {result.get('estimate')}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
