"""Run the Session-13 LangGraph estimation pipeline end to end, with a full trace.

Needs OPENAI_API_KEY (uv does not load .env by itself -- pass --env-file .env
or export it yourself) and a reachable Postgres (the same one used for
embeddings; see docker-compose.yml). To get a shareable trace URL, set
LOGFIRE_TOKEN in .env to a write token from https://logfire.pydantic.dev
(project settings -> write tokens) -- or run ``uv run logfire auth`` once
interactively. Without either, this still runs fine, just without shipping
spans anywhere.

    uv run --env-file .env python servicio_ia/scripts/run_graph_s13.py \\
        servicio_ia/s12_agent_resources/sample_transcript_complex.txt \\
        --estimation-id demo-complex-001
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make `app` importable regardless of invocation cwd, same convention as
# scripts/run_agent_s12.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logfire  # noqa: E402
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: E402

from app.lgraph.build import build_graph  # noqa: E402
from app.lgraph.checkpointer import get_checkpointer_dsn  # noqa: E402


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("transcript", type=Path, help="Path to the transcript .txt file")
    parser.add_argument(
        "--estimation-id",
        required=True,
        help="Identifies this estimation; used as the checkpointer thread_id.",
    )
    args = parser.parse_args()

    logfire.configure(service_name="servicio_ia.run_graph_s13", send_to_logfire="if-token-present")
    logfire.instrument_asyncpg()
    logfire.instrument_httpx()

    transcript = args.transcript.read_text(encoding="utf-8")

    with logfire.span("run_graph_s13", estimation_id=args.estimation_id):
        async with AsyncPostgresSaver.from_conn_string(get_checkpointer_dsn()) as checkpointer:
            await checkpointer.setup()
            graph = build_graph(checkpointer)
            config = {"configurable": {"thread_id": args.estimation_id}}
            result = await graph.ainvoke(
                {"transcript": transcript, "estimation_id": args.estimation_id}, config
            )

    print(f"status: {result['status']}")
    print(f"requirements: {len(result.get('requirements', []))}")
    print(f"components: {[c['name'] for c in result.get('components', [])]}")
    print(f"budget_matches: {len(result.get('budget_matches', []))}")
    if result.get("errors"):
        print(f"errors: {result['errors']}")
    print(f"estimate: {result.get('estimate')}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
