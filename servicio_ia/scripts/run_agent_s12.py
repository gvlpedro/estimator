"""Run the Session-12 manual agentic loop over a discovery-meeting transcript.

Needs OPENAI_API_KEY, which uv does NOT load from .env by itself -- pass
--env-file .env (from the repo root) or export the variable yourself first.

# Simple transcript
    uv run --env-file .env python servicio_ia/scripts/run_agent_s12.py \\
        servicio_ia/s12_agent_resources/sample_transcript_simple.txt \\
        --model gpt-5-mini --effort minimal --stub

# Complex transcript
    uv run --env-file .env python servicio_ia/scripts/run_agent_s12.py \\
        servicio_ia/s12_agent_resources/sample_transcript_complex.txt \\
        --model gpt-5 --effort medium --stub \\
        --out servicio_ia/s12_agent_resources/example_trace_complex.txt

"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make `app` importable regardless of invocation cwd (repo root, servicio_ia/,
# or a container whose working_dir is already servicio_ia/): this script's
# parent (servicio_ia/) is where the `app` package lives.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.loop import ToolContext, run_agent  # noqa: E402
from app.agents.schemas import format_trace  # noqa: E402
from app.agents.stub_retrieval import stub_search_budgets  # noqa: E402


def _stub_context() -> ToolContext:
    async def _search(args: dict) -> dict:
        return stub_search_budgets(args)

    return ToolContext(search_budgets=_search)


async def _real_context() -> tuple[ToolContext, "object"]:
    from app.agents.tools import search_budgets
    from app.dependencies import get_embedder, get_reranker
    from app.storage.db import session_factory

    session = session_factory()
    embedder = get_embedder()
    reranker = get_reranker()

    async def _search(args: dict) -> dict:
        return await search_budgets(args, session=session, embedder=embedder, reranker=reranker)

    return ToolContext(search_budgets=_search), session


async def _main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("transcript", type=Path, help="Path to the transcript .txt file")
    parser.add_argument("--model", default="gpt-5", help="Responses API model (default: gpt-5)")
    parser.add_argument(
        "--effort",
        default="medium",
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort (default: medium)",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use the canned in-memory corpus instead of the real retrieval pipeline",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Also write the trace + final estimate to this file"
    )
    parser.add_argument("--max-iterations", type=int, default=12)
    args = parser.parse_args()

    transcript = args.transcript.read_text(encoding="utf-8")

    session = None
    if args.stub:
        tool_ctx = _stub_context()
    else:
        tool_ctx, session = await _real_context()

    try:
        result = await run_agent(
            transcript,
            model=args.model,
            effort=args.effort,
            tool_ctx=tool_ctx,
            max_iterations=args.max_iterations,
        )
    finally:
        if session is not None:
            await session.close()

    trace_text = format_trace(result.trace)
    report = f"{trace_text}\n\n--- FINAL ESTIMATE ---\n{result.final_response}\n"
    print(report)

    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"Trace written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
