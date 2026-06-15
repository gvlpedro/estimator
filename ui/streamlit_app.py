"""Streamlit chat UI for the estimator.

Acts as an HTTP client of the FastAPI service: POSTs to
``/api/v1/estimate/stream`` and renders the SSE chunks live with a placeholder.
The endpoint base URL is read from ``ESTIMATOR_API_BASE_URL`` (loaded from the
same ``.env`` as the API).
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000")
STREAM_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/estimate/stream"
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "claude-haiku-4-5-20251001")


st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")
st.caption(
    "Paste a meeting transcription. The answer streams token by token from the "
    "FastAPI service over Server-Sent Events."
)


def stream_estimation(transcription: str) -> Iterator[str]:
    """POST to the SSE endpoint and yield text chunks as they arrive.

    SSE serialises multi-line messages as several ``data:`` lines that the
    client joins with ``\\n``; a blank line terminates the message.
    """
    payload = {"transcription": transcription}
    with httpx.stream(
        "POST",
        STREAM_ENDPOINT,
        json=payload,
        timeout=httpx.Timeout(120.0, connect=10.0),
        headers={"Accept": "text/event-stream"},
    ) as response:
        response.raise_for_status()
        current_event = "token"
        data_lines: list[str] = []
        for raw_line in response.iter_lines():
            if raw_line == "":
                if data_lines:
                    text = "\n".join(data_lines)
                    data_lines = []
                    if current_event == "token":
                        yield text
                    elif current_event == "error":
                        yield f"\n\n[error] {text}"
                    elif current_event == "done":
                        return
                current_event = "token"
                continue
            if raw_line.startswith("event:"):
                current_event = raw_line[6:].strip()
            elif raw_line.startswith("data:"):
                data_lines.append(
                    raw_line[6:] if raw_line.startswith("data: ") else raw_line[5:]
                )


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_call" not in st.session_state:
    st.session_state.last_call = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Paste your meeting transcription here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        char_count = 0
        t0 = time.perf_counter()
        try:
            for chunk in stream_estimation(prompt):
                full_response += chunk
                char_count += len(chunk)
                placeholder.markdown(full_response + "▍")
            placeholder.markdown(full_response)
            st.session_state.last_call = {
                "model": PRIMARY_MODEL,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "input_chars": len(prompt),
                "output_chars": char_count,
            }
        except httpx.HTTPError as exc:
            full_response = (
                f"Could not reach the estimator at `{STREAM_ENDPOINT}`: {exc}"
            )
            placeholder.error(full_response)
            st.session_state.last_call = None

    st.session_state.messages.append({"role": "assistant", "content": full_response})


with st.sidebar:
    st.header("Service")
    st.code(STREAM_ENDPOINT, language="text")
    st.markdown(f"**Primary model:** `{PRIMARY_MODEL}`")
    st.markdown(f"**Fallback model:** `{FALLBACK_MODEL}`")
    st.markdown(f"**Cache TTL:** `{os.getenv('CACHE_TTL', '86400')}s`")

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.session_state.last_call = None
        st.rerun()

    if st.session_state.last_call:
        st.divider()
        st.header("Last call")
        m = st.session_state.last_call
        col_a, col_b = st.columns(2)
        col_a.metric("Latency", f"{m['latency_ms']} ms")
        col_b.metric("Output chars", m["output_chars"])
