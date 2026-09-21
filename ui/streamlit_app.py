"""Streamlit conversational UI for the estimator.

Session lifecycle:

1. On first render we ``POST /api/v1/sessions`` and stash the ``session_id`` in
   ``st.session_state`` so it survives reruns.
2. Each estimation goes to ``POST /api/v1/sessions/{id}/estimate`` as
   ``multipart/form-data`` with ``transcript`` and optional ``attachments``.
3. After each estimation we re-fetch ``GET /api/v1/sessions/{id}`` to refresh
   the sidebar — ``project_metadata`` (memory) and recent messages (history)
   are shown in separate panels so the alumnae can see the split.
4. "Nueva conversacion" creates a fresh session and resets local state.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000").rstrip("/")
SESSIONS_ENDPOINT = f"{API_BASE_URL}/api/v1/sessions"


PROJECT_TYPES = [
    ("mobile_app", "Mobile app"),
    ("web_saas", "Web SaaS"),
    ("internal_tool", "Internal tool"),
    ("data_pipeline", "Data pipeline"),
]
DETAIL_LEVELS = [
    ("summary", "Summary"),
    ("medium", "Medium"),
    ("detailed", "Detailed"),
]
OUTPUT_FORMATS = [
    ("phases_table", "Phases table"),
    ("line_items", "Line items"),
    ("narrative", "Narrative"),
]


def _label_of(options: list[tuple[str, str]], value: str) -> str:
    return next(label for v, label in options if v == value)


def _create_session() -> str:
    """POST /sessions and return the new session_id."""
    response = httpx.post(SESSIONS_ENDPOINT, timeout=httpx.Timeout(10.0))
    response.raise_for_status()
    return response.json()["session_id"]


def _format_backend_error(response: httpx.Response) -> str:
    """Human-readable rendering of a FastAPI error response.

    Input-guardrail violations are 400s whose ``detail`` is a dict
    ``{"reason": ..., "message": ...}`` — surface the message and the reason
    instead of dumping the raw JSON to the user.
    """
    try:
        body = response.json()
    except ValueError:
        return f"Backend returned {response.status_code}: {response.text[:400]}"
    detail = body.get("detail") if isinstance(body, dict) else body
    if isinstance(detail, dict) and "message" in detail:
        reason = detail.get("reason")
        prefix = f"{response.status_code} ({reason})" if reason else str(response.status_code)
        return f"{prefix}: {detail['message']}"
    return f"Backend returned {response.status_code}: {detail}"


def _fetch_session_state(session_id: str) -> dict[str, Any] | None:
    """GET /sessions/{id}. Returns ``None`` if the server lost the session."""
    try:
        response = httpx.get(f"{SESSIONS_ENDPOINT}/{session_id}", timeout=httpx.Timeout(10.0))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def _reset_session() -> None:
    """Drop local state and provision a brand-new server-side session."""
    for key in ("session_id", "last_result", "session_state"):
        st.session_state.pop(key, None)
    try:
        st.session_state.session_id = _create_session()
    except httpx.HTTPError as exc:
        st.session_state.session_error = (
            f"Could not reach the estimator at `{SESSIONS_ENDPOINT}`: {exc}"
        )


# ---------------------------------------------------------------------------
# Page setup + session bootstrap
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Software Estimator", page_icon="📊", layout="wide")
st.title("Software Estimator")
st.caption(
    "Conversational estimator. Drop a transcript and optional PDFs; the backend "
    "renders the prompt, runs the LLM and keeps memory of the project."
)

if "session_id" not in st.session_state:
    try:
        st.session_state.session_id = _create_session()
    except httpx.HTTPError as exc:
        st.error(f"Could not reach the estimator at `{SESSIONS_ENDPOINT}`: {exc}")
        st.stop()

if "session_error" in st.session_state:
    st.error(st.session_state.pop("session_error"))
    st.stop()

session_id: str = st.session_state.session_id

# ---------------------------------------------------------------------------
# Sidebar: memory (project_metadata) + history + reset
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Session")
    st.button(
        "🆕 Nueva conversacion",
        on_click=_reset_session,
        use_container_width=True,
        help="Crea una nueva sesion en el servidor y resetea el estado local.",
    )
    st.caption(f"`{session_id}`")

    snapshot = _fetch_session_state(session_id) or {}
    metadata = snapshot.get("project_metadata") or {}
    messages = snapshot.get("messages") or []

    with st.expander("🧠 Project metadata (memory)", expanded=True):
        st.caption("Hechos acumulados por el extractor LLM tras cada turno.")
        if not any(metadata.values()):
            st.write("_(empty — no facts known yet)_")
        else:
            st.markdown(f"**project_name:** {metadata.get('project_name') or '_unknown_'}")
            st.markdown(
                f"**assumed_team_size:** {metadata.get('assumed_team_size') or '_unknown_'}"
            )
            techs = metadata.get("mentioned_technologies") or []
            st.markdown(
                "**mentioned_technologies:** "
                + (", ".join(f"`{t}`" for t in techs) if techs else "_none_")
            )
            st.markdown(f"**agreed_scope:** {metadata.get('agreed_scope') or '_unknown_'}")

    with st.expander(
        f"💬 Historial reciente ({sum(1 for m in messages if m['role'] != 'system')} msgs)",
        expanded=False,
    ):
        st.caption(
            "Ventana deslizante de pares user/assistant (system prompt anclado fuera de cuenta)."
        )
        non_system = [m for m in messages if m["role"] != "system"]
        if not non_system:
            st.write("_(no turns yet)_")
        for msg in non_system:
            with st.chat_message(msg["role"]):
                content = msg["content"]
                if len(content) > 600:
                    content = content[:600] + "…"
                st.write(content)

    st.divider()
    st.header("Service")
    st.code(API_BASE_URL, language="text")
    st.markdown(f"**Primary model:** `{os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')}`")
    st.markdown(
        f"**Fallback model:** `{os.getenv('FALLBACK_MODEL', 'claude-haiku-4-5-20251001')}`"
    )

# ---------------------------------------------------------------------------
# Main form: transcript + attachments + knobs
# ---------------------------------------------------------------------------

with st.form("session_estimate_form", clear_on_submit=False):
    transcript = st.text_area(
        "Transcript",
        height=220,
        placeholder=(
            "Pega aqui la transcripcion de la reunion o describe el proyecto. "
            "Si subes PDFs o DOCX, el servidor extrae el texto y lo concatena."
        ),
        help="Entre 20 y 80 000 caracteres.",
    )
    attachments = st.file_uploader(
        "Adjuntos (PDF, DOCX, ...)",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
        help="Se parsean en el servidor (pypdf / python-docx). Nunca se envian al proveedor LLM.",
    )

    col_a, col_b, col_c = st.columns(3)
    project_type = col_a.selectbox(
        "Project type",
        [v for v, _ in PROJECT_TYPES],
        format_func=lambda v: _label_of(PROJECT_TYPES, v),
        index=1,
    )
    detail_level = col_b.selectbox(
        "Detail level",
        [v for v, _ in DETAIL_LEVELS],
        format_func=lambda v: _label_of(DETAIL_LEVELS, v),
        index=1,
    )
    output_format = col_c.selectbox(
        "Output format",
        [v for v, _ in OUTPUT_FORMATS],
        format_func=lambda v: _label_of(OUTPUT_FORMATS, v),
    )
    submitted = st.form_submit_button("Estimate", use_container_width=True)

if submitted:
    if not (20 <= len(transcript) <= 80_000):
        st.error("Transcript must be between 20 and 80 000 characters.")
        st.stop()

    form_data = {
        "transcript": transcript,
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
    }
    files = [
        ("attachments", (f.name, f.getvalue(), f.type or "application/octet-stream"))
        for f in (attachments or [])
    ]

    estimate_url = f"{SESSIONS_ENDPOINT}/{session_id}/estimate"
    response: httpx.Response | None = None
    network_error: str | None = None
    # Spinner is intentionally tight around the network call only — rendering
    # errors inside the with-block leaves the spinner visible next to the
    # message (Streamlit only tears it down when the context exits).
    with st.spinner("Calling the estimator…"):
        try:
            response = httpx.post(
                estimate_url,
                data=form_data,
                files=files or None,
                timeout=httpx.Timeout(180.0, connect=10.0),
            )
        except httpx.HTTPError as exc:
            network_error = f"Could not reach the estimator at `{estimate_url}`: {exc}"

    if network_error:
        st.error(network_error)
        st.stop()

    assert response is not None  # guaranteed by the network_error short-circuit above
    if response.status_code == 404:
        st.warning(
            "La sesion se perdio en el servidor (probablemente reinicio). "
            "Crea una nueva conversacion para continuar."
        )
        st.stop()
    if response.is_error:
        st.error(_format_backend_error(response))
        st.stop()

    st.session_state.last_result = response.json()
    # Force a rerun so the sidebar refetches the updated session state.
    st.rerun()

# ---------------------------------------------------------------------------
# Render the latest result (survives reruns)
# ---------------------------------------------------------------------------

body = st.session_state.get("last_result")
if body:
    result = body.get("result") or {}
    st.caption(
        f"Prompt version: `{body.get('prompt_version', '?')}` · "
        f"cached: `{body.get('cached', False)}` · "
        f"confidence: `{result.get('confidence_pct', '?')}%`"
    )

    st.subheader("Summary")
    st.markdown(result.get("summary", "_(no summary)_"))

    st.subheader("Phases")
    phases = result.get("phases") or []
    if phases:
        st.dataframe(
            [
                {
                    "Phase": p["name"],
                    "Weeks": p["duration_weeks"],
                    "Cost (EUR)": p["cost_eur"],
                    "Detail": p["summary"],
                }
                for p in phases
            ],
            hide_index=True,
            use_container_width=True,
        )

    col_t1, col_t2 = st.columns(2)
    col_t1.metric("Total cost (EUR)", f"{result.get('total_cost_eur', 0):,}")
    col_t2.metric("Total duration (weeks)", result.get("total_duration_weeks", 0))
