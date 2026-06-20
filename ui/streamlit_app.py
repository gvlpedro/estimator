"""Streamlit form UI for the estimator.

POSTs an :class:`EstimationRequest`-shaped JSON to ``/api/v1/estimate`` and
renders the returned free-form text. The backend wraps the call in a
versioned Jinja template (``v1``), so the same form keeps working as the
prompt evolves.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000")
ESTIMATE_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/estimate"

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


st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")
st.caption("Describe the project and pick the format. The backend renders the v1 prompt and calls the LLM.")

with st.form("estimation_form"):
    description = st.text_area(
        "Project description",
        height=200,
        placeholder="What is being built? Who is it for? Any integrations, deadlines, constraints?",
        help="Between 20 and 2000 characters.",
    )
    col_a, col_b, col_c = st.columns(3)
    project_type = col_a.selectbox(
        "Project type",
        [v for v, _ in PROJECT_TYPES],
        format_func=lambda v: _label_of(PROJECT_TYPES, v),
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
    submitted = st.form_submit_button("Estimate")

if submitted:
    if not (20 <= len(description) <= 2000):
        st.error("Description must be between 20 and 2000 characters.")
        st.stop()

    payload = {
        "description": description,
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
    }

    with st.spinner("Calling the estimator…"):
        try:
            response = httpx.post(
                ESTIMATE_ENDPOINT,
                json=payload,
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            st.error(f"Backend returned {exc.response.status_code}: {exc.response.text}")
            st.stop()
        except httpx.HTTPError as exc:
            st.error(f"Could not reach the estimator at `{ESTIMATE_ENDPOINT}`: {exc}")
            st.stop()

    body = response.json()
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


with st.sidebar:
    st.header("Service")
    st.code(ESTIMATE_ENDPOINT, language="text")
    st.markdown(f"**Primary model:** `{os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')}`")
    st.markdown(f"**Fallback model:** `{os.getenv('FALLBACK_MODEL', 'claude-haiku-4-5-20251001')}`")
