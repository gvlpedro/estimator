"""Synthetic PDF generator for the attachment-size stress axis.

Five calibrated payload sizes drive the same estimation prompt:

    0 KB    no attachment (baseline)
    5 KB    ~2 pages of plain text
    20 KB   ~8 pages
    50 KB   ~20 pages
    100 KB  past the MAX_ATTACHMENT_CHARS cap → truncated, measures the cap

PDF byte size does not map cleanly to extracted-text length (PDF overhead,
compression, fonts). We calibrate by *extracted character count* targets and
record the realised byte size on the side. The extractor (and the cap that
the router enforces) sees character counts, so this is the honest axis.

A known **marker phrase** appears verbatim near the start of every payload.
The runner uses it to compute a content-recall signal: does the assistant
summary echo back something that was clearly only in the attachment?

Generation is deterministic — same target chars → same bytes — so re-running
the stress suite does not introduce noise on this axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


@dataclass(frozen=True)
class SyntheticAttachmentSpec:
    """One row in the attachment-size axis."""

    label: str               # e.g. "0KB", "5KB"
    target_chars: int        # extracted characters (post-PDF parsing)
    description: str         # one-line summary for the report


@dataclass(frozen=True)
class SyntheticAttachment:
    """Realised attachment after generation: bytes + introspection metadata."""

    spec: SyntheticAttachmentSpec
    filename: str
    data: bytes
    marker_phrase: str       # canary the runner greps for in the response


# The five canonical sizes — labelled by approximate file size so the report
# matches the convention used in the request. ``target_chars`` is the dial we
# actually control. The 100 KB row exceeds MAX_ATTACHMENT_CHARS=60_000 and is
# expected to be clipped by the extractor.
ATTACHMENT_SIZES: tuple[SyntheticAttachmentSpec, ...] = (
    SyntheticAttachmentSpec(label="0KB",   target_chars=0,      description="baseline (no attachment)"),
    SyntheticAttachmentSpec(label="5KB",   target_chars=2_500,  description="~2 pages of plain text"),
    SyntheticAttachmentSpec(label="20KB",  target_chars=10_000, description="~8 pages"),
    SyntheticAttachmentSpec(label="50KB",  target_chars=25_000, description="~20 pages"),
    SyntheticAttachmentSpec(label="100KB", target_chars=80_000, description="past the cap — truncates"),
)


# A stable marker phrase placed near the head of every non-empty attachment.
# Chosen to be unmistakably attachment-origin: nobody would put this in a
# transcript by accident, so a hit in the response summary is a real recall
# signal rather than vocabulary overlap.
MARKER_PHRASE = "INVOICE_SCAN_REF_QX7T9-ATLAS-2026-Q2"


def _filler_paragraph(seed: int) -> str:
    """Return a meaningful-looking paragraph deterministically derived from
    ``seed``. We avoid Lorem Ipsum because the LLM has been trained to ignore
    it; project-flavoured prose makes the recall metric meaningful."""
    sentences = [
        f"Section {seed}.1 — Stakeholders reaffirmed the scope agreed during "
        f"the kickoff workshop and added a calibration note (#{seed}).",
        f"Section {seed}.2 — Open risks include vendor onboarding delays and a "
        f"dependency on the legal review tracked as RISK-{seed:03d}.",
        f"Section {seed}.3 — The team estimates the work in this section at "
        f"{1 + (seed % 4)} engineer-weeks across the agreed phases.",
        f"Section {seed}.4 — Integration considerations were reviewed against "
        f"the architectural baseline document version {seed % 5 + 1}.0.",
        f"Section {seed}.5 — Acceptance criteria require a regression sweep and "
        f"sign-off from the product owner before promoting to production.",
    ]
    return " ".join(sentences)


def _compose_body(target_chars: int) -> str:
    """Produce a string of exactly (or slightly above) ``target_chars`` chars,
    composed of project-flavoured prose so the extractor has something
    realistic to parse and the recall metric has signal to find."""
    if target_chars <= 0:
        return ""

    header = (
        f"Project annex generated for stress testing. "
        f"Marker: {MARKER_PHRASE}. "
        f"Generated at {datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()}.\n\n"
    )
    parts: list[str] = [header]
    seed = 0
    current = len(header)
    while current < target_chars:
        paragraph = _filler_paragraph(seed) + "\n\n"
        parts.append(paragraph)
        current += len(paragraph)
        seed += 1
    return "".join(parts)[:target_chars] if target_chars > len(header) else "".join(parts)


def generate_attachment(spec: SyntheticAttachmentSpec) -> SyntheticAttachment | None:
    """Render one PDF for the given spec.

    Returns ``None`` for the baseline (``target_chars == 0``) so the runner
    can distinguish "no attachment" from "empty attachment". For non-zero
    targets, the function returns the bytes plus the marker phrase the
    runner uses to compute attachment-recall.
    """
    if spec.target_chars <= 0:
        return None

    body = _compose_body(spec.target_chars)
    buffer = BytesIO()
    # ReportLab compresses streams by default, so the realised PDF byte size
    # is well below the label (e.g. 80 K chars → ~26 KB on disk). The label
    # is nominal — the real axis the cap reacts to is *extracted* characters.
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title=f"Stress annex {spec.label}",
        author="estimator-stress",
    )
    styles = getSampleStyleSheet()
    body_style = styles["BodyText"]
    flowables: list = []
    # ReportLab Paragraph expects HTML-ish markup; we paragraph-split the
    # body so very long single paragraphs don't blow past the layout engine.
    for chunk in body.split("\n\n"):
        if not chunk.strip():
            continue
        # Replace anything ReportLab parses as a tag.
        safe = chunk.replace("<", "&lt;").replace(">", "&gt;")
        flowables.append(Paragraph(safe, body_style))
        flowables.append(Spacer(1, 6))
    doc.build(flowables)
    return SyntheticAttachment(
        spec=spec,
        filename=f"stress_annex_{spec.label}.pdf",
        data=buffer.getvalue(),
        marker_phrase=MARKER_PHRASE,
    )


def spec_by_label(label: str) -> SyntheticAttachmentSpec:
    """Lookup helper: ``"5KB"`` → the 2_500-char spec."""
    for spec in ATTACHMENT_SIZES:
        if spec.label.upper() == label.upper():
            return spec
    raise ValueError(
        f"unknown attachment size {label!r}; choose from {[s.label for s in ATTACHMENT_SIZES]}"
    )
