"""Local text extraction for uploaded attachments.

Strategy: parse PDFs with ``pypdf``, DOCX with ``python-docx``, and decode
plain-text formats directly. The extracted text is concatenated to the user
``transcript`` with an explicit separator before the prompt is built. No
attachment bytes ever reach the LLM provider — only the extracted text.

Why local extraction over a provider Files API
----------------------------------------------
* **Independence from the LLM vendor.** Parsing runs in our process, so we
  can switch between OpenAI, Anthropic, or self-hosted models without
  touching this layer. A vendor Files API ties retention, pricing and the
  attachment lifecycle to a single provider.
* **No second token bill per attachment per call.** A Files API charges
  tokens for the whole document every time it is referenced. We extract
  once and reuse the text as any other input.
* **Deterministic, auditable text.** The exact bytes that reach the prompt
  are visible, loggable and cacheable. A Files API hides parsing behind a
  black box we cannot inspect.
* **Sets up the RAG module (módulo 3).** Once extracted text is a
  first-class citizen in this layer, the next step — chunking + embeddings
  for retrieval — is a local concern that does not change the upload
  contract or the router.

Why ``pypdf`` + ``python-docx`` over a heavier toolkit
------------------------------------------------------
Both libraries are pure-Python (or pure-Python-plus-tiny-shim), MIT-licensed,
install on every platform we target (including Intel Mac) and have no native
ML dependencies. We deliberately trade some extraction quality on complex
PDFs (scanned pages, multi-column layouts, embedded forms) for portability
and a small dependency footprint. If higher fidelity becomes a real need
later, swapping to PyMuPDF or pdfplumber is a one-function change here.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from app.schemas.log import AttachmentExtracted, AttachmentExtractionFailed


ATTACHMENT_SEPARATOR_TEMPLATE = "--- attachment: {filename} ---"

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".md", ".txt")


class AttachmentExtractionError(RuntimeError):
    """Raised when an uploaded file cannot be parsed."""


def extract_attachment_text(*, filename: str, data: bytes) -> str:
    """Return the plain-text rendering of one attachment.

    Dispatches on the file extension. Raises :class:`AttachmentExtractionError`
    on unknown extensions or parsing failures so the caller can map both to a
    422 without leaking the underlying stack trace.
    """
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".pdf":
            text = _extract_pdf(data)
        elif suffix == ".docx":
            text = _extract_docx(data)
        elif suffix in {".md", ".txt"}:
            text = _decode_text(data)
        else:
            raise AttachmentExtractionError(
                f"Unsupported attachment type {suffix!r} for file {filename!r}. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}."
            )
    except AttachmentExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001 — wrap any parser failure
        AttachmentExtractionFailed(
            filename=filename,
            size_bytes=len(data),
            error=str(exc),
        ).emit()
        raise AttachmentExtractionError(
            f"Could not extract text from {filename!r}: {exc}"
        ) from exc

    AttachmentExtracted(
        filename=filename,
        size_bytes=len(data),
        text_chars=len(text),
    ).emit()
    return text


def _extract_pdf(data: bytes) -> str:
    """Concatenate the text of every page, separated by blank lines.

    Pages with no extractable text (scanned images without OCR) contribute
    nothing — the result keeps the original page order without placeholders.
    """
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        page_text = (page.extract_text() or "").strip()
        if page_text:
            parts.append(page_text)
    return "\n\n".join(parts)


def _extract_docx(data: bytes) -> str:
    """Return the text of every non-empty paragraph in document order."""
    from docx import Document

    document = Document(BytesIO(data))
    parts: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _decode_text(data: bytes) -> str:
    """Decode plain text. Fall back to latin-1 with replacement on bad UTF-8."""
    try:
        return data.decode("utf-8").strip()
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace").strip()


def merge_transcript_and_attachments(
    transcript: str,
    extracted: list[tuple[str, str]],
) -> str:
    """Concatenate ``transcript`` with each ``(filename, text)`` block.

    The separator is fixed so prompts and downstream diffs stay deterministic:
    ``--- attachment: <filename> ---`` on its own line, surrounded by blank
    lines. Order matches the order in ``extracted``.
    """
    if not extracted:
        return transcript

    parts: list[str] = [transcript.rstrip()]
    for filename, text in extracted:
        parts.append(ATTACHMENT_SEPARATOR_TEMPLATE.format(filename=filename))
        parts.append(text.strip())
    return "\n\n".join(parts) + "\n"
