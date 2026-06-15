from collections.abc import Iterator

from fastapi.testclient import TestClient

from app.dependencies import get_llm_wrapper
from app.main import app


class _StubWrapper:
    """Minimal stand-in for LLMWrapper that yields canned chunks."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.calls: list[dict] = []

    def complete_stream(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model_override: str | None,
        max_tokens: int,
    ) -> Iterator[str]:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "model_override": model_override,
                "max_tokens": max_tokens,
            }
        )
        for chunk in self._chunks:
            yield chunk


def test_stream_endpoint_emits_token_and_done_events() -> None:
    stub = _StubWrapper(chunks=["Hello ", "from ", "the ", "estimator."])
    app.dependency_overrides[get_llm_wrapper] = lambda: stub
    try:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/api/v1/estimate/stream",
                json={"transcription": "x" * 60},
            ) as response:
                assert response.status_code == 200
                body = b"".join(response.iter_bytes()).decode()
        # The response is one or more SSE messages separated by blank lines.
        assert "event: token" in body
        assert "data: Hello" in body
        assert "data: estimator." in body
        assert "event: done" in body
        assert len(stub.calls) == 1
        assert stub.calls[0]["user_message"] == "x" * 60
    finally:
        app.dependency_overrides.pop(get_llm_wrapper, None)


def test_stream_endpoint_serialises_multiline_chunk_as_multiple_data_lines() -> None:
    """A chunk with internal newlines must be split into one ``data:`` line per
    physical line, with the message terminated by a blank line. Clients must
    join those data lines with ``\\n`` to recover the original payload — that
    is the contract this test pins down on the server side.
    """
    multiline_chunk = "## Project summary\n\nThis is the body."
    stub = _StubWrapper(chunks=[multiline_chunk])
    app.dependency_overrides[get_llm_wrapper] = lambda: stub
    try:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/api/v1/estimate/stream",
                json={"transcription": "x" * 60},
            ) as response:
                assert response.status_code == 200
                body = b"".join(response.iter_bytes()).decode()
        assert "data: ## Project summary" in body
        assert "data: This is the body." in body
        assert "event: token" in body
        assert "event: done" in body
    finally:
        app.dependency_overrides.pop(get_llm_wrapper, None)


def test_stream_endpoint_rejects_short_transcription() -> None:
    stub = _StubWrapper(chunks=["irrelevant"])
    app.dependency_overrides[get_llm_wrapper] = lambda: stub
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/estimate/stream",
                json={"transcription": "too short"},
            )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_llm_wrapper, None)
