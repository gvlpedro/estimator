"""HTTP client for the AI service (servicio_ia).

This is the ONLY place in the business backend that talks HTTP to the
FastAPI AI service on :8001 — every other module goes through this client.
"""

from __future__ import annotations

import httpx
import structlog

log = structlog.get_logger()


class AIServiceError(Exception):
    """Raised when the AI service is unreachable or returns an error.

    ``status_code`` is the upstream HTTP status when there was a response,
    or ``None`` when the service could not be reached at all.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AIServiceClient:
    """Thin async client for the AI service's semantic search endpoint."""

    def __init__(self, base_url: str, timeout: float = 15.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def search(self, query: str, k: int = 5) -> dict:
        """POST /search on the AI service and return the raw response body."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/search",
                    json={"query": query, "k": k},
                )
                response.raise_for_status()
                try:
                    return response.json()
                except ValueError as exc:
                    log.error("ai_service_invalid_json", body=response.text[:500])
                    raise AIServiceError("AI service returned invalid JSON") from exc
        except httpx.HTTPStatusError as exc:
            log.error(
                "ai_service_search_failed",
                status_code=exc.response.status_code,
                body=exc.response.text[:500],
            )
            raise AIServiceError(
                f"AI service returned {exc.response.status_code}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.HTTPError as exc:
            log.error("ai_service_unreachable", base_url=self._base_url, error=str(exc))
            raise AIServiceError("AI service unreachable") from exc
