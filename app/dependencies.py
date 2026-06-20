"""FastAPI dependency factories for shared singletons."""

from __future__ import annotations

from functools import lru_cache

import structlog
from openai import OpenAI

from app.config import get_settings
from app.services.cache import EstimationCache
from app.services.estimation import EstimationService
from app.services.llm_wrapper import LLMWrapper
from app.services.sessions import SessionStore

log = structlog.get_logger()


@lru_cache
def get_cache() -> EstimationCache:
    settings = get_settings()
    return EstimationCache.from_url(settings.REDIS_URL, ttl=settings.CACHE_TTL)


@lru_cache
def get_llm_wrapper() -> LLMWrapper:
    settings = get_settings()
    return LLMWrapper(
        openai_api_key=settings.OPENAI_API_KEY,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        primary_model=settings.PRIMARY_MODEL,
        fallback_model=settings.FALLBACK_MODEL,
        timeout=settings.LLM_TIMEOUT,
        num_retries=settings.LLM_RETRIES,
        cache=get_cache(),
    )


@lru_cache
def get_openai_client() -> OpenAI | None:
    """Lazy OpenAI client used by the input moderation guardrail."""
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        log.warning("openai_client_disabled", reason="no_api_key")
        return None
    return OpenAI(api_key=settings.OPENAI_API_KEY)


@lru_cache
def get_estimation_service() -> EstimationService:
    return EstimationService(
        llm_wrapper=get_llm_wrapper(),
        exact_cache=get_cache(),
        openai_client=get_openai_client(),
    )


@lru_cache
def get_session_store() -> SessionStore:
    """Process-wide in-memory store for multi-turn conversation sessions."""
    settings = get_settings()
    return SessionStore(max_turns=settings.MAX_TURNS)
