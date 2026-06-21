"""Exact-match Redis cache for LLM responses.

The cache key is a SHA-256 of the *full* system prompt plus the user message
plus the generation knobs (model, max_tokens, thinking_budget). That means any
change in Session 2 controls (preprocessing, num_examples, example_format,
ACTIVE_OUTPUT_PROMPT) implicitly invalidates the cache without manual flushing,
because those changes alter the system prompt text.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis

from app.schemas.log import (
    CacheGetFailed,
    CacheHit,
    CacheMiss,
    CacheSetFailed,
    CacheStored,
)


class EstimationCache:
    """Thin wrapper around redis-py with deterministic keying and TTL."""

    def __init__(self, redis_client: redis.Redis, ttl: int = 86400):
        self.redis = redis_client
        self.ttl = ttl

    @classmethod
    def from_url(cls, url: str, ttl: int = 86400) -> "EstimationCache":
        return cls(redis.from_url(url, decode_responses=True), ttl=ttl)

    @staticmethod
    def make_key(
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int,
        thinking_budget: int | None,
    ) -> str:
        payload = json.dumps(
            {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "model": model,
                "max_tokens": max_tokens,
                "thinking_budget": thinking_budget,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"estimation:{digest}"

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            cached = self.redis.get(key)
        except redis.RedisError as exc:
            CacheGetFailed(error=str(exc)).emit()
            return None
        if cached:
            CacheHit(key_prefix=key[:24]).emit()
            return json.loads(cached)
        CacheMiss(key_prefix=key[:24]).emit()
        return None

    def set(self, key: str, response: dict[str, Any]) -> None:
        try:
            self.redis.setex(key, self.ttl, json.dumps(response))
            CacheStored(key_prefix=key[:24], ttl=self.ttl).emit()
        except redis.RedisError as exc:
            CacheSetFailed(error=str(exc)).emit()
