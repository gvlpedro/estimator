from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_CONFIG_PATH = Path(__file__).resolve().parent / "app_config.yml"


class ActorCriticBossConfig(BaseModel):
    iterations: int = Field(default=2, ge=1, le=10)


class AppConfig(BaseModel):
    actor_critic_boss: ActorCriticBossConfig = Field(default_factory=ActorCriticBossConfig)


class FileConfig(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)


@lru_cache
def get_file_config() -> FileConfig:
    """Load the YAML-backed app config. Falls back to defaults if the file is missing."""
    if not APP_CONFIG_PATH.is_file():
        return FileConfig()
    raw = yaml.safe_load(APP_CONFIG_PATH.read_text()) or {}
    return FileConfig.model_validate(raw)


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Session 2 fields (kept for backwards compatibility with the live demos) ---
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    LLM_PROVIDER: Literal["openai", "anthropic"] = "anthropic"
    LLM_MODEL: str = "claude-haiku-4-5"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"

    # --- Session 3 fields (LiteLLM wrapper, Redis cache, Streamlit transport) ---
    PRIMARY_MODEL: str = "gpt-4o-mini"
    FALLBACK_MODEL: str = "claude-haiku-4-5-20251001"
    LLM_TIMEOUT: int = 30
    LLM_RETRIES: int = 2

    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 86400

    ESTIMATOR_API_BASE_URL: str = "http://localhost:8000"

    # --- Session 5 fields (conversational sessions) ---
    # MAX_TURNS counts user+assistant PAIRS. The system prompt is pinned and
    # never counted. Default 6 keeps the prompt budget bounded while still
    # giving the LLM enough recent context for multi-turn refinement.
    MAX_TURNS: int = 6

    # Hard cap on extracted attachment text after parsing. The cap exists so
    # a malicious or accidental 200-page upload cannot blow up the prompt
    # budget. 60_000 chars ≈ 15 K tokens for English text — large enough to
    # absorb a meeting transcript or a short SOW, small enough to leave the
    # actor with room to think. Stress tests verify the truncation kicks in
    # at the boundary.
    MAX_ATTACHMENT_CHARS: int = 60_000

    @model_validator(mode="after")
    def validate_at_least_one_api_key(self) -> "Settings":
        """LiteLLM may try either provider via fallback, so we require at least one key."""
        if not self.OPENAI_API_KEY and not self.ANTHROPIC_API_KEY:
            raise ValueError(
                "At least one of OPENAI_API_KEY or ANTHROPIC_API_KEY must be set"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings (singleton)."""
    return Settings()
