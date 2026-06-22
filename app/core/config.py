"""
Application configuration via pydantic-settings.

All settings are loaded from environment variables or a .env file.
No service should import this directly — receive it via dependency injection.
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.exceptions import ConfigurationError


class AppSettings(BaseSettings):
    """Core application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: str = Field(default="development", description="Runtime environment")
    app_host: str = Field(default="0.0.0.0", description="Uvicorn bind host")
    app_port: int = Field(default=8000, description="Uvicorn bind port")
    log_level: str = Field(default="INFO", description="Python logging level")

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = Field(..., description="Qdrant Cloud cluster URL")
    qdrant_api_key: str = Field(..., description="Qdrant Cloud API key")
    qdrant_collection_name: str = Field(
        default="manufacturing_risk",
        description="Qdrant collection name",
    )

    # ── Ingestion ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, description="Token count per chunk")
    chunk_overlap: int = Field(default=64, description="Overlap between chunks")
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Output dimension of the embedding model",
    )

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Ensure log level is one of Python's standard levels."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in valid_levels:
            raise ConfigurationError(
                message=f"Invalid log level: {value}",
                detail=f"Must be one of: {', '.join(sorted(valid_levels))}",
            )
        return upper

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, overlap: int, info) -> int:
        """Ensure overlap is strictly less than chunk size."""
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and overlap >= chunk_size:
            raise ConfigurationError(
                message=f"chunk_overlap ({overlap}) must be less than chunk_size ({chunk_size})",
            )
        return overlap

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, value: str) -> str:
        """Ensure environment is a known value."""
        valid_envs = {"development", "staging", "production"}
        lower = value.lower()
        if lower not in valid_envs:
            raise ConfigurationError(
                message=f"Invalid app_env: {value}",
                detail=f"Must be one of: {', '.join(sorted(valid_envs))}",
            )
        return lower


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Return the singleton AppSettings instance.

    Cached after first call. In tests, call get_settings.cache_clear()
    before patching env vars to force re-evaluation.
    """
    try:
        return AppSettings()
    except Exception as exc:
        raise ConfigurationError(
            message="Failed to load application settings",
            detail=str(exc),
        ) from exc