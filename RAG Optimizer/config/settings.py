"""Application settings — single source of truth for all connection strings and tuning params."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Infrastructure ──────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    elasticsearch_url: str = "http://localhost:9200"
    # Plain postgresql:// used by asyncpg; Alembic prepends +asyncpg automatically
    postgres_dsn: str = "postgresql://rag:rag@localhost:5432/rag"
    redis_url: str = "redis://localhost:6379/0"

    # ── API Keys ────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    cohere_api_key: str = ""

    # ── Bandit ─────────────────────────────────────────────────────────
    bandit_epsilon: float = 0.15
    bandit_min_samples: int = 10

    # ── Cache TTLs (seconds) ────────────────────────────────────────────
    classifier_cache_ttl: int = 60
    registry_cache_ttl: int = 300

    # ── Retrieval defaults ─────────────────────────────────────────────
    default_top_k: int = 10
    rerank_top_n: int = 5

    # ── Observability ─────────────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    prometheus_port: int = 8001

    # ── Security ─────────────────────────────────────────────────
    # When non-empty, all API endpoints (except /health) require X-Api-Key header
    api_key: str = ""
    cors_origins: list[str] = ["*"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
