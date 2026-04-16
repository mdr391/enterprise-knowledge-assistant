"""
Centralised configuration — all settings read from environment variables.
Supports .env files via python-dotenv.
"""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"      # development | staging | production
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = ["*"]

    # ── LLM (Anthropic Claude) ────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""
    LLM_MODEL: str = "claude-sonnet-4-20250514"
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.2          # Lower = more deterministic / factual
    LLM_STREAM: bool = True

    # ── Embeddings (OpenAI — swappable) ──────────────────────────────────────
    OPENAI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536

    # ── Vector Store (ChromaDB) ───────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION: str = "enterprise_knowledge"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 5              # Number of chunks to retrieve
    RETRIEVAL_SCORE_THRESHOLD: float = 0.35  # Minimum similarity score

    # ── Ingestion ─────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512                 # Tokens per chunk
    CHUNK_OVERLAP: int = 64              # Token overlap between chunks
    MAX_DOCUMENT_SIZE_MB: float = 10.0

    # ── Monitoring ────────────────────────────────────────────────────────────
    ENABLE_METRICS: bool = True
    METRICS_WINDOW_SIZE: int = 1000       # Rolling window for latency tracking


settings = Settings()
