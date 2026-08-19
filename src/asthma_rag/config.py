"""Centralized, environment-driven configuration for the asthma RAG pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime settings loaded from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys
    groq_api_key: str = ""
    cohere_api_key: str = ""

    # Optional HuggingFace cache override
    hf_home: Path | None = None

    # Embedding backend: "local" (sentence-transformers, CPU) or "ollama" (GPU server)
    embedding_backend: str = "local"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"

    # Data paths
    chroma_path: Path = Path("chroma_db")
    data_raw_dir: Path = Path("data/raw")
    data_cleaned_dir: Path = Path("data/cleaned")
    data_chunks_dir: Path = Path("data/chunks")

    # Model names
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    groq_model: str = "openai/gpt-oss-120b"
    cohere_rerank_model: str = "rerank-v4.0-pro"

    # Retrieval / rerank parameters
    retrieval_top_k: int = 50
    rerank_top_n: int = 5
    rerank_threshold: float = 0.3

    # UI
    inhaler_video_id: str = "TuzCfpeieFA"

    @field_validator("embedding_backend")
    @classmethod
    def _validate_embedding_backend(cls, value: str) -> str:
        allowed = {"local", "ollama"}
        if value not in allowed:
            raise ValueError(
                f"embedding_backend must be one of {sorted(allowed)}, got {value!r}"
            )
        return value


# Lazy singleton so tests can patch settings without import-time side effects.
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached ``Settings`` instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
