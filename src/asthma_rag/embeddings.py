"""Embedding factory: Qwen3 via a GPU Ollama server or local sentence-transformers.

``get_embedding_function`` returns the Chroma embedding function matching the
configured backend (``ollama`` or ``local``). The local backend additionally
exposes query/document encoding helpers so the Qwen3 instruction prompt is
applied to queries only.
"""

from __future__ import annotations

import os
from functools import lru_cache

from chromadb.api.types import Documents, EmbeddingFunction
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from asthma_rag.config import Settings, get_settings
from asthma_rag.ollama_embeddings import OllamaEmbeddingFunction


def _apply_hf_home(settings: Settings) -> None:
    """Propagate the configured ``HF_HOME`` into the process environment.

    ``sentence-transformers`` reads the HuggingFace cache location from the
    environment (not from our settings object), so the ``.env`` value must be
    applied before the model is loaded. An existing environment variable
    always wins.
    """
    if settings.hf_home is not None:
        os.environ.setdefault("HF_HOME", str(settings.hf_home))


@lru_cache(maxsize=1)
def _get_local_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """Return a cached local sentence-transformers embedding function."""
    settings = get_settings()
    _apply_hf_home(settings)
    return SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model,
        device="cpu",
        normalize_embeddings=False,
        trust_remote_code=True,
    )


@lru_cache(maxsize=1)
def _get_ollama_embedding_function() -> OllamaEmbeddingFunction:
    """Return a cached Ollama-backed embedding function."""
    settings = get_settings()
    return OllamaEmbeddingFunction(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )


def get_embedding_function() -> EmbeddingFunction[Documents]:
    """Return the configured embedding function (``ollama`` or ``local``)."""
    if get_settings().embedding_backend == "ollama":
        return _get_ollama_embedding_function()
    return _get_local_embedding_function()


def _require_local_backend() -> None:
    if get_settings().embedding_backend != "local":
        raise RuntimeError(
            "encode_query and encode_documents require embedding_backend=local; "
            "the ollama backend embeds through get_embedding_function() instead."
        )


def encode_query(text: str) -> list[float]:
    """Embed a single query with the model's ``query`` instruction prompt."""
    _require_local_backend()
    model = _get_local_embedding_function()._model
    return model.encode(
        [text],
        prompt_name="query",
        normalize_embeddings=False,
    ).tolist()[0]


def encode_documents(texts: list[str]) -> list[list[float]]:
    """Embed documents without any instruction prompt."""
    _require_local_backend()
    model = _get_local_embedding_function()._model
    return model.encode(
        texts,
        normalize_embeddings=False,
    ).tolist()