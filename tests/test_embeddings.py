"""Tests for the local embedding factory.

Locks the contract of ``asthma_rag.embeddings``: the factory returns a
callable embedding function backed by the configured Qwen3 model, the
vectors have the expected dimension, and query encoding applies the
instruction prompt while document encoding does not.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from asthma_rag.config import Settings
from asthma_rag.embeddings import (
    _apply_hf_home,
    encode_documents,
    encode_query,
    get_embedding_function,
)

EXPECTED_DIM = 1024


@pytest.fixture(autouse=True)
def _local_backend(monkeypatch: pytest.MonkeyPatch):
    """Force the local backend so these tests never depend on .env contents."""
    from asthma_rag import embeddings as embeddings_module

    monkeypatch.setattr(
        embeddings_module,
        "get_settings",
        lambda: Settings(embedding_backend="local"),
    )
    embeddings_module._get_local_embedding_function.cache_clear()
    yield
    embeddings_module._get_local_embedding_function.cache_clear()


def test_get_embedding_function_returns_callable() -> None:
    """Given the factory, when called, then a callable embedding function is returned."""
    ef = get_embedding_function()

    assert callable(ef)


def test_embedding_dimension_is_1024() -> None:
    """Given Qwen3-Embedding-0.6B, when embedding a document, then the vector has 1024 dimensions."""
    vectors = encode_documents(["asthma is a chronic inflammatory airway disease"])

    assert len(vectors) == 1
    assert len(vectors[0]) == EXPECTED_DIM


def test_query_embedding_differs_from_document_embedding() -> None:
    """Given the same text, when encoded as query vs document, then the vectors differ (instruction prompt applied)."""
    text = "What is the first-line treatment for asthma?"
    query_vec = encode_query(text)
    doc_vec = encode_documents([text])[0]

    assert query_vec != doc_vec


def test_apply_hf_home_sets_env_when_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Given a configured hf_home, when applied, then HF_HOME is set in the environment."""
    monkeypatch.delenv("HF_HOME", raising=False)

    _apply_hf_home(Settings(hf_home=tmp_path))

    assert os.environ["HF_HOME"] == str(tmp_path)


def test_apply_hf_home_keeps_existing_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Given an existing HF_HOME env var, when applied, then the env var wins over settings."""
    monkeypatch.setenv("HF_HOME", r"C:\existing\cache")

    _apply_hf_home(Settings(hf_home=tmp_path))

    assert os.environ["HF_HOME"] == r"C:\existing\cache"


def test_apply_hf_home_noop_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given no configured hf_home, when applied, then the environment is untouched."""
    monkeypatch.delenv("HF_HOME", raising=False)

    _apply_hf_home(Settings(hf_home=None))

    assert "HF_HOME" not in os.environ


def test_encode_query_rejects_ollama_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the ollama backend, when calling encode_query, then a clear error is raised."""
    from asthma_rag import embeddings as embeddings_module

    monkeypatch.setattr(
        embeddings_module,
        "get_settings",
        lambda: Settings(embedding_backend="ollama"),
    )

    with pytest.raises(RuntimeError, match="embedding_backend=local"):
        encode_query("test")