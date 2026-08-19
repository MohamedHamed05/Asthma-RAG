"""Tests for the Ollama-backed Chroma embedding function."""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest

from asthma_rag import embeddings as embeddings_module
from asthma_rag.config import Settings
from asthma_rag.ollama_embeddings import OllamaEmbeddingFunction


class _StubbedOllama(OllamaEmbeddingFunction):
    """Ollama function with the HTTP layer replaced by deterministic vectors."""

    def __init__(self, calls: list[list[str]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls = calls

    def _fetch_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t)), 0.0, 0.0] for t in texts]


def test_call_sub_batches_and_preserves_order() -> None:
    calls: list[list[str]] = []
    ef = _StubbedOllama(calls, base_url="http://x", model="m", batch_size=2)

    vectors = ef(["a", "bb", "ccc", "d", "ee"])

    assert calls == [["a", "bb"], ["ccc", "d"], ["ee"]]
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 1.0, 2.0]


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_fetch_embeddings_posts_and_parses_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        assert request.full_url == "http://ollama:11434/api/embed"
        body = json.loads(request.data.decode("utf-8"))
        assert body == {"model": "test-model", "input": ["a", "b"]}
        return _FakeResponse({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ef = OllamaEmbeddingFunction(base_url="http://ollama:11434", model="test-model")

    assert ef._fetch_embeddings(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]


def test_fetch_embeddings_wraps_connection_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ef = OllamaEmbeddingFunction()

    with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
        ef._fetch_embeddings(["a"])


def test_fetch_embeddings_wraps_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        raise urllib.error.HTTPError(
            "http://ollama:11434/api/embed",
            404,
            "Not Found",
            None,
            io.BytesIO(b"no such model"),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ef = OllamaEmbeddingFunction()

    with pytest.raises(RuntimeError, match="HTTP 404"):
        ef._fetch_embeddings(["a"])


def test_fetch_embeddings_rejects_mismatched_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: _FakeResponse({"embeddings": [[0.1]]}),
    )
    ef = OllamaEmbeddingFunction()

    with pytest.raises(RuntimeError, match="1 embeddings for 2 inputs"):
        ef._fetch_embeddings(["a", "b"])


def test_config_round_trip() -> None:
    ef = OllamaEmbeddingFunction(base_url="http://o:11434", model="m", batch_size=8)

    config = ef.get_config()
    rebuilt = OllamaEmbeddingFunction.build_from_config(config)

    assert rebuilt.get_config() == config
    assert OllamaEmbeddingFunction.name() == "ollama-embedding"


def test_factory_dispatches_to_ollama_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        embeddings_module,
        "get_settings",
        lambda: Settings(
            embedding_backend="ollama",
            ollama_base_url="http://o:11434",
            ollama_embedding_model="qwen3-embedding:0.6b",
        ),
    )
    embeddings_module._get_ollama_embedding_function.cache_clear()

    try:
        ef = embeddings_module.get_embedding_function()

        assert isinstance(ef, OllamaEmbeddingFunction)
        assert ef.get_config()["model"] == "qwen3-embedding:0.6b"
    finally:
        embeddings_module._get_ollama_embedding_function.cache_clear()


def test_settings_rejects_unknown_embedding_backend() -> None:
    with pytest.raises(ValueError, match="embedding_backend"):
        Settings(embedding_backend="gpu-magic")
