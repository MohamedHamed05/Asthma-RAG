"""Tests for the Cohere v2 rerank wrapper.

RED phase: these tests should all FAIL before src/asthma_rag/rerank/cohere_v2.py exists.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from asthma_rag.rerank import cohere_v2
from asthma_rag.rerank.cohere_v2 import CohereRerank, RerankConfigError, RerankResult

DOCUMENTS = ["doc-alpha", "doc-beta", "doc-gamma"]


# ---------------------------------------------------------------------------
# Stubs for the Cohere v2 SDK response shape
# ---------------------------------------------------------------------------


class _StubResult:
    """Mimics cohere's RerankResult: .index + .relevance_score."""

    def __init__(self, index: int, relevance_score: float) -> None:
        self.index = index
        self.relevance_score = relevance_score


class _StubResponse:
    def __init__(self, results: list[_StubResult]) -> None:
        self.results = results


class _StubClient:
    def __init__(self, results: list[_StubResult]) -> None:
        self._results = results

    def rerank(self, model: str, query: str, documents: list[str], top_n: int):
        return _StubResponse(self._results)


def _patch_env(monkeypatch: pytest.MonkeyPatch, api_key: str) -> None:
    settings = SimpleNamespace(cohere_api_key=api_key, cohere_rerank_model="rerank-v4.0-pro")
    monkeypatch.setattr(cohere_v2, "get_settings", lambda: settings)


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: _StubClient) -> None:
    monkeypatch.setattr(cohere_v2.cohere, "ClientV2", lambda api_key: client)


# ---------------------------------------------------------------------------
# Configuration error
# ---------------------------------------------------------------------------

class TestConstruction:
    """CohereRerank must validate the API key at construction time."""

    def test_missing_api_key_raises_config_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_env(monkeypatch, api_key="")

        with pytest.raises(RerankConfigError):
            CohereRerank()


# ---------------------------------------------------------------------------
# Reranking behavior
# ---------------------------------------------------------------------------

class TestRerank:
    """rerank must map Cohere result indices back to the original documents."""

    def test_returns_results_in_stub_order_with_original_documents(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a stub response ordered [2, 0, 1], when reranked, then results keep that order mapped to original documents."""
        _patch_env(monkeypatch, api_key="test-key")
        _patch_client(
            monkeypatch,
            _StubClient([_StubResult(2, 0.9), _StubResult(0, 0.8), _StubResult(1, 0.7)]),
        )
        reranker = CohereRerank()

        results = reranker.rerank("what is asthma?", DOCUMENTS, top_n=3)

        assert [r.index for r in results] == [2, 0, 1]
        assert [r.document for r in results] == ["doc-gamma", "doc-alpha", "doc-beta"]
        assert [r.relevance_score for r in results] == [0.9, 0.8, 0.7]
        assert all(isinstance(r, RerankResult) for r in results)

    def test_empty_documents_returns_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given no documents, when reranked, then an empty list is returned without calling the API."""
        _patch_env(monkeypatch, api_key="test-key")

        class _ExplodingClient:
            def rerank(self, **kwargs):
                raise AssertionError("rerank must not be called with empty documents")

        _patch_client(monkeypatch, _ExplodingClient())
        reranker = CohereRerank()

        assert reranker.rerank("what is asthma?", [], top_n=5) == []
