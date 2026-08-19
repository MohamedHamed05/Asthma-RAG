"""Tests for asthma_rag.retrieval.rerank_results index mapping.

RED phase: these tests should all FAIL before src/asthma_rag/retrieval.py exists.
"""

from __future__ import annotations

import pytest

from asthma_rag import retrieval
from asthma_rag.rerank.cohere_v2 import RerankResult


class _StubReranker:
    """Mimics CohereRerank.rerank: records calls, returns fixed RerankResults."""

    def __init__(self, results: list[RerankResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        self.calls.append((query, documents, top_n))
        return self._results


def _patch_reranker(monkeypatch: pytest.MonkeyPatch, stub: _StubReranker) -> None:
    monkeypatch.setattr(retrieval, "CohereRerank", lambda: stub)


def _retrieved(
    documents: list[str], metadatas: list[dict], distances: list[float]
) -> dict:
    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


class TestRerankResults:
    """rerank_results must map Cohere result indices back to original documents/metadata."""

    def test_maps_indices_back_to_original_entries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given Cohere returns indices [2, 0, 1], when reranking, then documents, metadata, and distances are reordered to match."""
        documents = ["doc-alpha", "doc-beta", "doc-gamma"]
        metadatas = [
            {"doc_name": "g.pdf", "chunk_id": "g.pdf_0", "page_number": 1},
            {"doc_name": "g.pdf", "chunk_id": "g.pdf_1", "page_number": 2},
            {"doc_name": "g.pdf", "chunk_id": "g.pdf_2", "page_number": 3},
        ]
        distances = [0.9, 0.5, 0.1]
        stub = _StubReranker(
            [
                RerankResult(index=2, relevance_score=0.95, document="doc-gamma"),
                RerankResult(index=0, relevance_score=0.80, document="doc-alpha"),
                RerankResult(index=1, relevance_score=0.70, document="doc-beta"),
            ]
        )
        _patch_reranker(monkeypatch, stub)

        result = retrieval.rerank_results("query", _retrieved(documents, metadatas, distances), top_n=3)

        assert result["documents"][0] == ["doc-gamma", "doc-alpha", "doc-beta"]
        assert [m["chunk_id"] for m in result["metadatas"][0]] == ["g.pdf_2", "g.pdf_0", "g.pdf_1"]
        assert result["distances"][0] == [0.1, 0.9, 0.5]
        assert stub.calls == [("query", documents, 3)]

    def test_preserves_metadata_and_adds_scores(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a reranked entry, when mapping, then original metadata keys survive and rerank_score/vector_distance are added."""
        documents = ["doc-alpha", "doc-beta"]
        metadatas = [
            {"doc_name": "g.pdf", "chunk_id": "g.pdf_0", "page_number": 1},
            {"doc_name": "g.pdf", "chunk_id": "g.pdf_1", "page_number": 2},
        ]
        distances = [0.9, 0.2]
        stub = _StubReranker(
            [RerankResult(index=1, relevance_score=0.88, document="doc-beta")]
        )
        _patch_reranker(monkeypatch, stub)

        result = retrieval.rerank_results("query", _retrieved(documents, metadatas, distances), top_n=1)

        meta = result["metadatas"][0][0]
        assert meta["chunk_id"] == "g.pdf_1"
        assert meta["doc_name"] == "g.pdf"
        assert meta["page_number"] == 2
        assert meta["rerank_score"] == 0.88
        assert meta["vector_distance"] == 0.2
        assert result["documents"][0] == ["doc-beta"]
        assert result["distances"][0] == [0.2]

    def test_rerank_score_uses_cohere_not_distance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a low vector distance and a high rerank score, when mapping, then rerank_score holds the Cohere score, not the distance."""
        documents = ["doc-alpha"]
        metadatas = [{"chunk_id": "g.pdf_0"}]
        distances = [0.9]
        stub = _StubReranker([RerankResult(index=0, relevance_score=0.95, document="doc-alpha")])
        _patch_reranker(monkeypatch, stub)

        result = retrieval.rerank_results("query", _retrieved(documents, metadatas, distances), top_n=1)

        assert result["metadatas"][0][0]["rerank_score"] == 0.95
        assert result["metadatas"][0][0]["vector_distance"] == 0.9
