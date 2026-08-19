"""Tests for asthma_rag.retrieval.

RED phase: these tests should all FAIL before src/asthma_rag/retrieval.py exists.
"""

from __future__ import annotations

import pytest

from asthma_rag import retrieval


class _StubStore:
    """Mimics ChromaStore.query: records calls, returns a fixed Chroma-like dict."""

    def __init__(self, result: dict) -> None:
        self._result = result
        self.calls: list[tuple[str, int]] = []

    def query(self, query_text: str, n_results: int) -> dict:
        self.calls.append((query_text, n_results))
        return self._result


def _patch_store(monkeypatch: pytest.MonkeyPatch, stub: _StubStore) -> None:
    """Replace ChromaStore with the stub so retrieve() never hits disk."""
    monkeypatch.setattr(retrieval, "get_settings", lambda: object())
    monkeypatch.setattr(retrieval, "get_embedding_function", lambda: None)
    monkeypatch.setattr(
        retrieval, "ChromaStore", lambda _settings, embedding_function=None: stub
    )


# ---------------------------------------------------------------------------
# retrieve()
# ---------------------------------------------------------------------------


class TestRetrieve:
    """retrieve must forward the query to ChromaStore and return the raw result."""

    def test_returns_chroma_like_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given a stubbed ChromaStore, when retrieving, then the raw query dict is returned."""
        chroma_like = {
            "ids": [["guidelines_1_0"]],
            "documents": [["Inhaled corticosteroids are the preferred controller."]],
            "metadatas": [[{"doc_name": "guidelines", "page_number": 1}]],
            "distances": [[0.1234]],
        }
        stub = _StubStore(chroma_like)
        _patch_store(monkeypatch, stub)

        result = retrieval.retrieve("preferred controller for asthma")

        assert result == chroma_like
        assert stub.calls == [("preferred controller for asthma", 50)]

    def test_respects_custom_n_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given an explicit n_results, when retrieving, then it is forwarded unchanged."""
        stub = _StubStore({"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]})
        _patch_store(monkeypatch, stub)

        retrieval.retrieve("any query", n_results=7)

        assert stub.calls == [("any query", 7)]

    def test_uses_configured_embedding_function_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given no explicit embedding function, when retrieving, then the Qwen3 factory result is passed to the store."""
        sentinel = object()
        captured: dict[str, object] = {}
        stub = _StubStore({"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]})

        def fake_store(_settings: object, embedding_function: object = None) -> _StubStore:
            captured["ef"] = embedding_function
            return stub

        monkeypatch.setattr(retrieval, "get_settings", lambda: object())
        monkeypatch.setattr(retrieval, "get_embedding_function", lambda: sentinel)
        monkeypatch.setattr(retrieval, "ChromaStore", fake_store)

        retrieval.retrieve("any query")

        assert captured["ef"] is sentinel

    def test_explicit_embedding_function_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given an explicit embedding function, when retrieving, then it is used instead of the Qwen3 factory."""
        sentinel = object()
        marker = object()
        captured: dict[str, object] = {}
        stub = _StubStore({"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]})

        def fake_store(_settings: object, embedding_function: object = None) -> _StubStore:
            captured["ef"] = embedding_function
            return stub

        monkeypatch.setattr(retrieval, "get_settings", lambda: object())
        monkeypatch.setattr(retrieval, "get_embedding_function", lambda: marker)
        monkeypatch.setattr(retrieval, "ChromaStore", fake_store)

        retrieval.retrieve("any query", embedding_function=sentinel)

        assert captured["ef"] is sentinel


# ---------------------------------------------------------------------------
# format_retrieved_context()
# ---------------------------------------------------------------------------


class TestFormatRetrievedContext:
    """format_retrieved_context must render every chunk header and keep scores labeled as metadata."""

    def test_includes_all_chunk_headers_and_user_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a reranked result with scores, when formatted with a user query, then every expected header is present."""
        results = {
            "documents": [["Inhaled corticosteroids are the preferred controller."]],
            "metadatas": [
                [
                    {
                        "doc_name": "GINA-2026-Strategy-Report-WMS.pdf",
                        "chunk_id": "GINA-2026-Strategy-Report-WMS.pdf_10_0",
                        "page_number": 10,
                        "chunk_word_count": 12,
                        "section_title": "3.5 Management of Asthma",
                        "vector_distance": 0.1234,
                        "rerank_score": 0.8765,
                    }
                ]
            ],
            "distances": [[0.1234]],
        }

        context = retrieval.format_retrieved_context(
            results, user_query="What is the preferred controller?"
        )

        assert "USER QUERY:\nWhat is the preferred controller?" in context
        assert "RETRIEVED CLINICAL CONTEXT:" in context
        assert "===== Chunk 1 =====" in context
        for header in (
            "Source:",
            "Chunk ID:",
            "Page:",
            "Chunk Word Count:",
            "Vector Distance:",
            "Rerank Score:",
            "Content:",
        ):
            assert header in context

    def test_scores_are_metadata_not_evidence(self) -> None:
        """Given scores in metadata, when formatted, then they appear only as labeled lines, never inside Content."""
        results = {
            "documents": [["Inhaled corticosteroids are the preferred controller."]],
            "metadatas": [
                [
                    {
                        "doc_name": "guidelines.pdf",
                        "chunk_id": "guidelines.pdf_1_0",
                        "page_number": 1,
                        "chunk_word_count": 8,
                        "vector_distance": 0.5,
                        "rerank_score": 0.75,
                    }
                ]
            ],
            "distances": [[0.5]],
        }

        context = retrieval.format_retrieved_context(results)

        assert "Vector Distance: 0.5000" in context
        assert "Rerank Score: 0.7500" in context
        content_section = context.split("Content:")[1]
        assert "0.7500" not in content_section
        assert "0.5000" not in content_section
        assert "Inhaled corticosteroids are the preferred controller." in content_section

    def test_missing_scores_omit_score_lines(self) -> None:
        """Given metadata without scores (raw retrieve output), when formatted, then no score headers are rendered."""
        results = {
            "documents": [["SABA is the reliever of choice."]],
            "metadatas": [
                [{"doc_name": "guidelines.pdf", "chunk_id": "g_0", "page_number": 3, "chunk_word_count": 6}]
            ],
            "distances": [[0.2]],
        }

        context = retrieval.format_retrieved_context(results)

        assert "Source: guidelines.pdf" in context
        assert "Vector Distance:" not in context
        assert "Rerank Score:" not in context
