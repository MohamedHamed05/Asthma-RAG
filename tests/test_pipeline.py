"""Tests for the asthma_rag.pipeline orchestrator.

These tests exercise ingest, index, run, and query with the retrieval and LLM
layers stubbed so no real API calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from asthma_rag.config import Settings
from asthma_rag.llm.groq import GroqChat
from asthma_rag.ollama_embeddings import OllamaEmbeddingFunction
from asthma_rag.pipeline import Pipeline
from asthma_rag.vectorstore import ChromaStore

_LONG_BODY = (
    "Inhaled corticosteroids are the preferred controller medication for "
    "persistent asthma. They reduce airway inflammation, improve lung "
    "function, and decrease the frequency of exacerbations. "
) * 4


def _write_sample_pdf(path: Path) -> None:
    """Create a single-page PDF with enough text to produce one chunk."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(72, 72, 523, 742),
        "3.5 Management of Asthma\n\n" + _LONG_BODY,
    )
    doc.save(path)
    doc.close()


def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a fake Groq API key so GroqChat construction succeeds."""
    from asthma_rag.llm import groq as groq_module

    monkeypatch.setattr(
        groq_module,
        "get_settings",
        lambda: Settings(groq_api_key="test-key", groq_model="test-model"),
    )


def _patch_graph(
    monkeypatch: pytest.MonkeyPatch,
    retrieved: dict[str, Any],
    grades: list[list[bool]],
    answer: str,
) -> None:
    """Stub the graph's retrieval, reranking, and Groq completion."""
    from asthma_rag.agent import graph as graph_module

    monkeypatch.setattr(graph_module, "retrieve", lambda _query, _n=50: retrieved)

    def _fake_rerank(
        _query: str,
        retrieved: dict[str, Any],
        _top_n: int = 5,
    ) -> dict[str, Any]:
        return retrieved

    monkeypatch.setattr(graph_module, "rerank_results", _fake_rerank)

    class _SequenceJson:
        def __init__(self, sequence: list[list[bool]]) -> None:
            self._sequence = list(sequence)
            self._index = 0

        def __call__(self, _messages: list[dict[str, Any]]) -> str:
            payload = self._sequence[self._index]
            self._index += 1
            return json.dumps({"grades": payload})

    monkeypatch.setattr(GroqChat, "complete_json", _SequenceJson(grades))

    def _complete(_self: GroqChat, _messages: list[dict[str, Any]], **_kwargs: Any) -> str:
        return answer

    monkeypatch.setattr(GroqChat, "complete", _complete)

    monkeypatch.setattr(
        graph_module,
        "verifier_node",
        lambda _state: {"verifier_result": {"in_scope": True, "reason": "test"}},
    )
    monkeypatch.setattr(
        graph_module,
        "judge_node",
        lambda _state: {
            "judge_result": {
                "verdict": "approve",
                "reason": "test",
                "confidence": "High",
            }
        },
    )


def test_ingest_writes_chunks_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a PDF directory, when ingesting, then chunks.json is written."""
    _patch_settings(monkeypatch)
    pdf_dir = tmp_path / "raw"
    pdf_dir.mkdir()
    _write_sample_pdf(pdf_dir / "guidelines.pdf")
    out = tmp_path / "chunks.json"

    pipeline = Pipeline(settings=Settings(chroma_path=tmp_path / "chroma"))
    chunks = pipeline.ingest(pdf_dir=pdf_dir, out=out)

    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == chunks
    assert len(chunks) == 1
    assert chunks[0]["doc_name"] == "guidelines"
    assert chunks[0]["chunk_word_count"] >= 20


def test_index_rebuilds_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a chunks file, when indexing, then the Chroma collection is rebuilt."""
    _patch_settings(monkeypatch)
    chunks_path = tmp_path / "chunks.json"
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Asthma is a chronic inflammatory disease.",
            "doc_name": "guidelines",
            "page_number": 1,
            "chunk_word_count": 7,
            "section_title": "Diagnosis",
        }
    ]
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")

    pipeline = Pipeline(settings=Settings(chroma_path=tmp_path / "chroma"))
    pipeline.index(chunks_path=chunks_path)

    result = pipeline._store.query("asthma", n_results=1)
    assert result["ids"][0] == ["c1"]


def test_run_ingests_and_indexes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a PDF directory, when running the pipeline, then chunks are indexed."""
    _patch_settings(monkeypatch)
    pdf_dir = tmp_path / "raw"
    pdf_dir.mkdir()
    _write_sample_pdf(pdf_dir / "guidelines.pdf")
    chunks_path = tmp_path / "chunks.json"

    pipeline = Pipeline(settings=Settings(chroma_path=tmp_path / "chroma"))
    chunks = pipeline.run(pdf_dir=pdf_dir, chunks_path=chunks_path)

    assert chunks_path.exists()
    assert len(chunks) >= 1
    result = pipeline._store.query("asthma", n_results=1)
    assert len(result["ids"][0]) == 1


def test_query_returns_graph_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a question, when querying, then the graph returns the final answer."""
    _patch_settings(monkeypatch)
    _patch_graph(
        monkeypatch,
        retrieved={
            "ids": [["c1", "c2", "c3", "c4", "c5"]],
            "documents": [
                [
                    "Asthma is a chronic inflammatory disease.",
                    "Symptoms include wheeze and shortness of breath.",
                    "Diagnosis requires variable airflow limitation.",
                    "Inhaled corticosteroids are controllers.",
                    "SABA is a reliever.",
                ]
            ],
            "metadatas": [
                [
                    {
                        "doc_name": "guidelines",
                        "chunk_id": f"c{i}",
                        "page_number": i,
                        "chunk_word_count": 10,
                        "section_title": "Diagnosis",
                    }
                    for i in range(1, 6)
                ]
            ],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
        },
        grades=[[True, True, True, False, False]],
        answer="Asthma is a chronic inflammatory disease.",
    )

    pipeline = Pipeline(settings=Settings(chroma_path=tmp_path / "chroma"))
    result = pipeline.query("What is asthma?")

    assert result["final_answer"] == "Asthma is a chronic inflammatory disease."
    assert result["route"] == "retrieve"


def test_index_replaces_collection_with_conflicting_embedding_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a stale collection persisted with a different embedding config, when indexing, then it is rebuilt without conflict."""
    _patch_settings(monkeypatch)
    settings = Settings(chroma_path=tmp_path / "chroma")

    ChromaStore(settings, embedding_function=OllamaEmbeddingFunction())

    chunks = [
        {
            "chunk_id": "c1",
            "text": "Asthma is a chronic inflammatory disease.",
            "doc_name": "guidelines",
            "page_number": 1,
            "chunk_word_count": 7,
            "section_title": "Diagnosis",
        }
    ]

    pipeline = Pipeline(settings=settings)
    pipeline.index(chunks)

    result = pipeline._get_store().query("asthma", n_results=1)
    assert result["ids"][0] == ["c1"]
