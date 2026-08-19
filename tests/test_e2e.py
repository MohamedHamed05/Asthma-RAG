"""End-to-end smoke tests for the scenario contract.

These tests exercise a full ``Pipeline`` end-to-end with the retrieval and LLM
layers stubbed. They assert the four scenario outcomes from the project plan:

1. Happy-path definition question -> sourced answer + route=retrieve.
2. Inhaler query -> route=inhaler + embedded ALA video.
3. Insufficient evidence -> safe fallback.
4. Regression lock -> system prompt SHA-256 unchanged.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from asthma_rag.config import Settings
from asthma_rag.llm.groq import GroqChat
from asthma_rag.pipeline import Pipeline
from asthma_rag.prompts import SYSTEM_PROMPT


def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a fake Groq API key so GroqChat construction succeeds."""
    from asthma_rag.llm import groq as groq_module

    monkeypatch.setattr(
        groq_module,
        "get_settings",
        lambda: Settings(groq_api_key="test-key", groq_model="test-model"),
    )


def _patch_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    retrieved: dict[str, Any],
) -> None:
    """Stub retrieve/rerank in the graph so no Chroma/Cohere calls are made."""
    from asthma_rag.agent import graph as graph_module

    monkeypatch.setattr(graph_module, "retrieve", lambda _query, _n=50: retrieved)

    def _fake_rerank(
        _query: str,
        retrieved: dict[str, Any],
        _top_n: int = 5,
    ) -> dict[str, Any]:
        return retrieved

    monkeypatch.setattr(graph_module, "rerank_results", _fake_rerank)


def _patch_grades(
    monkeypatch: pytest.MonkeyPatch,
    grades_sequence: list[list[bool]],
) -> None:
    """Patch GroqChat.complete_json to return a sequence of grade payloads."""

    class _SequenceJson:
        def __init__(self, sequence: list[list[bool]]) -> None:
            self._sequence = list(sequence)
            self._index = 0

        def __call__(self, _messages: list[dict[str, Any]]) -> str:
            grades = self._sequence[self._index]
            self._index += 1
            return json.dumps({"grades": grades})

    monkeypatch.setattr(GroqChat, "complete_json", _SequenceJson(grades_sequence))


def _patch_answer(monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    """Patch GroqChat.complete to return a fixed answer."""

    def _complete(_self: GroqChat, _messages: list[dict[str, Any]], **_kwargs: Any) -> str:
        return answer

    monkeypatch.setattr(GroqChat, "complete", _complete)


def _patch_verifier(monkeypatch: pytest.MonkeyPatch, in_scope: bool = True) -> None:
    """Stub the entry verifier so tests control scope without an LLM call."""
    from asthma_rag.agent import graph as graph_module

    monkeypatch.setattr(
        graph_module,
        "verifier_node",
        lambda _state: {"verifier_result": {"in_scope": in_scope, "reason": "test"}},
    )


def _patch_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the judge to a no-op approve so final_answer is unchanged."""
    from asthma_rag.agent import graph as graph_module

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


_RETRIEVED_FIVE: dict[str, Any] = {
    "ids": [["c1", "c2", "c3", "c4", "c5"]],
    "documents": [[
        "Asthma is a chronic inflammatory disease of the airways.",
        "Symptoms include wheeze, shortness of breath, chest tightness, and cough.",
        "Diagnosis requires evidence of variable expiratory airflow limitation.",
        "Inhaled corticosteroids are the preferred controller therapy for asthma.",
        "SABA is the reliever of choice for acute symptoms.",
    ]],
    "metadatas": [[
        {
            "doc_name": "GINA-2026-Strategy-Report-WMS.pdf",
            "chunk_id": "g_1_0",
            "page_number": 1,
            "chunk_word_count": 10,
            "section_title": "Definition",
        },
        {
            "doc_name": "GINA-2026-Strategy-Report-WMS.pdf",
            "chunk_id": "g_2_0",
            "page_number": 2,
            "chunk_word_count": 12,
            "section_title": "Symptoms",
        },
        {
            "doc_name": "GINA-2026-Strategy-Report-WMS.pdf",
            "chunk_id": "g_3_0",
            "page_number": 3,
            "chunk_word_count": 11,
            "section_title": "Diagnosis",
        },
        {
            "doc_name": "GINA-2026-Strategy-Report-WMS.pdf",
            "chunk_id": "g_4_0",
            "page_number": 4,
            "chunk_word_count": 10,
            "section_title": "Treatment",
        },
        {
            "doc_name": "GINA-2026-Strategy-Report-WMS.pdf",
            "chunk_id": "g_5_0",
            "page_number": 5,
            "chunk_word_count": 9,
            "section_title": "Treatment",
        },
    ]],
    "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
}


def test_happy_path_definition_question_returns_sourced_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 1: definition question with passing grades produces a sourced answer."""
    _patch_settings(monkeypatch)
    _patch_verifier(monkeypatch)
    _patch_judge(monkeypatch)
    _patch_retrieval(monkeypatch, _RETRIEVED_FIVE)
    _patch_grades(monkeypatch, [[True, True, True, False, False]])
    _patch_answer(
        monkeypatch,
        (
            "Asthma is a chronic inflammatory airway disease.\n\n"
            "**Sources**\n"
            "- GINA-2026-Strategy-Report-WMS.pdf, p. 1"
        ),
    )

    pipeline = Pipeline(settings=Settings(chroma_path=tmp_path / "chroma"))
    result = pipeline.query("What is asthma?")

    assert result["route"] == "retrieve"
    assert "Asthma is a chronic inflammatory airway disease." in result["final_answer"]
    assert "**Sources**" in result["final_answer"]


def test_inhaler_query_returns_video_and_explanation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 2: inhaler-technique query routes directly to the video answer."""
    _patch_settings(monkeypatch)
    _patch_verifier(monkeypatch)
    _patch_judge(monkeypatch)

    pipeline = Pipeline(settings=Settings(chroma_path=tmp_path / "chroma"))
    result = pipeline.query("How do I use my inhaler?")

    assert result["route"] == "inhaler"
    assert result["final_answer"] is not None
    assert len(result["final_answer"]) > 0
    assert result["video_html"] is not None
    assert "youtube.com/embed/" in result["video_html"]
    assert "<iframe" in result["video_html"]


def test_insufficient_evidence_returns_safe_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 3: no relevant chunks triggers the safe fallback answer."""
    _patch_settings(monkeypatch)
    _patch_verifier(monkeypatch)
    _patch_judge(monkeypatch)
    _patch_retrieval(monkeypatch, _RETRIEVED_FIVE)
    _patch_grades(monkeypatch, [[False, False, False, False, False]])

    pipeline = Pipeline(settings=Settings(chroma_path=tmp_path / "chroma"))
    result = pipeline.query("What is the capital of France?")

    assert result["route"] == "retrieve"
    assert (
        result["final_answer"]
        == "The retrieved asthma sources do not contain enough information to answer this question reliably."
    )


def test_system_prompt_regression_lock_is_unchanged() -> None:
    """Scenario 4: the clinical system prompt has not drifted from the locked hash."""
    import hashlib

    expected = "182a1054909d4924cece001449c1f825fc2699b87c57b63eab23e97f9578f702"
    actual = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert actual == expected
