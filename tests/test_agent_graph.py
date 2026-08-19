"""Tests for the LangGraph agent wiring.

RED phase: these tests should FAIL before ``src/asthma_rag/agent/graph.py`` and
``src/asthma_rag/agent/state.py`` exist. They exercise the full agent graph
with stubbed retrieval, Groq, and Cohere so no real API calls are made.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from asthma_rag.config import Settings
from asthma_rag.llm.groq import GroqChat

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RETRIEVED_FIVE: dict[str, Any] = {
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
        },
        {
            "doc_name": "GINA-2026-Strategy-Report-WMS.pdf",
            "chunk_id": "g_2_0",
            "page_number": 2,
            "chunk_word_count": 12,
        },
        {
            "doc_name": "GINA-2026-Strategy-Report-WMS.pdf",
            "chunk_id": "g_3_0",
            "page_number": 3,
            "chunk_word_count": 11,
        },
        {
            "doc_name": "GINA-2026-Strategy-Report-WMS.pdf",
            "chunk_id": "g_4_0",
            "page_number": 4,
            "chunk_word_count": 10,
        },
        {
            "doc_name": "GINA-2026-Strategy-Report-WMS.pdf",
            "chunk_id": "g_5_0",
            "page_number": 5,
            "chunk_word_count": 9,
        },
    ]],
    "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
}


def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a fake Groq API key so GroqChat construction succeeds."""
    from asthma_rag.llm import groq as groq_module

    monkeypatch.setattr(
        groq_module,
        "get_settings",
        lambda: Settings(groq_api_key="test-key", groq_model="test-model"),
    )


def _patch_retrieval(monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]) -> None:
    """Replace retrieve/rerank_results in the graph module with fixed results."""
    from asthma_rag.agent import graph as graph_module

    monkeypatch.setattr(graph_module, "retrieve", lambda _query, _n=50: result)

    def _fake_rerank(
        _query: str, retrieved: dict[str, Any], _top_n: int = 5
    ) -> dict[str, Any]:
        return retrieved

    monkeypatch.setattr(graph_module, "rerank_results", _fake_rerank)


def _patch_grades(monkeypatch: pytest.MonkeyPatch, grades_sequence: list[list[bool]]) -> None:
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


def _patch_complete(monkeypatch: pytest.MonkeyPatch, response: str) -> None:
    """Patch GroqChat.complete to return a fixed response (rewrite + generate)."""
    def _complete(_self: GroqChat, _messages: list[dict[str, Any]], **_kwargs: Any) -> str:
        return response

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


# ---------------------------------------------------------------------------
# Inhaler branch
# ---------------------------------------------------------------------------


class TestInhalerBranch:
    """Inhaler-technique queries must route directly to the video answer."""

    def test_inhaler_query_routes_to_video_and_produces_iframe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given an inhaler technique query, when invoking the graph, then it produces a video iframe answer."""
        _patch_settings(monkeypatch)
        _patch_verifier(monkeypatch)
        _patch_judge(monkeypatch)
        from asthma_rag.agent.graph import build_graph

        graph = build_graph()
        result = graph.invoke({"query": "How do I use my inhaler?"})

        assert result["route"] == "inhaler"
        assert result["final_answer"] is not None
        assert len(result["final_answer"]) > 0
        assert result["video_html"] is not None
        assert "youtube.com/embed/" in result["video_html"]
        assert "<iframe" in result["video_html"]


# ---------------------------------------------------------------------------
# Retrieval branch
# ---------------------------------------------------------------------------


class TestRetrievalBranch:
    """General asthma questions must route through retrieve → grade → answer."""

    def test_definition_query_routes_through_retrieve_grade_generate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a definition query with passing grades, when invoking the graph, then it generates a sourced answer."""
        _patch_settings(monkeypatch)
        _patch_verifier(monkeypatch)
        _patch_judge(monkeypatch)
        _patch_retrieval(monkeypatch, RETRIEVED_FIVE)
        _patch_grades(monkeypatch, [[True, True, True, False, False]])
        _patch_complete(
            monkeypatch,
            "Asthma is a chronic inflammatory airway disease.\n\n**Sources**\n- GINA-2026-Strategy-Report-WMS.pdf, p. 1",
        )
        from asthma_rag.agent.graph import build_graph

        graph = build_graph()
        result = graph.invoke({"query": "What is asthma?"})

        assert result["route"] == "retrieve"
        assert result["retrieved"] is not None
        assert result["graded"] == [True, True, True, False, False]
        assert "Asthma is a chronic inflammatory airway disease." in result["final_answer"]
        assert "**Sources**" in result["final_answer"]

    def test_insufficient_query_routes_to_safe_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a query with zero relevant chunks, when invoking the graph, then it returns the safe fallback."""
        _patch_settings(monkeypatch)
        _patch_verifier(monkeypatch)
        _patch_judge(monkeypatch)
        _patch_retrieval(monkeypatch, RETRIEVED_FIVE)
        _patch_grades(monkeypatch, [[False, False, False, False, False]])
        from asthma_rag.agent.graph import build_graph

        graph = build_graph()
        result = graph.invoke({"query": "What is the capital of France?"})

        assert result["route"] == "retrieve"
        assert result["graded"] == [False, False, False, False, False]
        assert (
            result["final_answer"]
            == "The retrieved asthma sources do not contain enough information to answer this question reliably."
        )

    def test_rewrite_query_loops_back_then_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a query that grades rewrite then pass, when invoking the graph, then it rewrites once and generates an answer."""
        _patch_settings(monkeypatch)
        _patch_verifier(monkeypatch)
        _patch_judge(monkeypatch)
        _patch_retrieval(monkeypatch, RETRIEVED_FIVE)
        _patch_grades(monkeypatch, [
            [True, False, False, False, False],  # rewrite
            [True, True, True, False, False],  # pass after rewrite
        ])

        from unittest.mock import Mock

        mock_complete = Mock(
            side_effect=["What is asthma and how is it diagnosed?", "Final sourced answer."]
        )
        monkeypatch.setattr(GroqChat, "complete", mock_complete)
        from asthma_rag.agent.graph import build_graph

        graph = build_graph()
        result = graph.invoke({"query": "What is asthma?"})

        assert result["route"] == "retrieve"
        assert result["rewrite_count"] == 1
        assert result["query"] == "What is asthma and how is it diagnosed?"
        assert result["final_answer"] == "Final sourced answer."
