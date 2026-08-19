"""Tests for the document-grader LangGraph node.

RED phase: these tests should all FAIL before src/asthma_rag/agent/grader.py exists.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from asthma_rag.agent.grader import GradeResult, grade_documents
from asthma_rag.config import Settings
from asthma_rag.llm.groq import GroqChat

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RETRIEVED_FIVE: dict[str, Any] = {
    "documents": [[
        "Inhaled corticosteroids are the preferred controller therapy for asthma.",
        "Asthma is a chronic inflammatory disease of the airways.",
        "SABA is the reliever of choice for acute symptoms.",
        "Diabetes mellitus is a metabolic disorder of carbohydrate regulation.",
        "The capital of France is Paris.",
    ]],
    "metadatas": [[
        {"chunk_id": "g_1_0"},
        {"chunk_id": "g_2_0"},
        {"chunk_id": "g_3_0"},
        {"chunk_id": "g_4_0"},
        {"chunk_id": "g_5_0"},
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


def _patch_complete_json(
    monkeypatch: pytest.MonkeyPatch, grades: list[bool]
) -> MagicMock:
    """Patch GroqChat.complete_json to return a controlled JSON grade payload."""
    payload = json.dumps({"grades": grades})
    mock = MagicMock(return_value=payload)
    monkeypatch.setattr(GroqChat, "complete_json", mock)
    return mock


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------


class TestGradeDocuments:
    """grade_documents must grade the top-5 retrieved chunks and map grades to a decision."""

    def test_pass_when_three_or_more_relevant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given 3 of 5 chunks graded relevant, when grading, then decision is pass."""
        _patch_settings(monkeypatch)
        mock = _patch_complete_json(monkeypatch, [True, True, True, False, False])

        result = grade_documents(
            {"query": "What is the preferred asthma controller?", "retrieved": RETRIEVED_FIVE}
        )

        grade_result: GradeResult = result["grade_result"]
        assert grade_result["decision"] == "pass"
        assert grade_result["graded"] == [True, True, True, False, False]
        assert result["retrieved"] is RETRIEVED_FIVE
        mock.assert_called_once()

    def test_rewrite_when_one_or_two_relevant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given 2 of 5 chunks graded relevant, when grading, then decision is rewrite."""
        _patch_settings(monkeypatch)
        mock = _patch_complete_json(monkeypatch, [True, False, True, False, False])

        result = grade_documents(
            {"query": "What is the preferred asthma controller?", "retrieved": RETRIEVED_FIVE}
        )

        grade_result: GradeResult = result["grade_result"]
        assert grade_result["decision"] == "rewrite"
        assert grade_result["graded"] == [True, False, True, False, False]
        mock.assert_called_once()

    def test_insufficient_when_zero_relevant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given 0 of 5 chunks graded relevant, when grading, then decision is insufficient."""
        _patch_settings(monkeypatch)
        mock = _patch_complete_json(monkeypatch, [False, False, False, False, False])

        result = grade_documents(
            {"query": "What is the preferred asthma controller?", "retrieved": RETRIEVED_FIVE}
        )

        grade_result: GradeResult = result["grade_result"]
        assert grade_result["decision"] == "insufficient"
        assert grade_result["graded"] == [False, False, False, False, False]
        mock.assert_called_once()

    def test_grades_fewer_than_five_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given only 2 retrieved chunks, when grading, then both are graded and decision follows the count."""
        _patch_settings(monkeypatch)
        mock = _patch_complete_json(monkeypatch, [True, False])
        retrieved = {
            "documents": [[
                "Inhaled corticosteroids are the preferred controller therapy for asthma.",
                "The capital of France is Paris.",
            ]],
            "metadatas": [[{"chunk_id": "g_1_0"}, {"chunk_id": "g_5_0"}]],
            "distances": [[0.1, 0.5]],
        }

        result = grade_documents(
            {"query": "What is the preferred asthma controller?", "retrieved": retrieved}
        )

        grade_result: GradeResult = result["grade_result"]
        assert grade_result["decision"] == "rewrite"
        assert grade_result["graded"] == [True, False]
        assert result["retrieved"] is retrieved
        mock.assert_called_once()


# ---------------------------------------------------------------------------
# Prompt / API contract
# ---------------------------------------------------------------------------


class TestPromptAndApi:
    """grade_documents must build the right prompt and avoid real API calls."""

    def test_complete_json_receives_query_and_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a state, when grading, then complete_json receives the query and all top chunks in the prompt."""
        _patch_settings(monkeypatch)
        mock = _patch_complete_json(monkeypatch, [True, True, True, False, False])

        grade_documents(
            {"query": "What is the preferred asthma controller?", "retrieved": RETRIEVED_FIVE}
        )

        assert mock.call_count == 1
        messages = cast(list[dict[str, Any]], mock.call_args.args[0])
        content = messages[-1]["content"]
        assert "What is the preferred asthma controller?" in content
        for chunk in RETRIEVED_FIVE["documents"][0]:
            assert chunk in content

    def test_no_real_api_call_when_complete_json_is_patched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given complete_json is patched, when grading, then the underlying chat.completions.create is never called."""
        from asthma_rag.llm import groq as groq_module

        _patch_settings(monkeypatch)

        class _ExplodingCompletions:
            def create(self, **kwargs: Any) -> None:
                raise AssertionError("real Groq API must not be called")

        def _exploding_groq(api_key: str) -> Any:
            return SimpleNamespace(chat=SimpleNamespace(completions=_ExplodingCompletions()))

        monkeypatch.setattr(groq_module.groq, "Groq", _exploding_groq)
        _patch_complete_json(monkeypatch, [True, True, True, False, False])

        grade_documents(
            {"query": "What is asthma?", "retrieved": RETRIEVED_FIVE}
        )
