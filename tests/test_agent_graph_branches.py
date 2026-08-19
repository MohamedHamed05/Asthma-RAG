"""Tests for the iteration-2 graph branches.

Locks five behaviors added on top of the original graph:
1. The entry verifier refuses out-of-scope queries before any retrieval.
2. The judge replaces a dangerous answer with the fixed safety message.
3. The judge appends a Confidence/Reason footer to approved answers.
4. Drug-price queries route to the EXA search branch.
5. Tutorial (non-inhaler) queries route to the video-search branch.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from asthma_rag.config import Settings
from asthma_rag.llm.groq import GroqChat

_SAFETY_MESSAGE = (
    "For safety reasons, this system cannot provide an answer to this question. "
    "Please consult a qualified healthcare professional."
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
        {"doc_name": "GINA.pdf", "chunk_id": f"g_{i}_0", "page_number": i, "chunk_word_count": 10}
        for i in range(1, 6)
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


def _patch_verifier(monkeypatch: pytest.MonkeyPatch, in_scope: bool = True) -> None:
    """Stub the entry verifier so tests control scope without an LLM call."""
    from asthma_rag.agent import graph as graph_module

    monkeypatch.setattr(
        graph_module,
        "verifier_node",
        lambda _state: {"verifier_result": {"in_scope": in_scope, "reason": "test"}},
    )


def _patch_judge_noop(monkeypatch: pytest.MonkeyPatch) -> None:
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


class _SequenceJson:
    """complete_json stub returning one JSON payload per call."""

    def __init__(self, payloads: list[str]) -> None:
        self._payloads = list(payloads)
        self._index = 0

    def __call__(self, _messages: list[dict[str, Any]]) -> str:
        payload = self._payloads[self._index]
        self._index += 1
        return payload


def _patch_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub retrieve/rerank so no Chroma/Cohere calls are made."""
    from asthma_rag.agent import graph as graph_module

    monkeypatch.setattr(graph_module, "retrieve", lambda _query, _n=50: _RETRIEVED_FIVE)
    monkeypatch.setattr(
        graph_module, "rerank_results", lambda _q, retrieved, _top_n=5: retrieved
    )


def test_verifier_out_of_scope_returns_refusal_and_skips_rag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given an out-of-scope verdict, the graph refuses before any retrieval runs."""
    from asthma_rag.agent import graph as graph_module
    from asthma_rag.agent.graph import build_graph

    _patch_verifier(monkeypatch, in_scope=False)
    monkeypatch.setattr(
        graph_module,
        "verifier_node",
        lambda _state: {
            "verifier_result": {"in_scope": False, "reason": "geography question"}
        },
    )

    def _explode(_query: str, _n: int = 50) -> dict[str, Any]:
        raise AssertionError("retrieval must not run for out-of-scope queries")

    monkeypatch.setattr(graph_module, "retrieve", _explode)

    result = build_graph().invoke({"query": "What is the capital of France?"})

    assert result["route"] == "out_of_scope"
    assert "outside the scope" in result["final_answer"]
    assert result["refusal_reason"] == "geography question"


def test_judge_refuses_dangerous_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a dangerous generated answer, the judge replaces it with the safety message."""
    from asthma_rag.agent.graph import build_graph

    _patch_settings(monkeypatch)
    _patch_verifier(monkeypatch)
    _patch_retrieval(monkeypatch)

    def _complete(_self: GroqChat, _messages: list[dict[str, Any]], **_kwargs: Any) -> str:
        return "Take 40 puffs of your inhaler every day without seeing a doctor."

    monkeypatch.setattr(GroqChat, "complete", _complete)
    monkeypatch.setattr(
        GroqChat,
        "complete_json",
        _SequenceJson([
            json.dumps({"grades": [True, True, True, True, True]}),
            json.dumps({
                "verdict": "refuse",
                "reason": "unsafe dosage advice",
                "confidence": "High",
            }),
        ]),
    )

    result = build_graph().invoke({"query": "How much inhaler should I take?"})

    assert result["final_answer"] == _SAFETY_MESSAGE
    assert result["refusal_reason"] == "unsafe dosage advice"
    assert result["judge_result"]["verdict"] == "refuse"


def test_judge_appends_confidence_to_approved_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given an approved answer, the judge appends the confidence footer."""
    from asthma_rag.agent.graph import build_graph

    _patch_settings(monkeypatch)
    _patch_verifier(monkeypatch)
    _patch_retrieval(monkeypatch)

    def _complete(_self: GroqChat, _messages: list[dict[str, Any]], **_kwargs: Any) -> str:
        return "Asthma is a chronic inflammatory airway disease."

    monkeypatch.setattr(GroqChat, "complete", _complete)
    monkeypatch.setattr(
        GroqChat,
        "complete_json",
        _SequenceJson([
            json.dumps({"grades": [True, True, True, True, True]}),
            json.dumps({
                "verdict": "approve",
                "reason": "grounded in GINA definition",
                "confidence": "High",
            }),
        ]),
    )

    result = build_graph().invoke({"query": "What is asthma?"})

    assert "Asthma is a chronic inflammatory airway disease." in result["final_answer"]
    assert result["final_answer"].endswith(
        "**Confidence:** High\n**Reason:** grounded in GINA definition"
    )
    assert result["judge_result"]["confidence"] == "High"


def test_drug_price_query_routes_to_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a drug-price query, the graph routes to the EXA search branch."""
    from asthma_rag.agent import graph as graph_module
    from asthma_rag.agent.graph import build_graph

    _patch_verifier(monkeypatch)
    _patch_judge_noop(monkeypatch)
    monkeypatch.setattr(
        graph_module,
        "run_search_answer",
        lambda _state: {
            "final_answer": "Ventolin costs approximately 30-50 EGP in Egypt.",
            "route": "search",
            "search_results": [],
        },
    )

    result = build_graph().invoke({"query": "ventolin price in Egypt"})

    assert result["route"] == "search"
    assert "30-50 EGP" in result["final_answer"]


def test_video_query_routes_to_video_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a tutorial query, the graph routes to the video-search branch."""
    from asthma_rag.agent import graph as graph_module
    from asthma_rag.agent.graph import build_graph

    _patch_verifier(monkeypatch)
    _patch_judge_noop(monkeypatch)
    monkeypatch.setattr(
        graph_module,
        "run_video_search",
        lambda _state: {
            "final_answer": "Here is a tutorial video:\n\nNebulizer how-to",
            "video_html": '<iframe src="https://www.youtube.com/embed/abc123"></iframe>',
            "route": "video_search",
        },
    )

    result = build_graph().invoke({"query": "how to use a nebulizer"})

    assert result["route"] == "video_search"
    assert "youtube.com/embed/" in result["video_html"]
    assert "Nebulizer how-to" in result["final_answer"]
