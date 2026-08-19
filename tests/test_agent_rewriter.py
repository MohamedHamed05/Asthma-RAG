"""Tests for the agent query rewriter node.

Locks the contract of ``asthma_rag.agent.rewriter``: a fresh query is sent to
``GroqChat.complete`` and the state is updated with the rewritten text, while a
second pass (``rewrite_count >= 1``) short-circuits to keep the query unchanged
so the graph cannot loop forever. ``GroqChat.complete`` is patched — no network
or credentials are involved.
"""

from __future__ import annotations

import pytest

from asthma_rag.agent.rewriter import rewrite_question
from asthma_rag.llm.groq import GroqChat


class _StubComplete:
    """Callable stub that records messages and returns a canned rewrite."""

    def __init__(self, rewritten: str) -> None:
        self._rewritten = rewritten
        self.messages: list[dict[str, str]] | None = None

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return self._rewritten


def _make_chat(monkeypatch: pytest.MonkeyPatch, rewritten: str) -> tuple[GroqChat, _StubComplete]:
    """Build a GroqChat with a patched ``complete`` (no network)."""
    chat = GroqChat(api_key="test-key")
    stub = _StubComplete(rewritten)
    monkeypatch.setattr(chat, "complete", stub)
    return chat, stub


def test_rewrite_question_rewrites_query_and_increments_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a fresh query, when rewriting, then the rewritten text is stored and rewrite_count increments."""
    chat, stub = _make_chat(monkeypatch, "rewritten asthma question")
    state = {"query": "what is asthma?"}

    result = rewrite_question(state, llm=chat)

    assert result["query"] == "rewritten asthma question"
    assert result["rewrite_count"] == 1
    assert stub.messages is not None
    assert stub.messages[1]["content"] == "what is asthma?"


def test_rewrite_question_second_pass_returns_query_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given rewrite_count >= 1, when rewriting again, then the query is unchanged, count increments, and complete is not called."""
    chat, stub = _make_chat(monkeypatch, "should never be returned")
    state = {"query": "original question", "rewrite_count": 1}

    result = rewrite_question(state, llm=chat)

    assert result["query"] == "original question"
    assert result["rewrite_count"] == 2
    assert stub.messages is None


def test_rewrite_question_defaults_missing_count_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a state without rewrite_count, when rewriting, then the first pass proceeds and count becomes 1."""
    chat, _ = _make_chat(monkeypatch, "rewritten")

    result = rewrite_question({"query": "query"}, llm=chat)

    assert result["query"] == "rewritten"
    assert result["rewrite_count"] == 1
