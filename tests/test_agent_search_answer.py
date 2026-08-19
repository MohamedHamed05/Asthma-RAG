"""Tests for search_answer_node and video_search_node.

Both nodes must:
- accept an injected ``SearchClient`` (no real EXA calls)
- patch ``GroqChat.complete`` for search_answer_node (no real Groq calls)
- fall back to a friendly message on SearchConfigError
- fall back to a friendly message on generic transport failure (logged)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from asthma_rag.agent import search_answer
from asthma_rag.agent.search_answer import (
    SEARCH_ANSWER_SYSTEM_PROMPT,
    search_answer_node,
    video_search_node,
)
from asthma_rag.llm.groq import GroqChat
from asthma_rag.websearch import SearchClient, SearchConfigError, SearchHit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hit(
    *,
    title: str = "T",
    url: str = "https://example.com/x",
    text: str = "body",
    published_date: str | None = "2025-01-02",
    score: float | None = 0.5,
) -> SearchHit:
    return {
        "title": title,
        "url": url,
        "text": text,
        "published_date": published_date,
        "score": score,
    }


class _StubTransport:
    """Stand-in for the EXA transport; configurable per-call."""

    def __init__(self, results: list[SearchHit] | None = None, exc: Exception | None = None) -> None:
        self._results = results or []
        self._exc = exc
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, **kwargs: Any) -> Any:
        self.calls.append({"query": query, **kwargs})
        if self._exc is not None:
            raise self._exc
        # Mimic the EXA response shape: SimpleNamespace with .results iterable
        from types import SimpleNamespace

        objs = [SimpleNamespace(**hit) for hit in self._results]
        return SimpleNamespace(results=objs)


def _patch_groq_complete(monkeypatch: pytest.MonkeyPatch, return_value: str) -> MagicMock:
    mock = MagicMock(return_value=return_value)
    monkeypatch.setattr(GroqChat, "complete", mock)
    return mock


# ---------------------------------------------------------------------------
# System prompt sanity
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    """The locked system prompt must reference the EGP currency and a sources section."""

    def test_system_prompt_mentions_egyptian_price(self) -> None:
        assert "EGP" in SEARCH_ANSWER_SYSTEM_PROMPT

    def test_system_prompt_includes_sources_section_hint(self) -> None:
        assert "Sources" in SEARCH_ANSWER_SYSTEM_PROMPT or "sources" in SEARCH_ANSWER_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# search_answer_node — happy path
# ---------------------------------------------------------------------------


class TestSearchAnswerNodeHappy:
    """search_answer_node must call the LLM with a structured user context and return hits + answer."""

    def test_returns_final_answer_route_and_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given 2 hits, when running, then final_answer/route/search_results are populated."""
        transport = _StubTransport(results=[
            _hit(title="Ventolin price", url="https://pharma.eg/v", text="250 EGP"),
            _hit(title="Symbicort Egypt", url="https://pharma.eg/s", text="500 EGP"),
        ])
        client = SearchClient(transport=transport)  # type: ignore[arg-type]
        mock = _patch_groq_complete(monkeypatch, "Here is the price range.")

        result = search_answer_node(
            {"query": "ventolin price egypt"}, client=client
        )

        assert result["final_answer"] == "Here is the price range."
        assert result["route"] == "search"
        assert isinstance(result["search_results"], list)
        assert len(result["search_results"]) == 2
        mock.assert_called_once()

    def test_user_message_contains_title_url_published_and_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given hits, when running, then the user message sent to the LLM contains all metadata."""
        transport = _StubTransport(results=[
            _hit(title="Ventolin", url="https://pharma.eg/v", text="250 EGP", published_date="2025-01-02"),
        ])
        client = SearchClient(transport=transport)  # type: ignore[arg-type]
        mock = _patch_groq_complete(monkeypatch, "answer")

        search_answer_node({"query": "ventolin"}, client=client)

        messages = mock.call_args.args[0]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SEARCH_ANSWER_SYSTEM_PROMPT
        user_msg = messages[1]
        assert user_msg["role"] == "user"
        assert "ventolin" in user_msg["content"]
        assert "Ventolin" in user_msg["content"]
        assert "https://pharma.eg/v" in user_msg["content"]
        assert "2025-01-02" in user_msg["content"]
        assert "250 EGP" in user_msg["content"]

    def test_no_results_still_calls_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given zero hits, when running, then the LLM is still called with the query and an empty context."""
        transport = _StubTransport(results=[])
        client = SearchClient(transport=transport)  # type: ignore[arg-type]
        mock = _patch_groq_complete(monkeypatch, "No price info found.")

        result = search_answer_node({"query": "ventolin"}, client=client)

        assert result["search_results"] == []
        assert result["final_answer"] == "No price info found."
        mock.assert_called_once()


# ---------------------------------------------------------------------------
# search_answer_node — error paths
# ---------------------------------------------------------------------------


class TestSearchAnswerNodeErrors:
    """search_answer_node must fall back gracefully on configuration / transport errors."""

    def test_search_config_error_returns_fallback_message(self) -> None:
        """Given SearchConfigError, when running, then a friendly fallback is returned (no LLM call)."""
        client = SearchClient(api_key="explicit", transport=None)  # type: ignore[arg-type]
        # Force the constructor's check to pass by injecting a transport that raises config error
        # Actually we want the SearchConfigError to be raised FROM search_prices. Use a raising transport.
        transport = _StubTransport(exc=SearchConfigError("EXA_API_KEY is not set."))
        client = SearchClient(transport=transport)  # type: ignore[arg-type]

        result = search_answer_node({"query": "ventolin price"}, client=client)

        assert result["route"] == "search"
        assert "EXA_API_KEY" in result["final_answer"]

    def test_generic_transport_failure_returns_fallback_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Given a non-SearchConfigError exception, when running, then a fallback is returned and a warning is logged."""
        transport = _StubTransport(exc=RuntimeError("EXA API is down"))
        client = SearchClient(transport=transport)  # type: ignore[arg-type]

        with caplog.at_level("WARNING"):
            result = search_answer_node({"query": "ventolin"}, client=client)

        assert result["route"] == "search"
        assert "unavailable" in result["final_answer"].lower()
        # Warning logged with exc_info
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("EXA API is down" in r.getMessage() or "search_prices" in r.getMessage() for r in warnings)

    def test_default_client_used_when_none_injected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given client=None, when running, then SearchClient() is constructed (and we patch it)."""
        stub_transport = _StubTransport(results=[_hit(text="250 EGP")])
        constructed_clients: list[SearchClient] = []

        def _factory() -> SearchClient:
            c = SearchClient(api_key="x", transport=stub_transport)  # type: ignore[arg-type]
            constructed_clients.append(c)
            return c

        monkeypatch.setattr(search_answer, "SearchClient", _factory)
        mock = _patch_groq_complete(monkeypatch, "answer")

        result = search_answer_node({"query": "ventolin"})

        assert len(constructed_clients) == 1
        assert result["route"] == "search"
        mock.assert_called_once()


# ---------------------------------------------------------------------------
# video_search_node — happy path
# ---------------------------------------------------------------------------


class TestVideoSearchNodeHappy:
    """video_search_node must return an embedded YouTube iframe for the top hit."""

    def test_returns_video_html_and_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given a hit, when running, then video_html contains a YouTube embed for that ID."""
        transport = _StubTransport(results=[
            _hit(title="Inhaler how to", url="https://www.youtube.com/watch?v=TuzCfpeieFA"),
        ])
        client = SearchClient(transport=transport)  # type: ignore[arg-type]

        result = video_search_node({"query": "inhaler demo"}, client=client)

        assert result["route"] == "video_search"
        assert "youtube.com/embed/TuzCfpeieFA" in result["video_html"]
        assert "Inhaler how to" in result["final_answer"]

    def test_falls_back_message_when_no_youtube_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given a hit with no extractable YouTube ID, when running, then a 'couldn't find' fallback is returned."""
        transport = _StubTransport(results=[
            _hit(title="Article only", url="https://example.com/x"),
        ])
        client = SearchClient(transport=transport)  # type: ignore[arg-type]

        result = video_search_node({"query": "inhaler demo"}, client=client)

        assert result["route"] == "video_search"
        assert "couldn't find" in result["final_answer"].lower() or "could not find" in result["final_answer"].lower()
        assert "video_html" not in result

    def test_falls_back_message_when_no_hits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given zero hits, when running, then a friendly fallback is returned."""
        transport = _StubTransport(results=[])
        client = SearchClient(transport=transport)  # type: ignore[arg-type]

        result = video_search_node({"query": "inhaler demo"}, client=client)

        assert result["route"] == "video_search"
        assert "couldn't find" in result["final_answer"].lower() or "could not find" in result["final_answer"].lower()


# ---------------------------------------------------------------------------
# video_search_node — error paths
# ---------------------------------------------------------------------------


class TestVideoSearchNodeErrors:
    """video_search_node must fall back on configuration / transport errors."""

    def test_search_config_error_returns_fallback(self) -> None:
        transport = _StubTransport(exc=SearchConfigError("EXA_API_KEY is not set."))
        client = SearchClient(transport=transport)  # type: ignore[arg-type]

        result = video_search_node({"query": "inhaler"}, client=client)

        assert result["route"] == "video_search"
        assert "EXA_API_KEY" in result["final_answer"]

    def test_generic_transport_failure_logs_warning_and_returns_fallback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        transport = _StubTransport(exc=RuntimeError("EXA 500"))
        client = SearchClient(transport=transport)  # type: ignore[arg-type]

        with caplog.at_level("WARNING"):
            result = video_search_node({"query": "inhaler"}, client=client)

        assert result["route"] == "video_search"
        assert "unavailable" in result["final_answer"].lower()
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) >= 1
