"""Tests for the EXA web-search layer.

RED phase: these tests should FAIL before ``src/asthma_rag/websearch.py``
exists. The transport is injected so no real EXA calls are made.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from asthma_rag import websearch
from asthma_rag.config import Settings
from asthma_rag.websearch import (
    SearchClient,
    SearchConfigError,
    SearchHit,
    YOUTUBE_ID_RE,
    extract_youtube_id,
)


# ---------------------------------------------------------------------------
# Stubs for the EXA SDK response shape
# ---------------------------------------------------------------------------


def _make_exa_result(
    *,
    title: str | None = None,
    url: str = "https://example.com/x",
    text: str | None = "Some body text.",
    published_date: str | None = "2025-01-02",
    score: float | None = 0.42,
) -> SimpleNamespace:
    """Mimics an EXA result object (attr access, not dict)."""
    return SimpleNamespace(
        title=title,
        url=url,
        text=text,
        published_date=published_date,
        score=score,
    )


def _make_exa_response(results: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(results=results)


class _RecordingTransport:
    """Test transport that records the kwargs it received and returns canned results."""

    def __init__(self, response: SimpleNamespace) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append({"query": query, **kwargs})
        return self._response


def _patch_settings(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    """Provide deterministic settings; defaults match a happy search path."""
    base: dict[str, Any] = {
        "exa_api_key": "test-exa-key",
        "search_max_results": 5,
        "search_max_age_hours": 24,
    }
    base.update(overrides)
    monkeypatch.setattr(websearch, "get_settings", lambda: Settings(**base))


def _patch_exa_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the implementation from importing exa_py at all."""
    monkeypatch.setattr(websearch, "_build_exa_client", None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# extract_youtube_id + YOUTUBE_ID_RE
# ---------------------------------------------------------------------------


class TestExtractYoutubeId:
    """extract_youtube_id must return the 11-char video ID from common URL shapes."""

    def test_watch_url(self) -> None:
        assert extract_youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self) -> None:
        assert extract_youtube_id("https://youtu.be/TuzCfpeieFA") == "TuzCfpeieFA"

    def test_shorts_url(self) -> None:
        assert extract_youtube_id("https://www.youtube.com/shorts/TuzCfpeieFA") == "TuzCfpeieFA"

    def test_returns_none_for_non_youtube_url(self) -> None:
        assert extract_youtube_id("https://example.com/article") is None

    def test_returns_none_for_empty(self) -> None:
        assert extract_youtube_id("") is None

    def test_id_regex_matches_11_chars(self) -> None:
        match = YOUTUBE_ID_RE.search("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert match is not None
        assert match.group(1) == "dQw4w9WgXcQ"


# ---------------------------------------------------------------------------
# Configuration error
# ---------------------------------------------------------------------------


class TestConstruction:
    """SearchClient must validate the API key at construction time."""

    def test_missing_api_key_raises_config_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given an empty EXA key, when constructing SearchClient, then SearchConfigError is raised."""
        _patch_settings(monkeypatch, exa_api_key="")

        with pytest.raises(SearchConfigError, match="EXA_API_KEY"):
            SearchClient()

    def test_explicit_api_key_does_not_call_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given an explicit api_key, when constructing, then settings are not consulted."""
        transport = _RecordingTransport(_make_exa_response([]))
        calls: list[bool] = []

        def _explode() -> Settings:
            calls.append(True)
            raise AssertionError("settings must not be consulted")

        monkeypatch.setattr(websearch, "get_settings", _explode)

        client = SearchClient(api_key="explicit-key", transport=transport)

        assert calls == []
        assert client is not None

    def test_injected_transport_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given a transport, when constructing, then no EXA client is built."""
        _patch_settings(monkeypatch)
        transport = _RecordingTransport(_make_exa_response([]))

        client = SearchClient(transport=transport)

        assert client._transport is transport


# ---------------------------------------------------------------------------
# search_prices
# ---------------------------------------------------------------------------


class TestSearchPrices:
    """search_prices must query EXA with Egypt locale and map results to SearchHit."""

    def test_maps_results_to_search_hits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given 2 results, when search_prices, then both are mapped to SearchHit dicts."""
        _patch_settings(monkeypatch)
        response = _make_exa_response([
            _make_exa_result(title="Ventolin price in Egypt", url="https://pharma.eg/v", text="250 EGP"),
            _make_exa_result(title=None, url="https://pharma.eg/x", text=None, score=None),
        ])
        transport = _RecordingTransport(response)

        hits = SearchClient(transport=transport).search_prices("ventolin price egypt")

        assert len(hits) == 2
        first, second = hits
        assert isinstance(first, dict)
        assert first["title"] == "Ventolin price in Egypt"
        assert first["url"] == "https://pharma.eg/v"
        assert first["text"] == "250 EGP"
        assert first["published_date"] == "2025-01-02"
        assert first["score"] == 0.42
        # title=None -> url; text=None -> ""
        assert second["title"] == "https://pharma.eg/x"
        assert second["text"] == ""
        assert second["score"] is None

    def test_calls_transport_with_egypt_user_location_and_text_contents(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a transport, when search_prices, then Egypt user_location and text contents are sent."""
        _patch_settings(monkeypatch)
        transport = _RecordingTransport(_make_exa_response([]))

        SearchClient(transport=transport).search_prices("seretide price")

        assert transport.calls, "transport.search was not called"
        kwargs = transport.calls[0]
        assert kwargs["query"] == "seretide price"
        assert kwargs["user_location"] == "EG"
        assert kwargs["num_results"] == 5
        assert kwargs["max_age_hours"] == 24
        assert kwargs["contents"] == {"text": {"max_characters": 2000}}

    def test_uses_settings_max_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given a custom search_max_results setting, when search_prices, then num_results uses it."""
        _patch_settings(monkeypatch, search_max_results=7)
        transport = _RecordingTransport(_make_exa_response([]))

        SearchClient(transport=transport).search_prices("symbicort price")

        assert transport.calls[0]["num_results"] == 7

    def test_search_hit_typing(self) -> None:
        """SearchHit must be a TypedDict with the documented keys."""
        sample: SearchHit = {
            "title": "t",
            "url": "u",
            "text": "x",
            "published_date": "2025-01-01",
            "score": 0.1,
        }
        assert sample["title"] == "t"
        assert sample["url"] == "u"
        assert sample["text"] == "x"
        assert sample["published_date"] == "2025-01-01"
        assert sample["score"] == 0.1


# ---------------------------------------------------------------------------
# search_videos
# ---------------------------------------------------------------------------


class TestSearchVideos:
    """search_videos must query EXA with YouTube domain scoping and shorter text excerpts."""

    def test_calls_transport_with_youtube_domains(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given a transport, when search_videos, then YouTube domain scoping is sent."""
        _patch_settings(monkeypatch)
        transport = _RecordingTransport(_make_exa_response([]))

        SearchClient(transport=transport).search_videos("inhaler technique demo")

        kwargs = transport.calls[0]
        assert kwargs["query"] == "inhaler technique demo"
        assert kwargs["include_domains"] == ["youtube.com/watch"]
        assert kwargs["exclude_domains"] == ["youtube.com/shorts"]
        assert "user_location" not in kwargs
        assert kwargs["contents"] == {"text": {"max_characters": 500}}

    def test_maps_results_to_search_hits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given one result, when search_videos, then it is mapped to a SearchHit."""
        _patch_settings(monkeypatch)
        response = _make_exa_response([
            _make_exa_result(
                title="Inhaler how to",
                url="https://www.youtube.com/watch?v=TuzCfpeieFA",
                text="Step-by-step demo",
            ),
        ])
        transport = _RecordingTransport(response)

        hits = SearchClient(transport=transport).search_videos("inhaler demo")

        assert len(hits) == 1
        assert hits[0]["url"].endswith("TuzCfpeieFA")
