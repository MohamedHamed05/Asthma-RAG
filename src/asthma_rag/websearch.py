"""EXA web-search client for the asthma RAG pipeline.

Thin, typed wrapper around the ``exa-py`` SDK that supports constructor
injection of an arbitrary ``SearchTransport`` (so tests can run without
hitting the EXA API) and falls back to building a real ``Exa`` client from
the configured ``EXA_API_KEY``. Used by the LangGraph search / video nodes.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, TypedDict

from asthma_rag.config import get_settings


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class SearchConfigError(RuntimeError):
    """Raised when the EXA client cannot be configured (e.g. missing API key)."""


class SearchHit(TypedDict):
    """One web-search result, normalized for downstream consumers."""

    title: str
    url: str
    text: str
    published_date: str | None
    score: float | None


class SearchTransport(Protocol):
    """Structural contract for the underlying search backend (allows test fakes)."""

    def search(self, query: str, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# YouTube ID extraction
# ---------------------------------------------------------------------------

# Matches the 11-char video ID in watch?v=, youtu.be/ or shorts/ URLs.
YOUTUBE_ID_RE: re.Pattern[str] = re.compile(
    r"(?:v=|youtu\.be/|shorts/)([\w-]{11})"
)


def extract_youtube_id(url: str) -> str | None:
    """Return the 11-char YouTube video ID embedded in *url*, or ``None`` if absent."""
    if not url:
        return None
    match = YOUTUBE_ID_RE.search(url)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SearchClient:
    """Typed wrapper around the EXA search API with constructor injection."""

    def __init__(
        self,
        api_key: str | None = None,
        transport: SearchTransport | None = None,
    ) -> None:
        """Build the search client.

        When ``transport`` is provided it is used as-is (test path). Otherwise
        an ``Exa`` client is built from ``api_key`` (explicit) or
        ``Settings.exa_api_key``. Raises :class:`SearchConfigError` if the key
        is missing.
        """
        if transport is not None:
            self._transport: SearchTransport = transport
            return

        settings = get_settings()
        self._api_key = settings.exa_api_key if api_key is None else api_key
        if not self._api_key:
            raise SearchConfigError(
                "EXA_API_KEY is not set. "
                "Add it to your .env file or environment variables to use web search."
            )
        self._transport = _build_exa_client(self._api_key)

    def search_prices(self, query: str) -> list[SearchHit]:
        """Search EXA for asthma-drug price information, scoped to Egypt."""
        settings = get_settings()
        # Compute the cutoff date for freshness (max_age_hours ago)
        cutoff_date = datetime.now(timezone.utc) - timedelta(hours=settings.search_max_age_hours)
        start_published_date = cutoff_date.isoformat(timespec="seconds")  # e.g., "2026-08-19T00:00:00"

        response = self._transport.search(
            query=query,
            num_results=settings.search_max_results,
            user_location="EG",
            contents={"text": {"max_characters": 2000}},
            start_published_date=start_published_date,   # <-- freshness filter
        )
        return [_to_hit(r) for r in getattr(response, "results", [])]

    def search_videos(self, query: str) -> list[SearchHit]:
        """Search EXA for YouTube videos matching *query* (long-form only, not Shorts)."""
        settings = get_settings()
        response = self._transport.search(
            query=query,
            num_results=settings.search_max_results,
            include_domains=["youtube.com/watch"],
            exclude_domains=["youtube.com/shorts"],
            contents={"text": {"max_characters": 500}},
        )
        return [_to_hit(r) for r in getattr(response, "results", [])]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_exa_client(api_key: str) -> SearchTransport:
    """Lazily build a real ``Exa`` client. Kept as a module-level symbol so tests can patch it."""
    from exa_py import Exa  # local import so the dep is optional in tests

    return Exa(api_key=api_key)


def _to_hit(result: Any) -> SearchHit:
    """Map a raw EXA result object to a :class:`SearchHit` dict."""
    url: str = getattr(result, "url", "") or ""
    title: str | None = getattr(result, "title", None)
    text: str | None = getattr(result, "text", None)
    return SearchHit(
        title=title if title else url,
        url=url,
        text=text if text else "",
        published_date=getattr(result, "published_date", None),
        score=getattr(result, "score", None),
    )