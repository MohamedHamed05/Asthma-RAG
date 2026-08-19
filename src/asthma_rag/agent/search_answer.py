"""EXA-backed search nodes for the agentic asthma RAG graph.

``search_answer_node`` answers asthma drug price questions (Egypt-localized)
from live web results; ``video_search_node`` finds and embeds how-to videos.
Both degrade gracefully to friendly messages when EXA is unconfigured or
failing, and both accept an injected :class:`SearchClient` for tests.
"""

from __future__ import annotations

import logging

from asthma_rag.agent.state import AgentState
from asthma_rag.llm.groq import GroqChat
from asthma_rag.ui.video import build_youtube_embed
from asthma_rag.websearch import (
    SearchClient,
    SearchConfigError,
    SearchHit,
    extract_youtube_id,
)

logger = logging.getLogger(__name__)

SEARCH_ANSWER_SYSTEM_PROMPT: str = """You are an asthma drug price assistant for patients and caregivers in Egypt.

You will receive web search results about asthma medications. Use ONLY the provided search results to answer.

Your answer must:
1. State the approximate price range in EGP (Egyptian pounds) when the results contain price information.
2. End with a Sources section listing each source URL with its published date.
3. Include the caveat that prices fluctuate and vary by pharmacy — verify with a local pharmacist.
4. If the results contain no price information, say so honestly rather than estimating.

Keep the answer concise and formatted in Markdown.
"""

_PRICE_SEARCH_UNAVAILABLE = (
    "Drug price search is currently unavailable. "
    "Please set EXA_API_KEY to enable it."
)
_PRICE_SEARCH_FAILED = (
    "Drug price search is currently unavailable. Please try again later."
)
_VIDEO_SEARCH_UNAVAILABLE = (
    "Video search is currently unavailable. "
    "Please set EXA_API_KEY to enable it."
)
_VIDEO_SEARCH_FAILED = (
    "Video search is currently unavailable. Please try again later."
)
_VIDEO_NOT_FOUND = (
    "I couldn't find a video for that. "
    "Try rephrasing, or ask about inhaler technique specifically."
)


def _format_search_context(query: str, hits: list[SearchHit]) -> str:
    """Render the query and search hits as the LLM user message."""
    parts = [f"USER QUERY:\n{query}", "", "SEARCH RESULTS:"]
    if not hits:
        parts.append("(no results found)")
    for i, hit in enumerate(hits, start=1):
        parts.append(
            f"----- Result {i} -----\n"
            f"Title: {hit['title']}\n"
            f"URL: {hit['url']}\n"
            f"Published: {hit['published_date'] or 'Unknown'}\n"
            f"Text: {hit['text']}"
        )
    return "\n\n".join(parts)


def search_answer_node(
    state: AgentState, *, client: SearchClient | None = None
) -> dict:
    """Answer a drug-price query from live EXA search results."""
    query: str = state["query"]
    if client is None:
        client = SearchClient()

    try:
        hits = client.search_prices(query)
    except SearchConfigError:
        return {"final_answer": _PRICE_SEARCH_UNAVAILABLE, "route": "search"}
    except Exception:
        logger.warning("search_prices failed for query %r", query, exc_info=True)
        return {"final_answer": _PRICE_SEARCH_FAILED, "route": "search"}

    messages = [
        {"role": "system", "content": SEARCH_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": _format_search_context(query, hits)},
    ]
    answer = GroqChat().complete(messages)
    return {"final_answer": answer, "route": "search", "search_results": hits}


def video_search_node(
    state: AgentState, *, client: SearchClient | None = None
) -> dict:
    """Find a how-to video and return it as an embedded YouTube iframe."""
    query: str = state["query"]
    if client is None:
        client = SearchClient()

    try:
        hits = client.search_videos(query)
    except SearchConfigError:
        return {"final_answer": _VIDEO_SEARCH_UNAVAILABLE, "route": "video_search"}
    except Exception:
        logger.warning("search_videos failed for query %r", query, exc_info=True)
        return {"final_answer": _VIDEO_SEARCH_FAILED, "route": "video_search"}

    for hit in hits:
        video_id = extract_youtube_id(hit["url"])
        if video_id:
            return {
                "final_answer": "Here is a tutorial video:\n\n" + hit["title"],
                "video_html": build_youtube_embed(video_id),
                "route": "video_search",
            }
    return {"final_answer": _VIDEO_NOT_FOUND, "route": "video_search"}
