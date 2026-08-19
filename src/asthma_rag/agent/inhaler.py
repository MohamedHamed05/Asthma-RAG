"""Inhaler intent detection and payload building for the LangGraph agent.

Detects whether a user query is asking about inhaler *technique*
(how to use, demonstration, etc.) and builds a payload with a clinical-safe
explanation and an embedded demonstration video.
"""

from __future__ import annotations

import re

from asthma_rag.ui.video import INHALER_VIDEO_ID, build_youtube_embed

# ---------------------------------------------------------------------------
# Keyword patterns (case-insensitive, word-boundary anchored)
# ---------------------------------------------------------------------------

INHALER_KEYWORDS: re.Pattern[str] = re.compile(
    r"\b(?:inhaler|puffer|mdi|spacer|nebulizer|nebuliser|neb|dry\s+powder|dpi)\b",
    re.IGNORECASE,
)

TECHNIQUE_KEYWORDS: re.Pattern[str] = re.compile(
    r"\b(?:use|using|how\s*to|technique|demonstration|demo|step|steps|correctly|properly)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Negation detection
# ---------------------------------------------------------------------------

_NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "don't", "doesn't", "won't", "can't", "shouldn't",
    "never", "cannot", "nor",
})

_TWO_WORD_NEGATIONS: frozenset[str] = frozenset({
    "do not", "does not", "will not", "should not",
})


def _is_negated_technique(query: str, technique_match: re.Match[str]) -> bool:  # type: ignore[type-arg]
    """Return ``True`` if the technique keyword is directly preceded by a negation."""
    before = query[: technique_match.start()].rstrip()
    words = before.split()
    if not words:
        return False
    last_word = words[-1].lower()
    if last_word in _NEGATION_WORDS:
        return True
    if len(words) >= 2:
        two_words = f"{words[-2].lower()} {last_word}"
        if two_words in _TWO_WORD_NEGATIONS:
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_inhaler_intent(query: str) -> bool:
    """Return ``True`` if *query* contains both an inhaler keyword and a technique keyword.

    A technique keyword that is directly preceded by a negation word (e.g.
    "not use") is not counted as a technique signal.
    """
    if not INHALER_KEYWORDS.search(query):
        return False

    for m in TECHNIQUE_KEYWORDS.finditer(query):
        if not _is_negated_technique(query, m):
            return True
    return False


def build_inhaler_payload(query: str) -> dict[str, str]:
    """Build explanation + video embed payload for an inhaler technique query.

    Returns:
        ``{"explanation": str, "video_html": str}``
    """
    explanation = (
        "Here is a demonstration video and key steps for using an inhaler correctly."
    )
    video_html = build_youtube_embed(INHALER_VIDEO_ID)
    return {"explanation": explanation, "video_html": video_html}


def inhaler_route(state: dict) -> dict:
    """Route a LangGraph agent state to the inhaler branch or retrieval.

    If the query expresses inhaler-technique intent, returns a payload with
    the explanation and video embed. Otherwise falls through to retrieval.

    Returns:
        ``{"route": "inhaler", "explanation": str, "video_html": str}`` when
        intent is detected, else ``{"route": "retrieve"}``.
    """
    if not detect_inhaler_intent(state["query"]):
        return {"route": "retrieve"}
    payload = build_inhaler_payload(state["query"])
    return {"route": "inhaler", **payload}
