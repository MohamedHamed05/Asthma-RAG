"""Route a LangGraph agent state to one of: inhaler, video_search, search, retrieve.

Precedence (highest first):
    1. ``inhaler``    — inhaler-technique intent (delegated to ``inhaler_route``)
    2. ``video_search`` — tutorial/demonstration/nebulizer/spacer keywords
    3. ``search``     — drug-price signal AND asthma-drug keyword
    4. ``retrieve``   — default fallback to the RAG retrieval branch

The first match wins. Only the matching branch's ``route`` label is returned
from ``route_query``; downstream nodes own the payload building.
"""

from __future__ import annotations

import re

from asthma_rag.agent.inhaler import inhaler_route

# ---------------------------------------------------------------------------
# Keyword patterns (case-insensitive)
# ---------------------------------------------------------------------------

_VIDEO_KW: re.Pattern[str] = re.compile(
    r"\b(?:how\s*to|tutorial|demonstration|video|show\s+me|step\s+by\s+step|"
    r"nebulizer|nebuliser|spacer)\b",
    re.IGNORECASE,
)

_DRUG_PRICE_KW: re.Pattern[str] = re.compile(
    r"\b(?:price|cost|how\s+much|expensive|cheap|egypt|egp|pound|"
    r"جنيه|مصر)\b",
    re.IGNORECASE,
)

_ASTHMA_DRUG_KW: re.Pattern[str] = re.compile(
    r"\b(?:inhaler|ventolin|salbutamol|albuterol|budesonide|formoterol|"
    r"salmeterol|fluticasone|beclomethasone|montelukast|singulair|"
    r"steroid|prednisone|spiromax|seretide|symbicort|fostair)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route_query(state: dict) -> dict:
    """Pick a routing branch for *state* and return ``{"route": <label>}``.

    The inhaler branch's full payload is owned by :func:`inhaler_route`; here
    we only emit the route label and let the graph fan out from there.
    """
    query: str = state["query"]

    # 1. Inhaler technique always wins.
    inhaler_decision = inhaler_route(dict(state))
    if inhaler_decision.get("route") == "inhaler":
        return {"route": "inhaler"}

    # 2. Tutorial / demo / nebulizer / spacer keywords -> video search.
    if _VIDEO_KW.search(query):
        return {"route": "video_search"}

    # 3. Drug-price signal AND an asthma-drug keyword -> web search.
    if _DRUG_PRICE_KW.search(query) and _ASTHMA_DRUG_KW.search(query):
        return {"route": "search"}

    # 4. Default: RAG retrieval.
    return {"route": "retrieve"}
