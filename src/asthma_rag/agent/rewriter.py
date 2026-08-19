"""Query rewriter node for the LangGraph retrieval agent.

Improves the specificity of a user query for clinical asthma guideline
retrieval. Rewriting is capped at a single pass — once ``rewrite_count``
reaches 1 the query is returned unchanged, so the graph cannot loop forever.
"""

from __future__ import annotations

from asthma_rag.llm.groq import GroqChat

# A single rewrite pass is enough; further passes risk drifting off-topic.
MAX_REWRITES = 1

REWRITE_SYSTEM_PROMPT = (
    "Rewrite the following question to be more specific for retrieving "
    "clinical asthma guideline content. Do not change the topic. "
    "Return only the rewritten question."
)


def rewrite_question(state: dict, llm: GroqChat | None = None) -> dict:
    """Rewrite ``state["query"]`` for more specific retrieval.

    Reads ``state["query"]`` and ``state.get("rewrite_count", 0)``. When the
    count is already at ``MAX_REWRITES`` the query is returned unchanged and
    the count incremented (loop guard). Otherwise the query is sent to
    ``GroqChat.complete`` and the state is updated with the rewritten text.

    Returns:
        A new state dict with ``"query"`` and ``"rewrite_count"`` set.
    """
    query = state["query"]
    rewrite_count = state.get("rewrite_count", 0)

    if rewrite_count >= MAX_REWRITES:
        return {"query": query, "rewrite_count": rewrite_count + 1}

    chat = llm if llm is not None else GroqChat()
    rewritten = chat.complete(
        [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
    )
    return {"query": rewritten, "rewrite_count": rewrite_count + 1}
