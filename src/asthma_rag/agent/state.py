"""LangGraph state definition for the agentic RAG pipeline.

The state is intentionally small: the query plus the evidence and decisions the
agent accumulates while moving through the graph.
"""

from __future__ import annotations

from typing import Any, TypedDict

from asthma_rag.agent.grader import GradeResult


class AgentState(TypedDict, total=False):
    """Mutable state carried through the asthma RAG LangGraph agent.

    All keys except ``query`` are optional on entry; the graph fills them in as
    nodes execute. Using ``total=False`` lets the initial invoke carry only the
    user query while still giving nodes a typed view of the state.
    """

    query: str
    retrieved: dict[str, Any] | None
    graded: list[bool] | None
    rewrite_count: int
    route: str | None
    final_answer: str | None
    video_html: str | None
    explanation: str | None
    grade_result: GradeResult | None
