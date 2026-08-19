"""LangGraph state definition for the agentic RAG pipeline.

The state is intentionally small: the query plus the evidence and decisions the
agent accumulates while moving through the graph. The verifier/judge result
types live here (not in their node modules) because LangGraph resolves state
annotations at runtime via ``get_type_hints`` — the nodes import ``AgentState``
from this module, so defining the result types here keeps the dependency graph
acyclic while remaining importable everywhere.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from asthma_rag.agent.grader import GradeResult
from asthma_rag.websearch import SearchHit


class VerifierResult(TypedDict):
    """In-scope classification for a user query."""

    in_scope: bool
    reason: str


class JudgeResult(TypedDict):
    """Structured verdict returned by the LLM safety judge."""

    verdict: Literal["approve", "refuse"]
    reason: str
    confidence: Literal["High", "Medium", "Low", "Insufficient evidence"]


class AgentState(TypedDict, total=False):
    """Mutable state carried through the asthma RAG LangGraph agent.

    All keys except ``query`` are optional on entry; the graph fills them in as
    nodes execute. Using ``total=False`` lets the initial invoke carry only the
    user query while still giving nodes a typed view of the state.
    """

    query: str
    verifier_result: VerifierResult | None
    retrieved: dict[str, Any] | None
    graded: list[bool] | None
    rewrite_count: int
    route: str | None
    final_answer: str | None
    video_html: str | None
    explanation: str | None
    grade_result: GradeResult | None
    judge_result: JudgeResult | None
    refusal_reason: str | None
    search_results: list[SearchHit] | None
