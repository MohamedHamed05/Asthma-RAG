"""LangGraph StateGraph wiring for the agentic asthma RAG pipeline.

The graph has two top-level branches:

1. Inhaler-technique queries are detected by ``route`` and sent directly to
   ``answer_with_video``.
2. All other queries flow through retrieve → grade → (rewrite → retrieve)* →
   generate_answer or safe_fallback.

At most one rewrite is allowed; the ``grade`` conditional edge checks
``rewrite_count`` to prevent infinite loops.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from asthma_rag.agent.grader import GradeResult, grade_documents
from asthma_rag.agent.inhaler import inhaler_route
from asthma_rag.agent.rewriter import MAX_REWRITES, rewrite_question
from asthma_rag.agent.state import AgentState
from asthma_rag.llm.groq import GroqChat
from asthma_rag.prompts import SYSTEM_PROMPT
from asthma_rag.retrieval import (
    RerankedResult,
    format_retrieved_context,
    rerank_results,
    retrieve,
)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def route_node(state: AgentState) -> dict[str, Any]:
    """Route the query to the inhaler branch or the retrieval branch."""
    return inhaler_route(dict(state))


def retrieve_node(state: AgentState) -> dict[str, Any]:
    """Retrieve candidate chunks from Chroma and rerank them with Cohere."""
    query = state["query"]
    raw = retrieve(query)
    reranked = rerank_results(query, raw)
    return {"retrieved": reranked}


def grade_node(state: AgentState) -> dict[str, Any]:
    """Grade the top retrieved chunks and store the decision in state."""
    retrieved = state.get("retrieved")
    if retrieved is None:
        grade_result = GradeResult(decision="insufficient", graded=[])
        return {"grade_result": grade_result, "graded": [], "retrieved": None}

    result = grade_documents({"query": state["query"], "retrieved": retrieved})
    grade_result = result["grade_result"]
    return {
        "grade_result": grade_result,
        "graded": grade_result["graded"],
        "retrieved": result["retrieved"],
    }


def rewrite_question_node(state: AgentState) -> dict[str, Any]:
    """Rewrite the query for more specific retrieval."""
    return rewrite_question(dict(state))


def generate_answer_node(state: AgentState) -> dict[str, Any]:
    """Generate the final answer from the reranked context."""
    retrieved = state.get("retrieved")
    if retrieved is None:
        return {
            "final_answer": (
                "The retrieved asthma sources do not contain enough information "
                "to answer this question reliably."
            )
        }

    context = format_retrieved_context(cast(RerankedResult, retrieved), state["query"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    chat = GroqChat()
    answer = chat.complete(messages)
    return {"final_answer": answer}


def answer_with_video_node(state: AgentState) -> dict[str, Any]:
    """Return the inhaler explanation plus the embedded demonstration video."""
    return {
        "final_answer": state.get("explanation", ""),
        "video_html": state.get("video_html", ""),
    }


def safe_fallback_node(state: AgentState) -> dict[str, Any]:
    """Return the clinical-safe fallback when evidence is insufficient."""
    return {
        "final_answer": (
            "The retrieved asthma sources do not contain enough information "
            "to answer this question reliably."
        )
    }


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------


def route_condition(state: AgentState) -> Literal["answer_with_video", "retrieve"]:
    """After routing, choose the inhaler-video branch or the retrieval branch."""
    if state.get("route") == "inhaler":
        return "answer_with_video"
    return "retrieve"


def grade_condition(
    state: AgentState,
) -> Literal["generate_answer", "rewrite_question", "safe_fallback"]:
    """After grading, choose to answer, rewrite, or fall back safely.

    Rewrites are allowed at most once; a second ``rewrite`` decision is treated
    as insufficient evidence and routed to the safe fallback.
    """
    grade_result = state.get("grade_result")
    if grade_result is None:
        return "safe_fallback"

    decision = grade_result["decision"]
    if decision == "pass":
        return "generate_answer"
    if decision == "rewrite":
        rewrite_count = state.get("rewrite_count", 0)
        if rewrite_count < MAX_REWRITES:
            return "rewrite_question"
    return "safe_fallback"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph() -> CompiledStateGraph:
    """Build and compile the agentic RAG StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("answer_with_video", answer_with_video_node)
    graph.add_node("safe_fallback", safe_fallback_node)

    graph.set_entry_point("route")
    graph.add_conditional_edges("route", route_condition)
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", grade_condition)
    graph.add_edge("rewrite_question", "retrieve")
    graph.add_edge("generate_answer", END)
    graph.add_edge("answer_with_video", END)
    graph.add_edge("safe_fallback", END)

    return graph.compile()
