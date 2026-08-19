"""LangGraph StateGraph wiring for the agentic asthma RAG pipeline.

Flow (iteration 2)::

    verify -> (out_of_scope | route)
    route  -> (answer_with_video | search_answer | video_search | retrieve)
    retrieve -> grade -> (generate_answer | rewrite_question | safe_fallback)
    rewrite_question -> retrieve
    generate_answer / answer_with_video / search_answer / video_search -> judge -> END
    safe_fallback / out_of_scope -> END

The verifier gates clearly unrelated queries before any retrieval; the judge
safety-checks every produced answer (the already-safe fallback and the
out-of-scope refusal bypass it) and attaches a confidence label. At most one
rewrite is allowed; the ``grade`` conditional edge checks ``rewrite_count`` to
prevent infinite loops.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from asthma_rag.agent.grader import GradeResult, grade_documents
from asthma_rag.agent.inhaler import inhaler_route
from asthma_rag.agent.judge import judge_node
from asthma_rag.agent.rewriter import MAX_REWRITES, rewrite_question
from asthma_rag.agent.search_answer import search_answer_node as run_search_answer
from asthma_rag.agent.search_answer import video_search_node as run_video_search
from asthma_rag.agent.search_route import route_query
from asthma_rag.agent.state import AgentState
from asthma_rag.agent.verifier import verifier_node
from asthma_rag.llm.groq import GroqChat
from asthma_rag.observability import traced
from asthma_rag.prompts import SYSTEM_PROMPT
from asthma_rag.retrieval import (
    RerankedResult,
    format_retrieved_context,
    rerank_results,
    retrieve,
)

_OUT_OF_SCOPE_MESSAGE = (
    "This question is outside the scope of this asthma-focused assistant. "
    "Please ask about asthma, its symptoms, diagnosis, treatment, "
    "medications, or inhaler technique."
)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@traced("verify", as_type="guardrail")
def verify_graph_node(state: AgentState) -> dict[str, Any]:
    """Classify the query as in-scope or out-of-scope before any work."""
    return verifier_node(state)


def out_of_scope_node(state: AgentState) -> dict[str, Any]:
    """Terminate out-of-scope queries with a clear refusal message."""
    verifier_result = state.get("verifier_result") or {}
    reason = verifier_result.get("reason", "outside the asthma scope")
    return {
        "final_answer": _OUT_OF_SCOPE_MESSAGE,
        "route": "out_of_scope",
        "refusal_reason": reason,
    }


def route_node(state: AgentState) -> dict[str, Any]:
    """Route the query to the inhaler, search, video, or retrieval branch."""
    decision = route_query(dict(state))
    if decision.get("route") == "inhaler":
        # route_query only labels the branch; the inhaler branch needs its
        # full payload (explanation + video embed), which inhaler_route builds.
        return inhaler_route(dict(state))
    return decision


@traced("retrieve", as_type="retriever")
def retrieve_node(state: AgentState) -> dict[str, Any]:
    """Retrieve candidate chunks from Chroma and rerank them with Cohere."""
    query = state["query"]
    raw = retrieve(query)
    reranked = rerank_results(query, raw)
    return {"retrieved": reranked}


@traced("grade", as_type="evaluator")
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


@traced("rewrite_question", as_type="span")
def rewrite_question_node(state: AgentState) -> dict[str, Any]:
    """Rewrite the query for more specific retrieval."""
    return rewrite_question(dict(state))


@traced("generate_answer", as_type="generation")
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


@traced("search_answer", as_type="generation")
def search_answer_graph_node(state: AgentState) -> dict[str, Any]:
    """Answer drug-price questions from live EXA search results."""
    return run_search_answer(state)


@traced("video_search", as_type="retriever")
def video_search_graph_node(state: AgentState) -> dict[str, Any]:
    """Find and embed a how-to video for the query."""
    return run_video_search(state)


def safe_fallback_node(state: AgentState) -> dict[str, Any]:
    """Return the clinical-safe fallback when evidence is insufficient."""
    return {
        "final_answer": (
            "The retrieved asthma sources do not contain enough information "
            "to answer this question reliably."
        )
    }


@traced("judge", as_type="evaluator")
def judge_graph_node(state: AgentState) -> dict[str, Any]:
    """Safety-check the final answer and attach a confidence label."""
    return judge_node(state)


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------


def verify_condition(state: AgentState) -> Literal["out_of_scope", "route"]:
    """After verification, choose the refusal branch or the routing branch.

    A missing verifier result is treated as in-scope (fail-open).
    """
    result = state.get("verifier_result")
    if result is not None and not result["in_scope"]:
        return "out_of_scope"
    return "route"


def route_condition(
    state: AgentState,
) -> Literal["answer_with_video", "search_answer", "video_search", "retrieve"]:
    """After routing, choose the branch matching the detected route label."""
    route = state.get("route")
    if route == "inhaler":
        return "answer_with_video"
    if route == "search":
        return "search_answer"
    if route == "video_search":
        return "video_search"
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

    graph.add_node("verify", verify_graph_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("answer_with_video", answer_with_video_node)
    graph.add_node("search_answer", search_answer_graph_node)
    graph.add_node("video_search", video_search_graph_node)
    graph.add_node("safe_fallback", safe_fallback_node)
    graph.add_node("judge", judge_graph_node)

    graph.set_entry_point("verify")
    graph.add_conditional_edges("verify", verify_condition)
    graph.add_conditional_edges("route", route_condition)
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", grade_condition)
    graph.add_edge("rewrite_question", "retrieve")
    graph.add_edge("generate_answer", "judge")
    graph.add_edge("answer_with_video", "judge")
    graph.add_edge("search_answer", "judge")
    graph.add_edge("video_search", "judge")
    graph.add_edge("judge", END)
    graph.add_edge("safe_fallback", END)
    graph.add_edge("out_of_scope", END)

    return graph.compile()
