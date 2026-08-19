"""Document-grader node for the agentic RAG graph.

Grades each of the top-5 retrieved chunks for relevance to the user query and
produces a pass / rewrite / insufficient decision for the LangGraph state.
"""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from asthma_rag.llm.groq import GroqChat
from asthma_rag.prompts import SYSTEM_PROMPT


class GradeResult(TypedDict):
    """Result of grading the retrieved chunks for query relevance."""

    decision: Literal["pass", "rewrite", "insufficient"]
    graded: list[bool]


_GRADING_PROMPT: str = """You are grading retrieved clinical document chunks for relevance to a user query.

For each chunk below, decide whether it contains information relevant to answering the query.
Return ONLY a JSON object with the key "grades" containing a list of booleans (true for relevant, false for not relevant),
one per chunk, in the same order as the chunks. Do not include any explanation or markdown.

User query: {query}

Chunks to grade:
{chunks}
"""


def _build_grader_prompt(query: str, chunks: list[str]) -> str:
    """Format the grading prompt with the query and numbered chunks."""
    numbered = "\n\n".join(
        f"[{i + 1}] {chunk}" for i, chunk in enumerate(chunks)
    )
    return _GRADING_PROMPT.format(query=query, chunks=numbered)


def _parse_grades(raw_json: str, expected_count: int) -> list[bool]:
    """Parse the LLM JSON response into a list of booleans.

    Falls back to all False if the response is malformed or the wrong length.
    """
    try:
        data: dict[str, Any] = json.loads(raw_json)
        grades = data.get("grades", [])
        if not isinstance(grades, list) or len(grades) != expected_count:
            return [False] * expected_count
        return [bool(g) for g in grades]
    except json.JSONDecodeError:
        return [False] * expected_count


def _decide(grades: list[bool]) -> Literal["pass", "rewrite", "insufficient"]:
    """Map a list of relevance grades to a routing decision."""
    relevant = sum(grades)
    if relevant >= 3:
        return "pass"
    if relevant >= 1:
        return "rewrite"
    return "insufficient"


def grade_documents(state: dict) -> dict:
    """Grade the top-5 retrieved chunks and return an updated state dict.

    Expects ``state["query"]`` (str) and ``state["retrieved"]`` (Chroma-style
    result dict with ``documents`` / ``metadatas`` / ``distances``). Returns a
    dict containing ``grade_result`` and the original ``retrieved`` payload so
    downstream nodes can reuse the context.
    """
    query: str = state["query"]
    retrieved: dict = state["retrieved"]
    chunks: list[str] = retrieved["documents"][0][:5]

    prompt = _build_grader_prompt(query, chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    chat = GroqChat()
    raw_response = chat.complete_json(messages)
    grades = _parse_grades(raw_response, expected_count=len(chunks))
    decision = _decide(grades)

    return {
        "grade_result": GradeResult(decision=decision, graded=grades),
        "retrieved": retrieved,
    }
