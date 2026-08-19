"""LLM safety-judge node for the agentic RAG graph.

The judge is the last gate before an answer reaches the user. It receives the
original query, the retrieved clinical context, and the generated answer, and
returns one of two verdicts:

    * ``approve`` — answer is in-scope (asthma) AND not medically dangerous.
    * ``refuse``  — out-of-scope OR medically dangerous.

On ``refuse`` the node overwrites ``final_answer`` with a fixed safety message
and records the reason; on ``approve`` it appends a ``Confidence / Reason``
footer so the user can see the judge's reasoning. Any parse failure fails
OPEN (``approve`` with ``Low`` confidence) — refusing on parse error would let
a single malformed JSON payload take the whole pipeline down.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from asthma_rag.agent.state import AgentState, JudgeResult
from asthma_rag.config import get_settings
from asthma_rag.llm.groq import GroqChat
from asthma_rag.observability import add_metadata, score_trace
from asthma_rag.retrieval import format_retrieved_context

# Public type aliases describing the judge's structured output.
VerdictT = Literal["approve", "refuse"]
ConfidenceT = Literal["High", "Medium", "Low", "Insufficient evidence"]


# Fixed refusal message that replaces ``final_answer`` when the judge refuses.
_SAFETY_MESSAGE: str = (
    "For safety reasons, this system cannot provide an answer to this question. "
    "Please consult a qualified healthcare professional."
)

# Sentinel used when the agent reached the judge without retrieved context
# (e.g. the search/web route that bypasses Chroma).
_NO_CONTEXT_SENTINEL: str = "(no retrieved context — search route)"

# Allowed literal values, exposed for the parse helper and to keep the
# TypedDict honest in one place.
_ALLOWED_VERDICTS: frozenset[str] = frozenset({"approve", "refuse"})
_ALLOWED_CONFIDENCES: frozenset[str] = frozenset(
    {"High", "Medium", "Low", "Insufficient evidence"}
)


JUDGE_SYSTEM_PROMPT: str = """\
You are the safety judge for an asthma clinical-guideline RAG assistant.

You receive three sections in the user message:
  1. USER QUERY — the question the human asked.
  2. RETRIEVED CONTEXT — clinical document excerpts the assistant grounded on
     (or the literal sentinel "(no retrieved context — search route)" if the
     assistant reached you via the web-search route).
  3. GENERATED ANSWER — the text the assistant proposes to return.

Decide exactly one verdict:

  approve — when the GENERATED ANSWER is:
    * in-scope (the user asked something related to asthma or respiratory
      care), AND
    * not dangerous to user health. Specifically, it does NOT:
        - recommend unsafe medication, dosing, or device changes,
        - fabricate dosages, contraindications, or guideline citations,
        - discourage professional care for emergencies or red-flag symptoms
          (trouble breathing, cyanosis, severe wheeze, rescue-inhaler failure),
        - make a diagnosis or claim a specific patient has asthma.

  refuse — when the GENERATED ANSWER is out-of-scope OR medically dangerous
    per the criteria above.

Always set ``confidence`` to reflect how much evidence you had to decide:
  * "High"                 — clear-cut, well-grounded.
  * "Medium"               — some ambiguity but a defensible answer.
  * "Low"                  — borderline / partial evidence.
  * "Insufficient evidence"— you could not decide safely.

Return ONLY this JSON object (no markdown, no explanation):
  {"verdict": "approve" | "refuse", "reason": "<short reason>", "confidence": "High" | "Medium" | "Low" | "Insufficient evidence"}
"""


def _parse_judge(raw_json: str) -> JudgeResult:
    """Parse the LLM's JSON response into a typed :class:`JudgeResult`.

    Any malformed response (invalid JSON, wrong keys, unknown verdict /
    confidence values) returns the fail-open default:
    ``{"verdict": "approve", "reason": "judge response could not be parsed;
    failing open", "confidence": "Low"}``.
    """
    fail_open: JudgeResult = JudgeResult(
        verdict="approve",
        reason="judge response could not be parsed; failing open",
        confidence="Low",
    )
    try:
        data: Any = json.loads(raw_json)
    except json.JSONDecodeError:
        return fail_open
    if not isinstance(data, dict):
        return fail_open

    verdict = data.get("verdict")
    reason = data.get("reason")
    confidence = data.get("confidence")

    if not isinstance(verdict, str) or verdict not in _ALLOWED_VERDICTS:
        return fail_open
    if not isinstance(confidence, str) or confidence not in _ALLOWED_CONFIDENCES:
        return fail_open
    if not isinstance(reason, str):
        return fail_open

    return JudgeResult(
        verdict=verdict,  # type: ignore[typeddict-item]
        reason=reason,
        confidence=confidence,  # type: ignore[typeddict-item]
    )


def _build_user_message(state: AgentState) -> str:
    """Render the three-section user message for the judge call."""
    query: str = state.get("query", "")
    answer: str = state.get("final_answer", "")
    retrieved = state.get("retrieved")

    if retrieved:
        context_text: str = format_retrieved_context(retrieved, query)
    else:
        context_text = _NO_CONTEXT_SENTINEL

    return (
        "USER QUERY:\n"
        f"{query}\n\n"
        "RETRIEVED CONTEXT:\n"
        f"{context_text}\n\n"
        "GENERATED ANSWER:\n"
        f"{answer}"
    )


def judge_answer(
    state: AgentState, *, chat: GroqChat | None = None
) -> JudgeResult:
    """Run the LLM safety judge on the current ``state``.

    Builds a three-section user message (USER QUERY / RETRIEVED CONTEXT /
    GENERATED ANSWER), sends it to the configured Groq judge model, and
    parses the response. The ``chat`` argument is the optional injection
    point for tests; production callers leave it ``None`` and a
    ``GroqChat`` is constructed with the ``judge_model`` from settings.
    """
    settings = get_settings()
    chat_client: GroqChat = (
        chat if chat is not None else GroqChat(model=settings.judge_model)
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(state)},
    ]

    raw_response: str = chat_client.complete_json(messages)
    return _parse_judge(raw_response)


def judge_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: judge ``state["final_answer"]`` and update the state.

    Returns ``{}`` unchanged when there is no ``final_answer`` to judge (the
    graph cannot be blocked on a missing answer). On ``refuse`` the node
    replaces ``final_answer`` with a fixed safety message and surfaces
    ``refusal_reason``; on ``approve`` it appends a Confidence/Reason footer
    to the existing answer so the user sees the judge's reasoning.
    """
    final_answer = state.get("final_answer")
    if not final_answer:
        return {}

    result: JudgeResult = judge_answer(state)
    verdict: VerdictT = result["verdict"]
    reason: str = result["reason"]
    confidence: ConfidenceT = result["confidence"]

    if verdict == "refuse":
        score_trace("judge_verdict", "refuse")
        add_metadata({"refusal_reason": reason})
        return {
            "final_answer": _SAFETY_MESSAGE,
            "judge_result": result,
            "refusal_reason": reason,
        }

    score_trace("judge_confidence", confidence)
    add_metadata({"judge_reason": reason})
    footer = f"\n\n---\n**Confidence:** {confidence}\n**Reason:** {reason}"
    return {
        "final_answer": f"{final_answer}{footer}",
        "judge_result": result,
    }
