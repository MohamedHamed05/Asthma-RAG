"""In-scope query verifier for the agentic RAG graph.

Classifies whether a user query is within the asthma RAG pipeline's scope
before retrieval is attempted. The verifier is intentionally conservative:
anything plausibly asthma-related or a directly adjacent respiratory topic
(e.g. COPD overlap, wheeze, exercise-induced bronchoconstriction) is
in-scope; only clearly unrelated queries (weather, sports, coding, geography,
etc.) are out-of-scope. When the LLM response cannot be parsed, the verifier
fails open (in_scope=True) so the pipeline continues to serve the user rather
than refusing on a verifier error.
"""

from __future__ import annotations

import json
from typing import Any

from asthma_rag.agent.state import AgentState, VerifierResult
from asthma_rag.config import get_settings
from asthma_rag.llm.groq import GroqChat
from asthma_rag.observability import add_metadata, score_trace

__all__ = [
    "VERIFIER_SYSTEM_PROMPT",
    "VerifierResult",
    "verify_query",
    "verifier_node",
]


VERIFIER_SYSTEM_PROMPT: str = """You are a query-scope classifier for an asthma-focused clinical question-answering system grounded in authoritative guidelines (e.g. GINA, NAEPP, BTS/SIGN).

Decide whether the user's query is IN-SCOPE or OUT-OF-SCOPE for this asthma RAG pipeline.

IN-SCOPE topics include (but are not limited to):
- Asthma definition, pathophysiology, symptoms, and clinical presentation
- Asthma diagnosis, diagnostic testing, severity, and control
- Asthma triggers, exacerbations, prevention, and prognosis
- Asthma medications, including inhaled corticosteroids, biologics, SABA/LABA/LAMA, leukotriene modifiers, and oral steroids
- Inhaler technique and asthma action plans
- Asthma comorbidities and adjacent respiratory topics such as COPD overlap, wheeze, cough-variant asthma, exercise-induced bronchoconstriction, and allergic rhinitis when asthma-related
- Price of Asthma Related Medication

OUT-OF-SCOPE topics include clearly unrelated subjects, for example:
- Weather forecasts, sports scores, cooking, travel, finance
- Programming, geography, history, general trivia
- Non-respiratory medical questions with no asthma link (e.g. dermatology, orthopedics, psychiatry, oncology)

When in doubt, classify the query as IN-SCOPE.

You MUST return ONLY a JSON object of this exact shape, with no additional text, commentary, or markdown code fences:

{"in_scope": true|false, "reason": "<short one-sentence justification>"}

The "in_scope" field must be a JSON boolean (true or false). The "reason" field must be a short string explaining the decision.
"""


_FAIL_OPEN_REASON: str = "verifier response could not be parsed; failing open"


def _parse_verifier(raw_json: str) -> VerifierResult:
    """Parse the LLM JSON response into a ``VerifierResult``.

    Fails open (returns ``in_scope=True``) on any malformed response — invalid
    JSON, wrong top-level shape, missing or wrong-typed keys — so the pipeline
    continues to serve the user rather than refusing on a verifier error.
    """
    try:
        data: Any = json.loads(raw_json)
    except json.JSONDecodeError:
        return VerifierResult(in_scope=True, reason=_FAIL_OPEN_REASON)
    if not isinstance(data, dict):
        return VerifierResult(in_scope=True, reason=_FAIL_OPEN_REASON)
    in_scope = data.get("in_scope")
    reason = data.get("reason")
    if not isinstance(in_scope, bool):
        return VerifierResult(in_scope=True, reason=_FAIL_OPEN_REASON)
    if not isinstance(reason, str):
        return VerifierResult(in_scope=True, reason=_FAIL_OPEN_REASON)
    return VerifierResult(in_scope=in_scope, reason=reason)


def verify_query(
    state: AgentState, *, chat: GroqChat | None = None
) -> VerifierResult:
    """Classify the user query as in-scope or out-of-scope for the asthma pipeline.

    Expects ``state["query"]`` (str). Builds a two-message conversation (system
    + user) and calls ``GroqChat.complete_json`` either on an injected ``chat``
    (for tests) or on a freshly constructed ``GroqChat(model=verifier_model)``.
    Returns a ``VerifierResult``. Falls open on any parse failure.
    """
    query: str = state["query"]
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    if chat is None:
        chat = GroqChat(model=get_settings().verifier_model)
    raw_response = chat.complete_json(messages)
    return _parse_verifier(raw_response)


def verifier_node(state: AgentState) -> dict:
    """LangGraph node: classify the query and record an observability trace.

    Expects ``state["query"]`` (str). Returns ``{"verifier_result": VerifierResult}``
    and emits a ``verifier_in_scope`` score plus a ``verifier_reason`` metadata
    entry via the (guarded) observability layer — these are no-ops when
    Langfuse tracing is disabled.
    """
    result = verify_query(state)
    score_trace("verifier_in_scope", str(result["in_scope"]))
    add_metadata({"verifier_reason": result["reason"]})
    return {"verifier_result": result}
