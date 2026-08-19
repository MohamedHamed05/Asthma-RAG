"""Tests for the LLM safety-judge LangGraph node.

RED phase: these tests should all FAIL before src/asthma_rag/agent/judge.py exists.

The judge is the last gate before an answer reaches the user. It receives the
original query, the retrieved clinical context, and the generated answer, and
returns one of two verdicts:
    * approve  — answer is in-scope (asthma) AND not medically dangerous
    * refuse   — out-of-scope OR medically dangerous

On ``refuse`` the node overwrites ``final_answer`` with a fixed safety message.
On ``approve`` it appends a ``Confidence / Reason`` footer. Any parse failure
fails OPEN: ``approve`` with ``confidence="Low"`` and a "could not be parsed"
reason — refusing on parse error would let a single bad JSON payload take the
pipeline down.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from asthma_rag.config import Settings
from asthma_rag.llm.groq import GroqChat

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RETRIEVED: dict[str, Any] = {
    "documents": [[
        "Inhaled corticosteroids are the preferred controller therapy for asthma.",
        "SABA is the reliever of choice for acute symptoms.",
    ]],
    "metadatas": [[
        {"chunk_id": "j_1_0", "doc_name": "GINA-2024", "page_number": 42},
        {"chunk_id": "j_1_1", "doc_name": "GINA-2024", "page_number": 43},
    ]],
    "distances": [[0.1, 0.2]],
}

SAFE_ANSWER: str = (
    "Inhaled corticosteroids are the preferred controller therapy for asthma."
)
UNSAFE_ANSWER: str = (
    "Take 500 mg of prednisone twice a day for two weeks and skip your inhaler."
)


def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a fake Groq API key so GroqChat construction succeeds."""
    from asthma_rag.llm import groq as groq_module

    monkeypatch.setattr(
        groq_module,
        "get_settings",
        lambda: Settings(
            groq_api_key="test-key",
            groq_model="test-model",
            judge_model="test-judge-model",
        ),
    )


def _patch_complete_json(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> MagicMock:
    """Patch ``GroqChat.complete_json`` to return a controlled JSON payload."""
    mock = MagicMock(return_value=json.dumps(payload))
    monkeypatch.setattr(GroqChat, "complete_json", mock)
    return mock


def _patch_observability(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Replace ``score_trace`` and ``add_metadata`` on the judge module with mocks.

    The real ``.env`` has Langfuse keys, so we MUST not let the judge module's
    imports hit the network. The mocks live on the judge module's namespace
    because that is where the names were bound at import time.
    """
    score_mock = MagicMock(return_value=None)
    meta_mock = MagicMock(return_value=None)
    monkeypatch.setattr("asthma_rag.agent.judge.score_trace", score_mock)
    monkeypatch.setattr("asthma_rag.agent.judge.add_metadata", meta_mock)
    return score_mock, meta_mock


# ---------------------------------------------------------------------------
# judge_node: routing
# ---------------------------------------------------------------------------


class TestJudgeNodeRouting:
    """judge_node must skip work when there is no answer to judge."""

    def test_returns_empty_dict_when_final_answer_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a state without final_answer, when judging, then the state is unchanged."""
        _patch_settings(monkeypatch)
        _patch_observability(monkeypatch)
        # Even a real-looking payload must not be touched.
        mock = _patch_complete_json(
            monkeypatch, {"verdict": "approve", "reason": "x", "confidence": "High"}
        )

        from asthma_rag.agent.judge import judge_node

        result = judge_node({"query": "what is asthma?"})

        assert result == {}
        mock.assert_not_called()


# ---------------------------------------------------------------------------
# judge_node: approve path
# ---------------------------------------------------------------------------


class TestJudgeNodeApprove:
    """judge_node must append a Confidence/Reason footer on approve."""

    def test_appends_confidence_and_reason_footer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given approve with confidence Medium, when judging, then the footer is appended and judge_result is returned."""
        _patch_settings(monkeypatch)
        _patch_complete_json(
            monkeypatch,
            {"verdict": "approve", "reason": "in-scope and safe", "confidence": "Medium"},
        )
        score_mock, meta_mock = _patch_observability(monkeypatch)

        from asthma_rag.agent.judge import JudgeResult, judge_node

        result = judge_node(
            {
                "query": "What is asthma?",
                "retrieved": RETRIEVED,
                "final_answer": SAFE_ANSWER,
            }
        )

        assert result["final_answer"].startswith(SAFE_ANSWER)
        assert "**Confidence:** Medium" in result["final_answer"]
        assert "**Reason:** in-scope and safe" in result["final_answer"]

        judge_result: JudgeResult = result["judge_result"]
        assert judge_result["verdict"] == "approve"
        assert judge_result["confidence"] == "Medium"
        assert judge_result["reason"] == "in-scope and safe"

        assert "refusal_reason" not in result

        score_mock.assert_called_once_with("judge_confidence", "Medium")
        meta_mock.assert_called_once_with({"judge_reason": "in-scope and safe"})


# ---------------------------------------------------------------------------
# judge_node: refuse path
# ---------------------------------------------------------------------------


class TestJudgeNodeRefuse:
    """judge_node must replace final_answer on refuse and record the reason."""

    SAFETY_MSG = (
        "For safety reasons, this system cannot provide an answer to this question. "
        "Please consult a qualified healthcare professional."
    )

    def test_replaces_final_answer_with_safety_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a refuse verdict, when judging, then final_answer is the safety message and refusal_reason is set."""
        _patch_settings(monkeypatch)
        _patch_complete_json(
            monkeypatch,
            {"verdict": "refuse", "reason": "fabricated dosage", "confidence": "High"},
        )
        score_mock, meta_mock = _patch_observability(monkeypatch)

        from asthma_rag.agent.judge import judge_node

        result = judge_node(
            {
                "query": "How much prednisone?",
                "retrieved": RETRIEVED,
                "final_answer": UNSAFE_ANSWER,
            }
        )

        assert result["final_answer"] == self.SAFETY_MSG
        assert result["refusal_reason"] == "fabricated dosage"
        assert result["judge_result"]["verdict"] == "refuse"
        assert result["judge_result"]["reason"] == "fabricated dosage"

        score_mock.assert_called_once_with("judge_verdict", "refuse")
        meta_mock.assert_called_once_with({"refusal_reason": "fabricated dosage"})


# ---------------------------------------------------------------------------
# Parse failure / fail-open
# ---------------------------------------------------------------------------


class TestJudgeParseFailure:
    """Any parse error must fail OPEN (approve, Low)."""

    def test_invalid_json_fails_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a non-JSON LLM response, when parsing, then judge fails open (approve, Low, fixed reason)."""
        _patch_settings(monkeypatch)
        mock = MagicMock(return_value="not json at all")
        monkeypatch.setattr(GroqChat, "complete_json", mock)
        score_mock, meta_mock = _patch_observability(monkeypatch)

        from asthma_rag.agent.judge import JudgeResult, judge_node

        result = judge_node(
            {
                "query": "What is asthma?",
                "retrieved": RETRIEVED,
                "final_answer": SAFE_ANSWER,
            }
        )

        judge_result: JudgeResult = result["judge_result"]
        assert judge_result["verdict"] == "approve"
        assert judge_result["confidence"] == "Low"
        assert judge_result["reason"] == "judge response could not be parsed; failing open"

        # Footer still appended — we only refuse when the LLM EXPLICITLY refuses.
        assert "**Confidence:** Low" in result["final_answer"]
        assert "**Reason:** judge response could not be parsed; failing open" in result["final_answer"]
        assert "refusal_reason" not in result

        score_mock.assert_called_once_with("judge_confidence", "Low")
        meta_mock.assert_called_once_with({"judge_reason": "judge response could not be parsed; failing open"})

    def test_invalid_verdict_fails_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given an unknown verdict value, when parsing, then judge fails open (approve, Low)."""
        _patch_settings(monkeypatch)
        _patch_complete_json(
            monkeypatch,
            {"verdict": "maybe", "reason": "x", "confidence": "High"},
        )
        _patch_observability(monkeypatch)

        from asthma_rag.agent.judge import judge_node

        result = judge_node(
            {
                "query": "What is asthma?",
                "retrieved": RETRIEVED,
                "final_answer": SAFE_ANSWER,
            }
        )

        assert result["judge_result"]["verdict"] == "approve"
        assert result["judge_result"]["confidence"] == "Low"
        assert "failing open" in result["judge_result"]["reason"]

    def test_invalid_confidence_fails_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given an unknown confidence value, when parsing, then judge fails open (approve, Low)."""
        _patch_settings(monkeypatch)
        _patch_complete_json(
            monkeypatch,
            {"verdict": "approve", "reason": "x", "confidence": "VeryHigh"},
        )
        _patch_observability(monkeypatch)

        from asthma_rag.agent.judge import judge_node

        result = judge_node(
            {
                "query": "What is asthma?",
                "retrieved": RETRIEVED,
                "final_answer": SAFE_ANSWER,
            }
        )

        assert result["judge_result"]["verdict"] == "approve"
        assert result["judge_result"]["confidence"] == "Low"

    def test_missing_keys_fails_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given JSON missing required keys, when parsing, then judge fails open (approve, Low)."""
        _patch_settings(monkeypatch)
        _patch_complete_json(monkeypatch, {"verdict": "approve"})  # missing reason + confidence
        _patch_observability(monkeypatch)

        from asthma_rag.agent.judge import judge_node

        result = judge_node(
            {
                "query": "What is asthma?",
                "retrieved": RETRIEVED,
                "final_answer": SAFE_ANSWER,
            }
        )

        assert result["judge_result"]["verdict"] == "approve"
        assert result["judge_result"]["confidence"] == "Low"
        assert "failing open" in result["judge_result"]["reason"]


# ---------------------------------------------------------------------------
# judge_answer: prompt construction
# ---------------------------------------------------------------------------


class TestJudgeAnswerPrompt:
    """judge_answer must build a three-section user message."""

    def test_prompt_includes_query_context_and_answer_sections(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a state, when judging, then the user message contains USER QUERY / RETRIEVED CONTEXT / GENERATED ANSWER sections."""
        _patch_settings(monkeypatch)
        mock = _patch_complete_json(
            monkeypatch,
            {"verdict": "approve", "reason": "ok", "confidence": "High"},
        )

        from asthma_rag.agent.judge import judge_answer

        judge_answer(
            {
                "query": "What is the preferred controller therapy?",
                "retrieved": RETRIEVED,
                "final_answer": SAFE_ANSWER,
            }
        )

        assert mock.call_count == 1
        messages = mock.call_args.args[0]
        user_content = messages[-1]["content"]
        assert "USER QUERY" in user_content
        assert "What is the preferred controller therapy?" in user_content
        assert "RETRIEVED CONTEXT" in user_content
        # The retrieved chunk text must reach the judge via the formatted context.
        assert "Inhaled corticosteroids are the preferred controller therapy for asthma." in user_content
        assert "GENERATED ANSWER" in user_content
        assert SAFE_ANSWER in user_content

    def test_prompt_handles_missing_retrieved_with_search_route_sentinel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given no retrieved context, when judging, then the context section is the search-route sentinel."""
        _patch_settings(monkeypatch)
        mock = _patch_complete_json(
            monkeypatch,
            {"verdict": "approve", "reason": "ok", "confidence": "Medium"},
        )

        from asthma_rag.agent.judge import judge_answer

        judge_answer(
            {
                "query": "What is asthma?",
                "final_answer": SAFE_ANSWER,
            }
        )

        user_content = mock.call_args.args[0][-1]["content"]
        assert "(no retrieved context — search route)" in user_content

    def test_prompt_uses_injected_chat_when_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given an injected chat client, when judging, then judge_answer uses the injected client and not GroqChat() default."""
        _patch_settings(monkeypatch)

        from asthma_rag.agent.judge import judge_answer
        from asthma_rag.llm.groq import GroqChat as _GroqChat

        # Sentinel: if judge_answer ever calls the default GroqChat, our patch
        # here will be replaced by a MagicMock and the assertion below fails.
        calls = []

        def _fake_complete_json(self, messages, json_schema=None, **overrides):  # type: ignore[no-untyped-def]
            calls.append(messages)
            return json.dumps({"verdict": "approve", "reason": "ok", "confidence": "High"})

        monkeypatch.setattr(_GroqChat, "complete_json", _fake_complete_json)

        injected = GroqChat(api_key="test-key", model="test-judge-model")
        # Confirm the injected chat is used, not a fresh one.
        result = judge_answer(
            {
                "query": "What is asthma?",
                "retrieved": RETRIEVED,
                "final_answer": SAFE_ANSWER,
            },
            chat=injected,
        )

        assert len(calls) == 1
        assert result["verdict"] == "approve"

    def test_no_real_api_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given complete_json is patched, when judging, then the underlying Groq chat.completions.create is never called."""
        from asthma_rag.llm import groq as groq_module

        _patch_settings(monkeypatch)

        class _ExplodingCompletions:
            def create(self, **kwargs: Any) -> None:
                raise AssertionError("real Groq API must not be called")

        def _exploding_groq(api_key: str) -> Any:
            return SimpleNamespace(chat=SimpleNamespace(completions=_ExplodingCompletions()))

        monkeypatch.setattr(groq_module.groq, "Groq", _exploding_groq)
        _patch_complete_json(
            monkeypatch,
            {"verdict": "approve", "reason": "ok", "confidence": "High"},
        )
        _patch_observability(monkeypatch)

        from asthma_rag.agent.judge import judge_node

        judge_node(
            {
                "query": "What is asthma?",
                "retrieved": RETRIEVED,
                "final_answer": SAFE_ANSWER,
            }
        )


# ---------------------------------------------------------------------------
# judge_answer: confidence Literal coverage
# ---------------------------------------------------------------------------


class TestJudgeAnswerConfidenceLevels:
    """All four confidence literals must round-trip through the prompt."""

    @pytest.mark.parametrize("confidence", ["High", "Medium", "Low", "Insufficient evidence"])
    def test_confidence_literal_round_trip(
        self, monkeypatch: pytest.MonkeyPatch, confidence: str
    ) -> None:
        """Given any allowed confidence value, when parsing, then it is preserved exactly."""
        _patch_settings(monkeypatch)
        _patch_complete_json(
            monkeypatch,
            {"verdict": "approve", "reason": "ok", "confidence": confidence},
        )
        _patch_observability(monkeypatch)

        from asthma_rag.agent.judge import judge_answer

        result = judge_answer(
            {
                "query": "What is asthma?",
                "retrieved": RETRIEVED,
                "final_answer": SAFE_ANSWER,
            }
        )

        assert result["confidence"] == confidence


# ---------------------------------------------------------------------------
# judge_node: defaults when final_answer is missing
# ---------------------------------------------------------------------------


class TestJudgeNodeDefaults:
    """judge_node must not construct a chat client when there is nothing to judge."""

    def test_does_not_construct_chat_when_no_final_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given no final_answer, when judging, then GroqChat is never instantiated."""
        _patch_settings(monkeypatch)
        _patch_observability(monkeypatch)

        from asthma_rag.agent.judge import judge_node
        from asthma_rag.llm.groq import GroqChat as _GroqChat

        constructed: list[_GroqChat] = []
        original_init = _GroqChat.__init__

        def _tracking_init(self, *args: Any, **kwargs: Any) -> None:
            constructed.append(self)
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(_GroqChat, "__init__", _tracking_init)

        result = judge_node({"query": "what is asthma?"})

        assert result == {}
        assert constructed == []
