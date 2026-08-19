"""Tests for the in-scope query verifier LangGraph node.

RED phase: these tests FAIL before ``src/asthma_rag/agent/verifier.py`` exists.

The verifier classifies a user query as in-scope (asthma-related or a directly
adjacent respiratory topic) or out-of-scope (clearly unrelated). It is the first
node in the agent graph, before retrieval is attempted.

Conventions mirror ``tests/test_agent_grader.py``:
- patch ``GroqChat.complete_json`` to avoid real API calls,
- patch the verifier module's imported ``score_trace``/``add_metadata`` so the
  real Langfuse client (keys exist in ``.env``) is never reached,
- provide a fake ``Settings`` so ``GroqChat`` construction does not raise on a
  missing API key.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from asthma_rag.agent import verifier as verifier_module
from asthma_rag.agent.state import AgentState
from asthma_rag.agent.verifier import (
    VERIFIER_SYSTEM_PROMPT,
    VerifierResult,
    verify_query,
    verifier_node,
)
from asthma_rag.config import Settings
from asthma_rag.llm.groq import GroqChat


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable observability side-effects (Langfuse keys exist in .env).

    The verifier imports ``score_trace`` and ``add_metadata`` by name into its
    module namespace, so patching those names on the verifier module replaces
    the local bindings and guarantees no real Langfuse network call can happen.
    """
    monkeypatch.setattr(verifier_module, "score_trace", MagicMock())
    monkeypatch.setattr(verifier_module, "add_metadata", MagicMock())


@pytest.fixture
def patched_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a fake Groq API key so ``GroqChat`` construction succeeds."""
    from asthma_rag.llm import groq as groq_module

    monkeypatch.setattr(
        groq_module,
        "get_settings",
        lambda: Settings(groq_api_key="test-key", groq_model="test-model"),
    )


def _patch_complete_json(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> MagicMock:
    """Patch ``GroqChat.complete_json`` to return a controlled JSON payload."""
    mock = MagicMock(return_value=json.dumps(payload))
    monkeypatch.setattr(GroqChat, "complete_json", mock)
    return mock


# ---------------------------------------------------------------------------
# verify_query — happy paths
# ---------------------------------------------------------------------------


class TestVerifyQuery:
    """``verify_query`` must classify the user query via the LLM JSON response."""

    def test_returns_in_scope_true_for_asthma_query(
        self, monkeypatch: pytest.MonkeyPatch, patched_settings: None
    ) -> None:
        """Given an asthma query and the LLM returns in_scope=true, when verifying, then the result mirrors the model."""
        mock = _patch_complete_json(
            monkeypatch, {"in_scope": True, "reason": "asks for asthma definition"}
        )
        state: AgentState = {"query": "What is asthma?"}

        result = verify_query(state)

        assert result == VerifierResult(
            in_scope=True, reason="asks for asthma definition"
        )
        mock.assert_called_once()

    def test_returns_in_scope_false_for_unrelated_query(
        self, monkeypatch: pytest.MonkeyPatch, patched_settings: None
    ) -> None:
        """Given an unrelated query and the LLM returns in_scope=false, when verifying, then the result is out-of-scope."""
        mock = _patch_complete_json(
            monkeypatch, {"in_scope": False, "reason": "about the weather, not asthma"}
        )
        state: AgentState = {"query": "Will it rain tomorrow in Paris?"}

        result = verify_query(state)

        assert result == VerifierResult(
            in_scope=False, reason="about the weather, not asthma"
        )
        mock.assert_called_once()

    def test_messages_contain_system_prompt_and_user_query(
        self, monkeypatch: pytest.MonkeyPatch, patched_settings: None
    ) -> None:
        """Given a state, when verifying, then complete_json receives the system prompt and the raw user query."""
        mock = _patch_complete_json(
            monkeypatch, {"in_scope": True, "reason": "asthma inhaler"}
        )
        state: AgentState = {"query": "How do I use an inhaler?"}

        verify_query(state)

        assert mock.call_count == 1
        messages = mock.call_args.args[0]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == VERIFIER_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "How do I use an inhaler?"

    def test_injected_chat_bypasses_groq_chat_construction(
        self, monkeypatch: pytest.MonkeyPatch, patched_settings: None
    ) -> None:
        """Given a chat injected via the kwarg, when verifying, then ``GroqChat`` is not constructed and the injected chat handles the call."""

        def _explode(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError(
                "GroqChat must not be constructed when chat is injected"
            )

        monkeypatch.setattr(verifier_module, "GroqChat", _explode)
        fake_chat = MagicMock(spec=GroqChat)
        fake_chat.complete_json.return_value = json.dumps(
            {"in_scope": True, "reason": "injected chat path"}
        )

        result = verify_query({"query": "Asthma triggers?"}, chat=fake_chat)

        assert result == VerifierResult(
            in_scope=True, reason="injected chat path"
        )
        fake_chat.complete_json.assert_called_once()

    def test_no_real_api_call_even_if_complete_method_used(
        self, monkeypatch: pytest.MonkeyPatch, patched_settings: None
    ) -> None:
        """Defense in depth: even if some future code path bypassed complete_json, the inner chat.completions.create would explode."""
        from asthma_rag.llm import groq as groq_module

        class _ExplodingCompletions:
            def create(self, **_kwargs: Any) -> None:
                raise AssertionError("real Groq API must not be called")

        def _exploding_groq(api_key: str) -> Any:
            return SimpleNamespace(
                chat=SimpleNamespace(completions=_ExplodingCompletions())
            )

        monkeypatch.setattr(groq_module.groq, "Groq", _exploding_groq)
        _patch_complete_json(
            monkeypatch, {"in_scope": True, "reason": "x"}
        )

        result = verify_query({"query": "What is asthma?"})

        assert result["in_scope"] is True


# ---------------------------------------------------------------------------
# _parse_verifier — fail-open semantics
# ---------------------------------------------------------------------------


class TestParseVerifier:
    """``_parse_verifier`` must fail-open (in_scope=True) on any malformed response."""

    def test_fails_open_on_invalid_json(self) -> None:
        """Given a string that is not valid JSON, when parsing, then fail-open."""
        from asthma_rag.agent.verifier import _parse_verifier

        result = _parse_verifier("not json at all")

        assert result["in_scope"] is True
        assert "could not be parsed" in result["reason"]

    def test_fails_open_on_non_object_json(self) -> None:
        """Given a JSON array (not an object), when parsing, then fail-open."""
        from asthma_rag.agent.verifier import _parse_verifier

        result = _parse_verifier("[1, 2, 3]")

        assert result["in_scope"] is True
        assert "could not be parsed" in result["reason"]

    def test_fails_open_on_missing_in_scope(self) -> None:
        """Given a JSON object missing the in_scope key, when parsing, then fail-open."""
        from asthma_rag.agent.verifier import _parse_verifier

        result = _parse_verifier(json.dumps({"reason": "missing in_scope"}))

        assert result["in_scope"] is True

    def test_fails_open_on_non_bool_in_scope(self) -> None:
        """Given in_scope is a string (not a bool), when parsing, then fail-open."""
        from asthma_rag.agent.verifier import _parse_verifier

        result = _parse_verifier(json.dumps({"in_scope": "yes", "reason": "x"}))

        assert result["in_scope"] is True

    def test_fails_open_on_missing_reason(self) -> None:
        """Given a JSON object missing the reason key, when parsing, then fail-open."""
        from asthma_rag.agent.verifier import _parse_verifier

        result = _parse_verifier(json.dumps({"in_scope": False}))

        assert result["in_scope"] is True

    def test_fails_open_on_non_string_reason(self) -> None:
        """Given reason is not a string, when parsing, then fail-open."""
        from asthma_rag.agent.verifier import _parse_verifier

        result = _parse_verifier(
            json.dumps({"in_scope": False, "reason": 42})
        )

        assert result["in_scope"] is True

    def test_parses_valid_response(self) -> None:
        """Given a valid response, when parsing, then it round-trips faithfully."""
        from asthma_rag.agent.verifier import _parse_verifier

        result = _parse_verifier(
            json.dumps({"in_scope": False, "reason": "sports trivia"})
        )

        assert result == VerifierResult(in_scope=False, reason="sports trivia")


# ---------------------------------------------------------------------------
# verifier_node — state dict + observability hooks
# ---------------------------------------------------------------------------


class TestVerifierNode:
    """``verifier_node`` wraps ``verify_query``, emits observability hooks, and returns a state update."""

    def test_returns_state_dict_with_verifier_result(
        self, monkeypatch: pytest.MonkeyPatch, patched_settings: None
    ) -> None:
        """Given a state, when the node runs, then it returns a dict whose verifier_result matches the model output."""
        _patch_complete_json(
            monkeypatch, {"in_scope": True, "reason": "asthma query"}
        )

        result = verifier_node({"query": "What is asthma?"})

        assert "verifier_result" in result
        assert result["verifier_result"] == VerifierResult(
            in_scope=True, reason="asthma query"
        )

    def test_emits_score_trace_when_in_scope(
        self, monkeypatch: pytest.MonkeyPatch, patched_settings: None
    ) -> None:
        """Given an in-scope classification, when the node runs, then score_trace is called with name verifier_in_scope and value 'True'."""
        mock_score = MagicMock()
        monkeypatch.setattr(verifier_module, "score_trace", mock_score)
        _patch_complete_json(
            monkeypatch, {"in_scope": True, "reason": "asthma related"}
        )

        verifier_node({"query": "What is asthma?"})

        mock_score.assert_called_once_with("verifier_in_scope", "True")

    def test_emits_score_trace_when_out_of_scope(
        self, monkeypatch: pytest.MonkeyPatch, patched_settings: None
    ) -> None:
        """Given an out-of-scope classification, when the node runs, then score_trace is called with 'False'."""
        mock_score = MagicMock()
        monkeypatch.setattr(verifier_module, "score_trace", mock_score)
        _patch_complete_json(
            monkeypatch, {"in_scope": False, "reason": "weather question"}
        )

        verifier_node({"query": "Will it rain?"})

        mock_score.assert_called_once_with("verifier_in_scope", "False")

    def test_emits_add_metadata_with_reason(
        self, monkeypatch: pytest.MonkeyPatch, patched_settings: None
    ) -> None:
        """Given any classification, when the node runs, then add_metadata is called with a dict containing verifier_reason."""
        mock_meta = MagicMock()
        monkeypatch.setattr(verifier_module, "add_metadata", mock_meta)
        _patch_complete_json(
            monkeypatch, {"in_scope": True, "reason": "asthma related"}
        )

        verifier_node({"query": "What is asthma?"})

        mock_meta.assert_called_once_with({"verifier_reason": "asthma related"})


# ---------------------------------------------------------------------------
# VERIFIER_SYSTEM_PROMPT
# ---------------------------------------------------------------------------


class TestVerifierSystemPrompt:
    """The system prompt must instruct the model to return only JSON with the required keys."""

    def test_prompt_instructs_json_only_output(self) -> None:
        """The prompt must require JSON-only output and name both required keys."""
        assert "JSON" in VERIFIER_SYSTEM_PROMPT
        assert "in_scope" in VERIFIER_SYSTEM_PROMPT
        assert "reason" in VERIFIER_SYSTEM_PROMPT

    def test_prompt_defines_in_scope_topics(self) -> None:
        """The prompt must enumerate the in-scope clinical topics."""
        prompt = VERIFIER_SYSTEM_PROMPT.lower()
        assert "asthma" in prompt
        assert "symptom" in prompt or "diagnosis" in prompt
        assert "inhaler" in prompt or "medication" in prompt

    def test_prompt_defines_out_of_scope_topics(self) -> None:
        """The prompt must enumerate clearly unrelated out-of-scope topics."""
        prompt = VERIFIER_SYSTEM_PROMPT.lower()
        assert "out-of-scope" in prompt or "out of scope" in prompt
