"""Tests for the Groq chat wrapper.

Locks the contract of ``asthma_rag.llm.groq``: construction must fail fast
with a friendly error when the API key is missing, and ``complete`` /
``complete_json`` must forward the right parameters to the underlying client.
A scripted fake client records the kwargs it receives, so no network or
credentials are needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from asthma_rag.config import Settings
from asthma_rag.llm.groq import GroqChat, LLMConfigError


class _RecordingCompletions:
    """Fake ``chat.completions`` that records kwargs and returns a canned reply."""

    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self._captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="canned answer"))]
        )


class _RecordingClient:
    """Fake ``groq.Groq`` exposing ``chat.completions.create``."""

    def __init__(self, captured: dict[str, Any]) -> None:
        self.chat = SimpleNamespace(completions=_RecordingCompletions(captured))


def _make_chat(captured: dict[str, Any]) -> GroqChat:
    return GroqChat(api_key="test-key", client=_RecordingClient(captured))


def test_missing_api_key_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given an empty configured API key, when constructing GroqChat, then LLMConfigError with a friendly message is raised."""
    from asthma_rag.llm import groq as groq_module

    monkeypatch.setattr(
        groq_module,
        "get_settings",
        lambda: Settings(groq_api_key=""),
    )

    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        GroqChat()


def test_complete_sends_default_model_and_returns_content() -> None:
    """Given a client, when completing with only messages, then defaults (model/temperature/max_tokens) are sent and content returned."""
    captured: dict[str, Any] = {}
    chat = _make_chat(captured)
    messages = [{"role": "user", "content": "hi"}]

    result = chat.complete(messages)

    assert result == "canned answer"
    assert captured["model"] == Settings().groq_model
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 1024
    assert captured["messages"] == messages


def test_complete_forwards_explicit_overrides() -> None:
    """Given explicit model/temperature/max_tokens and extra kwargs, when completing, then all are forwarded to the client."""
    captured: dict[str, Any] = {}
    chat = _make_chat(captured)

    chat.complete(
        [{"role": "user", "content": "hi"}],
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=512,
        top_p=0.9,
    )

    assert captured["model"] == "llama-3.1-8b-instant"
    assert captured["temperature"] == 0.7
    assert captured["max_tokens"] == 512
    assert captured["top_p"] == 0.9


def test_complete_json_sets_json_object_response_format() -> None:
    """Given messages, when completing in JSON mode, then response_format={'type': 'json_object'} is sent."""
    captured: dict[str, Any] = {}
    chat = _make_chat(captured)

    result = chat.complete_json([{"role": "user", "content": "return json"}])

    assert result == "canned answer"
    assert captured["response_format"] == {"type": "json_object"}


def test_complete_json_with_schema_uses_structured_output() -> None:
    """Given a JSON schema, when completing in JSON mode, then response_format uses the structured-output shape."""
    captured: dict[str, Any] = {}
    chat = _make_chat(captured)
    schema = {"name": "answer", "schema": {"type": "object"}}

    chat.complete_json(
        [{"role": "user", "content": "return json"}],
        json_schema=schema,
    )

    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": schema,
    }
