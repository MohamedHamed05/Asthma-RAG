"""Groq chat client wrapper.

Wraps the Groq chat-completions API behind a small, typed interface with
sane defaults for the RAG pipeline: deterministic (temperature 0), bounded
output, and an optional JSON mode for the grader node.
"""

from __future__ import annotations

from typing import Any, Protocol

import groq
from groq.types.chat import ChatCompletion

from asthma_rag.config import get_settings

Message = dict[str, Any]


class ChatCompletions(Protocol):
    """Structural contract for ``client.chat.completions`` (allows test fakes)."""

    def create(self, **kwargs: Any) -> ChatCompletion: ...


class ChatClient(Protocol):
    completions: ChatCompletions


class GroqClientLike(Protocol):
    chat: ChatClient


class LLMConfigError(RuntimeError):
    """Raised when the Groq client cannot be configured (e.g. missing API key)."""


class GroqChat:
    """Thin wrapper around ``groq.Groq.chat.completions.create``."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: GroqClientLike | None = None,
    ) -> None:
        """Build the wrapper, failing fast with ``LLMConfigError`` on a missing key.

        ``api_key`` and ``model`` default to the values from ``Settings``; a
        pre-built ``client`` may be injected for tests.
        """
        settings = get_settings()
        self._api_key = settings.groq_api_key if api_key is None else api_key
        self._model = settings.groq_model if model is None else model
        if not self._api_key:
            raise LLMConfigError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or environment variables to use the Groq LLM."
            )
        self._client = client if client is not None else groq.Groq(api_key=self._api_key)

    def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **overrides: Any,
    ) -> str:
        """Send ``messages`` and return the assistant's text content.

        Parameters not provided fall back to the wrapper defaults (the
        configured model, temperature 0.0, 1024 max tokens); any extra
        keyword arguments are forwarded to the Groq API untouched.
        """
        kwargs: dict[str, Any] = dict(overrides)
        kwargs["model"] = self._model if model is None else model
        kwargs["temperature"] = 0.0 if temperature is None else temperature
        kwargs["max_tokens"] = 1024 if max_tokens is None else max_tokens
        kwargs["messages"] = messages
        completion = self._client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content
        if content is None:
            raise RuntimeError("Groq chat completion returned no message content.")
        return content

    def complete_json(
        self,
        messages: list[Message],
        json_schema: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> str:
        """Like ``complete`` but force a JSON response (for the grader node).

        With ``json_schema`` provided, Groq's structured output is requested;
        otherwise plain ``json_object`` mode is used.
        """
        response_format: dict[str, Any] = (
            {"type": "json_schema", "json_schema": json_schema}
            if json_schema is not None
            else {"type": "json_object"}
        )
        return self.complete(messages, response_format=response_format, **overrides)
