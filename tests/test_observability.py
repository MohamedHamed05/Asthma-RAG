"""Tests for the Langfuse observability layer.

RED phase: these tests FAIL before ``src/asthma_rag/observability.py`` exists.
All tests are hermetic — the Langfuse constructor is mocked; no network calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from asthma_rag import observability
from asthma_rag.config import Settings


@pytest.fixture(autouse=True)
def _reset_cache():
    observability.reset_langfuse_cache()
    yield
    observability.reset_langfuse_cache()


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "langfuse_public_key": "",
        "langfuse_secret_key": "",
        "langfuse_base_url": "",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _patch_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr(observability, "get_settings", lambda: settings)


class TestGetLangfuse:
    def test_returns_none_when_keys_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch, _settings())

        assert observability.get_langfuse() is None

    def test_returns_none_when_tracing_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(
            monkeypatch,
            _settings(
                langfuse_public_key="pk-x",
                langfuse_secret_key="sk-x",
                langfuse_tracing_enabled=False,
            ),
        )

        assert observability.get_langfuse() is None

    def test_builds_configured_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = MagicMock(name="langfuse_client")
        constructor = MagicMock(name="Langfuse", return_value=fake)
        monkeypatch.setattr("langfuse.Langfuse", constructor)
        _patch_settings(
            monkeypatch,
            _settings(
                langfuse_public_key="pk-x",
                langfuse_secret_key="sk-x",
                langfuse_base_url="https://lf.example.com",
            ),
        )

        client = observability.get_langfuse()

        assert client is fake
        constructor.assert_called_once_with(
            public_key="pk-x",
            secret_key="sk-x",
            base_url="https://lf.example.com",
        )

    def test_uses_default_cloud_url_when_base_url_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        constructor = MagicMock(name="Langfuse", return_value=MagicMock())
        monkeypatch.setattr("langfuse.Langfuse", constructor)
        _patch_settings(
            monkeypatch,
            _settings(langfuse_public_key="pk-x", langfuse_secret_key="sk-x"),
        )

        observability.get_langfuse()

        assert constructor.call_args.kwargs["base_url"] == "https://cloud.langfuse.com"

    def test_never_raises_on_constructor_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "langfuse.Langfuse",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        _patch_settings(
            monkeypatch,
            _settings(langfuse_public_key="pk-x", langfuse_secret_key="sk-x"),
        )

        assert observability.get_langfuse() is None

    def test_caches_client_across_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = MagicMock()
        monkeypatch.setattr("langfuse.Langfuse", MagicMock(return_value=fake))
        _patch_settings(
            monkeypatch,
            _settings(langfuse_public_key="pk-x", langfuse_secret_key="sk-x"),
        )

        assert observability.get_langfuse() is fake
        assert observability.get_langfuse() is fake


class TestTracedDecorator:
    def test_noop_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_settings(monkeypatch, _settings())

        @observability.traced("my_node", as_type="retriever")
        def node(x: int) -> int:
            return x * 2

        assert node(21) == 42

    def test_creates_span_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        span = MagicMock(name="span")
        client = MagicMock(name="client")
        client.start_as_current_observation.return_value.__enter__ = MagicMock(
            return_value=span
        )
        client.start_as_current_observation.return_value.__exit__ = MagicMock(
            return_value=False
        )
        monkeypatch.setattr("langfuse.Langfuse", MagicMock(return_value=client))
        _patch_settings(
            monkeypatch,
            _settings(langfuse_public_key="pk-x", langfuse_secret_key="sk-x"),
        )

        @observability.traced("my_node", as_type="retriever")
        def node(x: int) -> int:
            return x * 2

        assert node(21) == 42
        client.start_as_current_observation.assert_called_once_with(
            name="my_node", as_type="retriever"
        )

    def test_captures_exceptions_as_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        span = MagicMock(name="span")
        client = MagicMock(name="client")
        client.start_as_current_observation.return_value.__enter__ = MagicMock(
            return_value=span
        )
        client.start_as_current_observation.return_value.__exit__ = MagicMock(
            return_value=False
        )
        monkeypatch.setattr("langfuse.Langfuse", MagicMock(return_value=client))
        _patch_settings(
            monkeypatch,
            _settings(langfuse_public_key="pk-x", langfuse_secret_key="sk-x"),
        )

        @observability.traced("boom_node")
        def node() -> None:
            raise ValueError("kaput")

        with pytest.raises(ValueError, match="kaput"):
            node()

        span.update.assert_called_once_with(level="ERROR", status_message="kaput")


class TestGuardedHelpers:
    def test_score_trace_noop_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch, _settings())

        observability.score_trace("judge_confidence", "High")

    def test_score_trace_calls_client_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = MagicMock(name="client")
        monkeypatch.setattr("langfuse.Langfuse", MagicMock(return_value=client))
        _patch_settings(
            monkeypatch,
            _settings(langfuse_public_key="pk-x", langfuse_secret_key="sk-x"),
        )

        observability.score_trace("judge_confidence", "High")

        client.score_current_trace.assert_called_once_with(
            name="judge_confidence", value="High", data_type="CATEGORICAL"
        )

    def test_score_trace_swallows_client_errors_with_log(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = MagicMock(name="client")
        client.score_current_trace.side_effect = RuntimeError("network down")
        monkeypatch.setattr("langfuse.Langfuse", MagicMock(return_value=client))
        _patch_settings(
            monkeypatch,
            _settings(langfuse_public_key="pk-x", langfuse_secret_key="sk-x"),
        )

        observability.score_trace("judge_confidence", "High")

    def test_add_metadata_noop_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch, _settings())

        observability.add_metadata({"refusal_reason": "out of scope"})

    def test_add_metadata_calls_client_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = MagicMock(name="client")
        monkeypatch.setattr("langfuse.Langfuse", MagicMock(return_value=client))
        _patch_settings(
            monkeypatch,
            _settings(langfuse_public_key="pk-x", langfuse_secret_key="sk-x"),
        )

        observability.add_metadata({"refusal_reason": "out of scope"})

        client.update_current_span.assert_called_once_with(
            metadata={"refusal_reason": "out of scope"}
        )

    def test_set_trace_attributes_applies_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = MagicMock(name="client")
        monkeypatch.setattr("langfuse.Langfuse", MagicMock(return_value=client))
        _patch_settings(
            monkeypatch,
            _settings(langfuse_public_key="pk-x", langfuse_secret_key="sk-x"),
        )

        observability.set_trace_attributes(session_id="s1", tags=["asthma-rag"])

        client.update_current_span.assert_called_once_with(
            session_id="s1", tags=["asthma-rag"]
        )

    def test_flush_traces_noop_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch, _settings())

        observability.flush_traces()

    def test_flush_traces_calls_client_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = MagicMock(name="client")
        monkeypatch.setattr("langfuse.Langfuse", MagicMock(return_value=client))
        _patch_settings(
            monkeypatch,
            _settings(langfuse_public_key="pk-x", langfuse_secret_key="sk-x"),
        )

        observability.flush_traces()

        client.flush.assert_called_once_with()
