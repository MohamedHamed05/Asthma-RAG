"""Tests for Settings fields added in iteration 2 (search, observability, audio)."""

from __future__ import annotations

from asthma_rag.config import Settings


def test_search_settings_defaults() -> None:
    settings = Settings(exa_api_key="")

    assert settings.exa_api_key == ""
    assert settings.search_max_results == 5
    assert settings.search_max_age_hours == 24


def test_langfuse_settings_defaults() -> None:
    settings = Settings(langfuse_public_key="", langfuse_secret_key="", langfuse_base_url="")

    assert settings.langfuse_public_key == ""
    assert settings.langfuse_secret_key == ""
    assert settings.langfuse_base_url == ""
    assert settings.langfuse_tracing_enabled is True


def test_agent_role_model_defaults() -> None:
    settings = Settings()

    assert settings.verifier_model == "openai/gpt-oss-20b"
    assert settings.judge_model == "openai/gpt-oss-120b"


def test_audio_settings_defaults() -> None:
    settings = Settings()

    assert settings.stt_model == "whisper-large-v3-turbo"
    assert settings.tts_model == "canopylabs/orpheus-v1-english"
    assert settings.tts_voice == "troy"


def test_new_fields_are_env_overridable(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-test")
    monkeypatch.setenv("VERIFIER_MODEL", "some/model")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    monkeypatch.setenv("TTS_VOICE", "diana")

    settings = Settings()

    assert settings.exa_api_key == "exa-test"
    assert settings.verifier_model == "some/model"
    assert settings.langfuse_tracing_enabled is False
    assert settings.tts_voice == "diana"
