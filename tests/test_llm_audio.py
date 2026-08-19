"""Tests for the Groq audio (STT/TTS) wrapper.

Locks the contract of ``asthma_rag.llm.audio``: construction must fail fast
with a friendly error when the API key is missing, ``transcribe`` must forward
the right STT parameters and return the transcript text, ``speak`` must
truncate long text to whole sentences and return WAV bytes, and
``speak_to_file`` must persist those bytes. A scripted fake client records the
kwargs it receives, so no network or credentials are needed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from asthma_rag.config import Settings
from asthma_rag.llm.audio import (
    AudioConfigError,
    GroqAudio,
    _truncate_for_tts,
)


class _RecordingTranscriptions:
    """Fake ``audio.transcriptions`` that records kwargs and returns a canned reply."""

    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self._captured.update(kwargs)
        return SimpleNamespace(text="canned transcript")


class _RecordingSpeech:
    """Fake ``audio.speech`` that records kwargs and returns canned WAV bytes."""

    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self._captured.update(kwargs)
        return SimpleNamespace(content=b"RIFF....WAVE")


class _RecordingAudio:
    """Fake ``groq.Groq.audio`` exposing ``transcriptions`` and ``speech``."""

    def __init__(self, captured: dict[str, Any]) -> None:
        self.transcriptions = _RecordingTranscriptions(captured)
        self.speech = _RecordingSpeech(captured)


class _RecordingClient:
    """Fake ``groq.Groq`` exposing ``audio``."""

    def __init__(self, captured: dict[str, Any]) -> None:
        self.audio = _RecordingAudio(captured)


def _make_audio(captured: dict[str, Any]) -> GroqAudio:
    return GroqAudio(api_key="test-key", client=_RecordingClient(captured))


def test_missing_api_key_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given an empty configured API key, when constructing GroqAudio, then AudioConfigError with a friendly message is raised."""
    from asthma_rag.llm import audio as audio_module

    monkeypatch.setattr(
        audio_module,
        "get_settings",
        lambda: Settings(groq_api_key=""),
    )

    with pytest.raises(AudioConfigError, match="GROQ_API_KEY"):
        GroqAudio()


def test_transcribe_sends_stt_params_and_returns_text(tmp_path: Path) -> None:
    """Given a client and an audio file, when transcribing, then STT params are sent and the transcript text returned."""
    captured: dict[str, Any] = {}
    audio = _make_audio(captured)
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"audio")

    result = audio.transcribe(audio_file)

    assert result == "canned transcript"
    assert captured["model"] == Settings().stt_model
    assert captured["language"] == "en"
    assert captured["response_format"] == "json"
    assert captured["file"] == audio_file


def test_speak_sends_tts_params_and_returns_wav_bytes() -> None:
    """Given a client and short text, when speaking, then TTS params are sent and WAV bytes returned."""
    captured: dict[str, Any] = {}
    audio = _make_audio(captured)

    result = audio.speak("Hello world.")

    assert result == b"RIFF....WAVE"
    assert captured["model"] == Settings().tts_model
    assert captured["voice"] == Settings().tts_voice
    assert captured["response_format"] == "wav"
    assert captured["input"] == "Hello world."


def test_speak_truncates_long_text_to_whole_sentences() -> None:
    """Given text longer than the TTS budget, when speaking, then input is truncated to whole sentences within budget."""
    captured: dict[str, Any] = {}
    audio = _make_audio(captured)

    long_text = (
        "First sentence is short. "
        "Second sentence is also short. "
        "Third sentence is short too. "
        "Fourth sentence is short as well. "
        "Fifth sentence is short again. "
        "Sixth sentence is short once more. "
        "Seventh sentence is short yet again. "
        "Eighth sentence is short still. "
        "Ninth sentence is short now. "
        "Tenth sentence is short finally."
    )

    audio.speak(long_text)

    sent = captured["input"]
    assert len(sent) <= 200
    assert sent.endswith(".")
    assert sent.startswith("First sentence is short.")


def test_speak_to_file_writes_bytes_and_returns_path(tmp_path: Path) -> None:
    """Given a client and an output path, when speaking to file, then WAV bytes are written and the Path returned."""
    captured: dict[str, Any] = {}
    audio = _make_audio(captured)
    out_path = tmp_path / "out.wav"

    result = audio.speak_to_file("Hello world.", out_path)

    assert result == out_path
    assert out_path.read_bytes() == b"RIFF....WAVE"


def test_truncate_returns_text_unchanged_when_within_budget() -> None:
    """Given text within the budget, when truncating, then it is returned as-is."""
    text = "Short text."

    result = _truncate_for_tts(text)

    assert result == text


def test_truncate_keeps_whole_sentences_within_budget() -> None:
    """Given text over budget, when truncating, then whole sentences are kept while total stays within budget."""
    text = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten."

    result = _truncate_for_tts(text, max_chars=20)

    assert len(result) <= 20
    assert result == "One. Two. Three."


def test_truncate_hard_cuts_when_first_sentence_exceeds_budget() -> None:
    """Given a first sentence longer than the budget, when truncating, then it is hard-cut at the budget."""
    text = "This is a very long first sentence that exceeds the budget entirely."

    result = _truncate_for_tts(text, max_chars=20)

    assert len(result) == 20
    assert result == text[:20]


def test_truncate_never_returns_empty_on_hard_cut() -> None:
    """Given a single sentence over budget, when truncating, then a non-empty hard-cut string is returned."""
    result = _truncate_for_tts("A" * 300, max_chars=10)

    assert result != ""
    assert len(result) == 10
