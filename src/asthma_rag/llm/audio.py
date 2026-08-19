"""Groq audio (STT/TTS) wrapper.

Wraps the Groq speech-to-text (``audio.transcriptions``) and text-to-speech
(``audio.speech``) APIs behind a small, typed interface with sane defaults for
the RAG pipeline: English STT returning JSON, and WAV TTS (the only format the
Orpheus voice model supports) with long text truncated to whole sentences.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

import groq

from asthma_rag.config import get_settings


class AudioTranscriptions(Protocol):
    """Structural contract for ``client.audio.transcriptions`` (allows test fakes)."""

    def create(self, **kwargs: Any) -> Any: ...


class AudioSpeech(Protocol):
    """Structural contract for ``client.audio.speech`` (allows test fakes)."""

    def create(self, **kwargs: Any) -> Any: ...


class AudioClient(Protocol):
    """Structural contract for ``client.audio`` (allows test fakes)."""

    transcriptions: AudioTranscriptions
    speech: AudioSpeech


class GroqAudioLike(Protocol):
    """Structural contract for ``groq.Groq`` restricted to the audio surface."""

    audio: AudioClient


class AudioConfigError(RuntimeError):
    """Raised when the Groq audio client cannot be configured (e.g. missing API key)."""


def _truncate_for_tts(text: str, max_chars: int = 200) -> str:
    """Truncate ``text`` to whole sentences within ``max_chars``.

    Strips surrounding whitespace. If the text already fits, it is returned
    unchanged. Otherwise sentences are accumulated (split on sentence enders)
    while the running total stays within budget. If even the first sentence
    exceeds the budget, the text is hard-cut at ``max_chars``. Never returns an
    empty string.
    """
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped

    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    kept: list[str] = []
    total = 0
    for sentence in sentences:
        # Account for the space that joins this sentence to the previous one.
        sep = 1 if kept else 0
        if total + sep + len(sentence) > max_chars:
            break
        kept.append(sentence)
        total += sep + len(sentence)

    if kept:
        return " ".join(kept)

    return stripped[:max_chars]


class GroqAudio:
    """Thin wrapper around ``groq.Groq.audio`` for STT and TTS."""

    def __init__(
        self,
        api_key: str | None = None,
        client: GroqAudioLike | None = None,
    ) -> None:
        """Build the wrapper, failing fast with ``AudioConfigError`` on a missing key.

        ``api_key`` defaults to the value from ``Settings``; a pre-built
        ``client`` may be injected for tests.
        """
        settings = get_settings()
        self._api_key = settings.groq_api_key if api_key is None else api_key
        self._stt_model = settings.stt_model
        self._tts_model = settings.tts_model
        self._tts_voice = settings.tts_voice
        if not self._api_key:
            raise AudioConfigError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or environment variables to use Groq audio."
            )
        self._client = client if client is not None else groq.Groq(api_key=self._api_key)

    def transcribe(self, file_path: str | Path) -> str:
        """Transcribe an audio file to English text.

        ``file_path`` may be a string or ``Path``; the file is passed to Groq
        as a ``Path`` and the resulting transcript text is returned.
        """
        transcription = self._client.audio.transcriptions.create(
            file=Path(file_path),
            model=self._stt_model,
            language="en",
            response_format="json",
        )
        return transcription.text

    def speak(self, text: str) -> bytes:
        """Synthesize ``text`` into WAV audio bytes.

        Long text is truncated to whole sentences (see ``_truncate_for_tts``)
        before being sent, and the resulting WAV content is returned as bytes.
        """
        truncated = _truncate_for_tts(text)
        speech = self._client.audio.speech.create(
            model=self._tts_model,
            voice=self._tts_voice,
            input=truncated,
            response_format="wav",
        )
        return speech.content

    def speak_to_file(self, text: str, out_path: str | Path) -> Path:
        """Synthesize ``text`` and write the WAV bytes to ``out_path``.

        Returns the resolved ``Path`` of the written file.
        """
        path = Path(out_path)
        path.write_bytes(self.speak(text))
        return path
