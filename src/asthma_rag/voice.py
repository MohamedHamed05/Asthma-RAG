"""ASR and TTS wrappers around Groq audio APIs."""

from __future__ import annotations

import os
import re
import tempfile
import wave
from io import BytesIO
from pathlib import Path
from typing import Literal
from asthma_rag.config import Settings, get_settings
from groq import Groq

Voice = Literal["autumn", "diana", "hannah", "austin", "daniel", "troy"]

VOICE_LABELS: dict[Voice, str] = {
    "autumn": "Autumn (warm)",
    "diana": "Diana (calm)",
    "hannah": "Hannah (clear)",
    "austin": "Austin (gentle)",
    "daniel": "Daniel (steady)",
    "troy": "Troy (soft)",
}


class GroqVoice:
    """Groq-powered speech-to-text (Whisper) and text-to-speech (Orpheus)."""

    ASR_MODEL: str = "whisper-large-v3-turbo"
    TTS_MODEL: str = "canopylabs/orpheus-v1-english"
    TTS_MAX_CHARS: int = 200

    def __init__(self, api_key: str | None = None) -> None:
        """Initialise the Groq audio client.

        Args:
            api_key: Groq API key. If ``None``, the ``GROQ_API_KEY``
                environment variable is used.
        """
        self.settings = get_settings()
        self._client = Groq(api_key=self.settings.groq_api_key)

    # ------------------------------------------------------------------ #
    # ASR
    # ------------------------------------------------------------------ #
    def transcribe(self, audio_path: str | Path, language: str = "en") -> str:
        """Transcribe an audio file to text."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with audio_path.open("rb") as fh:
            result = self._client.audio.transcriptions.create(
                file=fh,
                model=self.ASR_MODEL,
                language=language,
            )
        return result.text.strip()

    # ------------------------------------------------------------------ #
    # TTS
    # ------------------------------------------------------------------ #
    def speak(
        self,
        text: str,
        voice: Voice = "hannah",
    ) -> str:
        """Synthesize speech and return a path to a temporary WAV file.

        If *text* exceeds the Groq input limit, it is split into sentence-
        bounded chunks and the resulting WAVs are concatenated so the full
        text is spoken without cuts.
        """
        chunks = self._split_into_chunks(text, self.TTS_MAX_CHARS)
        if not chunks:
            raise ValueError("Nothing to speak.")

        wav_parts: list[bytes] = []
        for i, chunk in enumerate(chunks):
            print(f"[TTS] chunk {i+1}/{len(chunks)}  voice={voice}  chars={len(chunk)}")
            wav_parts.append(self._synthesize_chunk(chunk, voice))

        merged_path = self._concat_wavs(wav_parts)
        print(f"[TTS] Merged {len(wav_parts)} chunks -> {merged_path}")
        return merged_path

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _synthesize_chunk(self, text: str, voice: Voice) -> bytes:
        """Call Groq TTS for a single chunk and return raw WAV bytes."""
        response = self._client.audio.speech.create(
            model=self.TTS_MODEL,
            voice=voice,
            input=text,
            response_format="wav",
        )

        if hasattr(response, "content"):
            return response.content
        if hasattr(response, "read"):
            return response.read()

        # Fallback: stream to a temp file and read back
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        response.stream_to_file(tmp)
        data = Path(tmp).read_bytes()
        Path(tmp).unlink(missing_ok=True)
        return data

    @staticmethod
    def _split_into_chunks(text: str, max_chars: int) -> list[str]:
        """Split *text* into chunks ≤ *max_chars* at sentence boundaries."""
        chunks: list[str] = []
        remaining = text.strip()

        while remaining:
            if len(remaining) <= max_chars:
                chunks.append(remaining)
                break

            window = remaining[:max_chars]
            cut = GroqVoice._last_sentence_end(window)
            if cut <= 0:
                # No sentence boundary found — fall back to last space
                cut = window.rfind(" ")
                if cut <= 0:
                    cut = max_chars  # hard cut (should be rare)

            chunk = remaining[:cut].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[cut:].strip()

        return chunks

    @staticmethod
    def _last_sentence_end(window: str) -> int:
        """Return the index *after* the last sentence-ending punctuation in *window*."""
        for pattern in (r"\.(?=\s|$)", r"!(?=\s|$)", r"\?(?=\s|$)"):
            matches = list(re.finditer(pattern, window))
            if matches:
                return matches[-1].end()
        return -1

    @staticmethod
    def _concat_wavs(parts: list[bytes]) -> str:
        """Concatenate multiple WAV byte strings into one file, return path."""
        if not parts:
            raise ValueError("No WAV parts to concatenate.")
        if len(parts) == 1:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            Path(path).write_bytes(parts[0])
            return path

        # Parse first part to get audio parameters
        with wave.open(BytesIO(parts[0]), "rb") as w:
            nchannels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()

        # Concatenate raw frames
        all_frames = b"".join(
            wave.open(BytesIO(p), "rb").readframes(wave.open(BytesIO(p), "rb").getnframes())
            for p in parts
        )

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as w:
            w.setnchannels(nchannels)
            w.setsampwidth(sampwidth)
            w.setframerate(framerate)
            w.writeframes(all_frames)
        return path