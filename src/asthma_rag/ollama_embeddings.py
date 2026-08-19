"""Ollama-backed embedding function for Chroma.

Embeds through a GPU-accelerated Ollama server (``/api/embed``) so large PDF
collections index in seconds instead of hours of CPU inference. Query and
document texts are embedded identically, keeping the vector space symmetric
between index time and query time.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma embedding function backed by Ollama's ``/api/embed`` endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-embedding:0.6b",
        batch_size: int = 64,
        timeout: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._batch_size = batch_size
        self._timeout = timeout

    def __call__(self, input: Documents) -> Embeddings:
        vectors: Embeddings = []
        for start in range(0, len(input), self._batch_size):
            batch = list(input[start : start + self._batch_size])
            vectors.extend(self._fetch_embeddings(batch))
        return vectors

    def _fetch_embeddings(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self._model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            raise RuntimeError(
                f"Ollama embed request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._base_url} - check that the server "
                f"is running and OLLAMA_BASE_URL is correct ({exc.reason})"
            ) from exc

        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            count = len(embeddings) if isinstance(embeddings, list) else "no"
            raise RuntimeError(
                f"Ollama embed returned {count} embeddings for {len(texts)} inputs"
            )
        return embeddings

    @staticmethod
    def name() -> str:
        return "ollama-embedding"

    def get_config(self) -> dict[str, Any]:
        return {
            "base_url": self._base_url,
            "model": self._model,
            "batch_size": self._batch_size,
        }

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> OllamaEmbeddingFunction:
        return OllamaEmbeddingFunction(
            base_url=config.get("base_url", "http://localhost:11434"),
            model=config.get("model", "qwen3-embedding:0.6b"),
            batch_size=int(config.get("batch_size", 64)),
        )
