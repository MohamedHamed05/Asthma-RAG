"""Cohere v2 reranking wrapper.

Wraps ``cohere.ClientV2.rerank`` and maps result indices back to the
original input documents, so downstream code never has to touch the raw
SDK response shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import cohere

from asthma_rag.config import get_settings


class RerankConfigError(RuntimeError):
    """Raised when the Cohere API key is missing or misconfigured."""


@dataclass(frozen=True, slots=True)
class RerankResult:
    """One reranked document: position, relevance, and original text."""

    index: int
    relevance_score: float
    document: str


class CohereRerank:
    """Client for the Cohere v2 rerank endpoint."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.cohere_api_key:
            raise RerankConfigError("COHERE_API_KEY is not set")
        self._client = cohere.ClientV2(api_key=settings.cohere_api_key)
        self._model = settings.cohere_rerank_model

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        """Rerank ``documents`` against ``query``, returning the top ``top_n`` results."""
        if not documents:
            return []
        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_n,
        )
        return [
            RerankResult(
                index=result.index,
                relevance_score=result.relevance_score,
                document=documents[result.index],
            )
            for result in response.results
        ]
