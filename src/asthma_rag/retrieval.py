"""Retrieval, reranking, and context formatting for the asthma RAG pipeline.

Wave 3: retrieves candidate chunks from Chroma, reranks them with Cohere,
and renders the evidence block consumed by the answer-generation prompt.
"""

from __future__ import annotations

from typing import TypedDict

from chromadb.api.types import Documents, EmbeddingFunction

from asthma_rag.config import get_settings
from asthma_rag.embeddings import get_embedding_function
from asthma_rag.rerank.cohere_v2 import CohereRerank
from asthma_rag.vectorstore import ChromaStore

DEFAULT_N_RESULTS = 50
DEFAULT_TOP_N = 5

# Chroma metadata values are scalars (str/int/float).
ChunkMetadata = dict[str, str | int | float]

# Unknown sentinel for missing metadata fields (mirrors the notebook).
_UNKNOWN = "Unknown"


class RetrievalResult(TypedDict):
    """Raw Chroma query-result shape: one-query lists wrapping per-chunk lists."""

    ids: list[list[str]]
    documents: list[list[str]]
    metadatas: list[list[ChunkMetadata]]
    distances: list[list[float]]


class RerankedResult(TypedDict):
    """Reranked result, same shape as Chroma but without ``ids``."""

    documents: list[list[str]]
    metadatas: list[list[ChunkMetadata]]
    distances: list[list[float]]


def retrieve(
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    embedding_function: EmbeddingFunction[Documents] | None = None,
) -> RetrievalResult:
    """Return the raw Chroma query dict for ``query`` using default settings.

    Queries are embedded with the configured Qwen3 model by default so they
    match the index built by ``scripts/run_pipeline.py``. Tests and dev
    environments that index with the deterministic hash embedding can pass
    their own ``embedding_function``.
    """
    if embedding_function is None:
        embedding_function = get_embedding_function()
    store = ChromaStore(get_settings(), embedding_function=embedding_function)
    return store.query(query, n_results)


def rerank_results(
    query: str, retrieved: RetrievalResult, top_n: int = DEFAULT_TOP_N
) -> RerankedResult:
    """Rerank ``retrieved["documents"][0]`` with Cohere and remap to Chroma shape.

    Original metadata is preserved and enriched with ``vector_distance`` and
    ``rerank_score``, keeping the ordering Cohere returned.
    """
    documents = (retrieved.get("documents") or [[]])[0]
    metadatas = (retrieved.get("metadatas") or [[]])[0]
    distances = (retrieved.get("distances") or [[]])[0]

    results = CohereRerank().rerank(query, documents, top_n)

    reranked_documents: list[str] = []
    reranked_metadatas: list[ChunkMetadata] = []
    reranked_distances: list[float] = []

    for result in results:
        idx = result.index
        metadata = dict(metadatas[idx]) if idx < len(metadatas) else {}
        metadata["vector_distance"] = distances[idx] if idx < len(distances) else 0.0
        metadata["rerank_score"] = result.relevance_score
        reranked_documents.append(documents[idx])
        reranked_metadatas.append(metadata)
        reranked_distances.append(distances[idx])

    return {
        "documents": [reranked_documents],
        "metadatas": [reranked_metadatas],
        "distances": [reranked_distances],
    }


def format_retrieved_context(
    results: RetrievalResult | RerankedResult, user_query: str | None = None
) -> str:
    """Build the context block consumed by the answer prompt.

    Mirrors the original notebook: each chunk is rendered under labeled
    headers, and vector/rerank scores are exposed as retrieval metadata
    lines only — never as content.
    """
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]

    context_parts: list[str] = []

    if user_query is not None:
        context_parts.append(f"USER QUERY:\n{user_query}\n")

    context_parts.append("RETRIEVED CLINICAL CONTEXT:\n")

    for i, document in enumerate(documents, start=1):
        metadata = metadatas[i - 1] if i - 1 < len(metadatas) else {}

        source = metadata.get("doc_name", _UNKNOWN)
        chunk_id = metadata.get("chunk_id", _UNKNOWN)
        page = metadata.get("page_number", _UNKNOWN)
        word_count = metadata.get("chunk_word_count", _UNKNOWN)
        vector_distance = metadata.get("vector_distance")
        rerank_score = metadata.get("rerank_score")

        chunk = (
            f"===== Chunk {i} =====\n"
            f"Source: {source}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Page: {page}\n"
            f"Chunk Word Count: {word_count}\n"
        )

        if isinstance(vector_distance, (int, float)):
            chunk += f"Vector Distance: {vector_distance:.4f}\n"

        if isinstance(rerank_score, (int, float)):
            chunk += f"Rerank Score: {rerank_score:.4f}\n"

        chunk += f"\nContent:\n{document.strip()}\n"

        context_parts.append(chunk)

    return "\n".join(context_parts)
