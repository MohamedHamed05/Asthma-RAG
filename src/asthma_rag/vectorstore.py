"""Chroma vector-store wrapper for the asthma RAG pipeline.

The wrapper owns the Chroma client/collection lifecycle and exposes a
narrow, typed surface: add chunks, query, and delete the collection.
Embeddings default to a deterministic offline hash embedding so the
pipeline runs without network access; the Qwen3 embedding model plugs in
later via the ``embedding_function`` parameter.
"""

from __future__ import annotations

import hashlib
import math

import chromadb
import chromadb.errors
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from asthma_rag.config import Settings

COLLECTION_NAME = "asthma_kb"
_EMBEDDING_DIM = 384


class _HashEmbedding(EmbeddingFunction[Documents]):
    """Deterministic, offline, fixed-dimension embedding for tests/dev."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors: list[list[float]] = []
        for text in input:
            vector = [0.0] * _EMBEDDING_DIM
            for token in text.lower().split():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % _EMBEDDING_DIM
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors

    @staticmethod
    def name() -> str:
        return "hash-embedding"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> _HashEmbedding:
        return _HashEmbedding()


class ChromaStore:
    """Persistent Chroma collection for asthma guideline chunks."""

    def __init__(
        self,
        settings: Settings,
        collection_name: str = COLLECTION_NAME,
        embedding_function: EmbeddingFunction[Documents] | None = None,
    ) -> None:
        self._client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_function or _HashEmbedding(),
        )

    def add_chunks(self, chunks: list[dict], batch_size: int = 500) -> None:
        """Add chunk dicts (``chunk_id``, ``text``, metadata keys) to the store.

        Chunks are added in batches so large PDF collections stay under
        Chroma's per-request batch limit.
        """
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._collection.add(
                ids=[chunk["chunk_id"] for chunk in batch],
                documents=[chunk["text"] for chunk in batch],
                metadatas=[
                    {
                        "doc_name": chunk["doc_name"],
                        "page_number": chunk["page_number"],
                        "chunk_word_count": chunk["chunk_word_count"],
                        "chunk_id": chunk["chunk_id"],
                        "section_title": chunk["section_title"] or "",
                    }
                    for chunk in batch
                ],
            )

    def query(self, query_text: str, n_results: int) -> dict:
        """Return the raw Chroma query result dict for ``query_text``."""
        return self._collection.query(query_texts=[query_text], n_results=n_results)

    def delete_collection(self) -> None:
        """Delete the collection (used by tests and re-indexing)."""
        self._client.delete_collection(self._collection.name)


def delete_collection_if_exists(
    settings: Settings, collection_name: str = COLLECTION_NAME
) -> None:
    """Delete ``collection_name`` if it exists, so a fresh one can be created.

    Works at the client level without any embedding function, so it can clear
    collections persisted with a different (conflicting) embedding config.
    """
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    try:
        client.delete_collection(collection_name)
    except chromadb.errors.NotFoundError:
        return