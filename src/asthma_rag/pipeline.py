"""High-level orchestrator for the asthma RAG pipeline.

A single ``Pipeline`` instance wires together ingestion, indexing, and the
agentic retrieval graph. It is intentionally thin: each stage delegates to the
focused modules in ``cleaning/``, ``ingest.py``, ``vectorstore.py``, and
``agent/graph.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chromadb.api.types import EmbeddingFunction, Documents

from asthma_rag.agent.graph import build_graph
from asthma_rag.config import Settings, get_settings
from asthma_rag.ingest import chunk_pages, extract_pages
from asthma_rag.vectorstore import ChromaStore, delete_collection_if_exists


class Pipeline:
    """End-to-end orchestrator: PDFs -> chunks -> Chroma -> agent answers."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_function: EmbeddingFunction[Documents] | None = None,
    ) -> None:
        """Create a pipeline using the given (or default) settings.

        ``embedding_function`` is forwarded to ``ChromaStore`` so callers can
        plug in the real Qwen3 embedding model; the default hash embedding is
        used when none is supplied.
        """
        self.settings = settings or get_settings()
        self._embedding_function = embedding_function
        self._store: ChromaStore | None = None
        self._graph = build_graph()

    def _get_store(self) -> ChromaStore:
        """Return the Chroma store, creating it lazily on first use.

        Query-only processes (the Gradio UI, ``scripts/ask.py``) never touch
        the index store — the graph retrieves through ``asthma_rag.retrieval``
        — so the store is only constructed when ingest/index actually runs.
        """
        if self._store is None:
            self._store = ChromaStore(
                self.settings,
                embedding_function=self._embedding_function,
            )
        return self._store

    def ingest(
        self,
        pdf_dir: Path | None = None,
        out: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Extract and chunk every PDF in ``pdf_dir`` and write ``chunks.json``.

        Returns the chunk list so callers can inspect or transform it before
        indexing.
        """
        pdf_dir = pdf_dir or self.settings.data_raw_dir
        out = out or self.settings.data_chunks_dir / "chunks.json"

        chunks: list[dict[str, Any]] = []
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            chunks.extend(chunk_pages(extract_pages(pdf_path)))

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
        return chunks

    def index(
        self,
        chunks: list[dict[str, Any]] | None = None,
        chunks_path: Path | None = None,
    ) -> None:
        """Load chunks and rebuild the Chroma collection.

        The existing collection is always deleted first so repeated runs do not
        create duplicate entries.
        """
        if chunks is None:
            path = chunks_path or self.settings.data_chunks_dir / "chunks.json"
            chunks = json.loads(path.read_text(encoding="utf-8"))

        delete_collection_if_exists(self.settings)
        self._store = None
        self._get_store().add_chunks(chunks)

    def query(self, question: str) -> dict[str, Any]:
        """Run the LangGraph agent on ``question`` and return the final state."""
        return self._graph.invoke({"query": question})

    def run(
        self,
        pdf_dir: Path | None = None,
        chunks_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Ingest PDFs and rebuild the index in one call."""
        chunks = self.ingest(pdf_dir, chunks_path)
        self.index(chunks)
        return chunks
