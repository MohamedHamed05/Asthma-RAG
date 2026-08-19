"""Tests for asthma_rag.vectorstore.ChromaStore."""

from __future__ import annotations

from pathlib import Path

import chromadb
import chromadb.errors
import pytest

from asthma_rag.config import Settings
from asthma_rag.vectorstore import ChromaStore, delete_collection_if_exists


@pytest.fixture()
def store(tmp_path: Path) -> ChromaStore:
    return ChromaStore(Settings(chroma_path=tmp_path))


def _chunks() -> list[dict]:
    return [
        {
            "chunk_id": "guidelines_1_0",
            "text": "Inhaled corticosteroids are the preferred controller for asthma.",
            "doc_name": "guidelines",
            "page_number": 1,
            "chunk_word_count": 10,
            "section_title": "3.5 Management of Asthma",
        },
        {
            "chunk_id": "guidelines_2_0",
            "text": "Long-acting beta-agonists require combination therapy.",
            "doc_name": "guidelines",
            "page_number": 2,
            "chunk_word_count": 7,
            "section_title": "3.5 Management of Asthma",
        },
    ]


def test_chroma_store_add_and_query_round_trip(store: ChromaStore) -> None:
    store.add_chunks(_chunks())

    result = store.query("controller medication for persistent asthma", n_results=1)

    assert result["ids"][0] == ["guidelines_1_0"]
    assert result["documents"][0][0] == _chunks()[0]["text"]
    assert result["metadatas"][0][0]["doc_name"] == "guidelines"
    assert result["metadatas"][0][0]["section_title"] == "3.5 Management of Asthma"


def test_chroma_store_query_returns_metadata_fields(store: ChromaStore) -> None:
    store.add_chunks(_chunks())

    result = store.query("combination therapy", n_results=1)

    assert result["ids"][0] == ["guidelines_2_0"]
    meta = result["metadatas"][0][0]
    assert set(meta) == {
        "doc_name",
        "page_number",
        "chunk_word_count",
        "chunk_id",
        "section_title",
    }


def test_chroma_store_delete_collection(store: ChromaStore, tmp_path: Path) -> None:
    store.add_chunks(_chunks())

    store.delete_collection()

    with pytest.raises(chromadb.errors.NotFoundError):
        store.query("controller medication", n_results=1)

    # A fresh store on the same path starts clean and works again.
    fresh = ChromaStore(Settings(chroma_path=tmp_path))
    fresh.add_chunks(_chunks())
    assert fresh.query("controller medication", n_results=1)["ids"][0] == [
        "guidelines_1_0"
    ]


def test_chroma_store_add_chunks_batches_large_collections(store: ChromaStore) -> None:
    """Given more chunks than the batch size, when adding, then all chunks are indexed."""
    chunks = [
        {
            "chunk_id": f"c{i}",
            "text": f"Asthma guideline fact number {i} for testing.",
            "doc_name": "guidelines",
            "page_number": i + 1,
            "chunk_word_count": 7,
            "section_title": "Facts",
        }
        for i in range(7)
    ]

    store.add_chunks(chunks, batch_size=3)

    result = store.query("asthma guideline fact", n_results=7)
    assert set(result["ids"][0]) == {f"c{i}" for i in range(7)}


def test_chroma_store_handles_none_section_title(store: ChromaStore) -> None:
    """Given chunks without a section heading, when adding, then metadata stores an empty string."""
    chunks = [
        {
            "chunk_id": "g_9_0",
            "text": "Regular review of inhaler technique is essential for asthma control.",
            "doc_name": "guidelines",
            "page_number": 9,
            "chunk_word_count": 11,
            "section_title": None,
        }
    ]

    store.add_chunks(chunks)

    result = store.query("inhaler technique", n_results=1)
    assert result["ids"][0] == ["g_9_0"]
    assert result["metadatas"][0][0]["section_title"] == ""


def test_delete_collection_if_exists_is_noop_when_missing(tmp_path: Path) -> None:
    delete_collection_if_exists(Settings(chroma_path=tmp_path))


def test_delete_collection_if_exists_removes_collection(tmp_path: Path) -> None:
    settings = Settings(chroma_path=tmp_path)
    ChromaStore(settings).add_chunks(_chunks())

    delete_collection_if_exists(settings)

    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    assert client.list_collections() == []
