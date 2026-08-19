"""PDF ingestion: extract clean pages and split them into retrievable chunks.

Pipeline stage: ``data/raw/*.pdf`` -> ``data/chunks/chunks.json``.

Run with::

    python -m asthma_rag.ingest --pdf-dir data/raw --out data/chunks/chunks.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from asthma_rag.cleaning.section import extract_section_title
from asthma_rag.cleaning.text import clean_text
from asthma_rag.config import get_settings

_MIN_CHUNK_WORDS = 20


class PageData(TypedDict):
    doc_name: str
    page_number: int
    text: str
    section_title: str | None


class ChunkData(TypedDict):
    chunk_id: str
    text: str
    doc_name: str
    page_number: int
    chunk_word_count: int
    section_title: str | None


def extract_pages(pdf_path: Path) -> list[PageData]:
    """Extract, clean, and section-tag every page of ``pdf_path``."""
    doc_name = pdf_path.stem
    pages: list[PageData] = []
    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            raw_text = page.get_text()
            text = clean_text(raw_text)
            if not text:
                continue
            pages.append(
                PageData(
                    doc_name=doc_name,
                    page_number=page_number,
                    text=text,
                    section_title=extract_section_title(text),
                )
            )
    return pages


def chunk_pages(
    pages: list[PageData],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[ChunkData]:
    """Split each page into overlapping chunks, dropping chunks under 20 words."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks: list[ChunkData] = []
    for page in pages:
        for index, piece in enumerate(splitter.split_text(page["text"])):
            word_count = len(piece.split())
            if word_count < _MIN_CHUNK_WORDS:
                continue
            chunks.append(
                ChunkData(
                    chunk_id=f"{page['doc_name']}_{page['page_number']}_{index}",
                    text=piece,
                    doc_name=page["doc_name"],
                    page_number=page["page_number"],
                    chunk_word_count=word_count,
                    section_title=page["section_title"],
                )
            )
    return chunks


def _print_stats(chunks: list[ChunkData]) -> None:
    by_doc: dict[str, list[ChunkData]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["doc_name"], []).append(chunk)
    for doc_name, doc_chunks in sorted(by_doc.items()):
        total_words = sum(c["chunk_word_count"] for c in doc_chunks)
        print(f"{doc_name}: {len(doc_chunks)} chunks, {total_words} words")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk asthma guideline PDFs.")
    parser.add_argument("--pdf-dir", type=Path, default=get_settings().data_raw_dir)
    parser.add_argument("--out", type=Path, default=get_settings().data_chunks_dir / "chunks.json")
    args = parser.parse_args()

    all_chunks: list[ChunkData] = []
    for pdf_path in sorted(args.pdf_dir.glob("*.pdf")):
        all_chunks.extend(chunk_pages(extract_pages(pdf_path)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_chunks)} chunks to {args.out}")
    _print_stats(all_chunks)


if __name__ == "__main__":
    main()