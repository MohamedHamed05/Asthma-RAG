"""Tests for asthma_rag.ingest: PDF page extraction and chunking."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from asthma_rag.ingest import chunk_pages, extract_pages

_LONG_BODY = (
    "Inhaled corticosteroids are the preferred controller medication for "
    "persistent asthma in both adults and adolescents. They reduce airway "
    "inflammation, improve lung function, and decrease the frequency of "
    "exacerbations. Long-acting beta-agonists should only be used in "
    "combination with inhaled corticosteroids. Regular review of inhaler "
    "technique is essential to ensure adequate drug delivery to the airways. "
    "A written asthma action plan should be provided to every patient. "
) * 2  # ~96 words, well above the 20-word floor and under one 1000-char chunk


def _write_sample_pdf(path: Path) -> None:
    """Create a two-page PDF: a long heading page and a tiny fragment page."""
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_textbox(
        pymupdf.Rect(72, 72, 523, 742),
        "3.5 Management of Asthma\n\n" + _LONG_BODY,
    )
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Only a few words.")
    doc.save(path)
    doc.close()


def test_extract_pages_returns_cleaned_text_and_section_title(tmp_path: Path) -> None:
    pdf_path = tmp_path / "guidelines.pdf"
    _write_sample_pdf(pdf_path)

    pages = extract_pages(pdf_path)

    assert len(pages) == 2
    first = pages[0]
    assert first["doc_name"] == "guidelines"
    assert first["page_number"] == 1
    assert "Management of Asthma" in first["text"]
    assert first["section_title"] == "3.5 Management of Asthma"
    assert "COPYRIGHTED MATERIAL" not in first["text"]


def test_chunk_pages_drops_short_chunks_and_preserves_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "guidelines.pdf"
    _write_sample_pdf(pdf_path)
    pages = extract_pages(pdf_path)

    chunks = chunk_pages(pages)

    # Only the long page survives the 20-word floor.
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["chunk_id"] == "guidelines_1_0"
    assert chunk["doc_name"] == "guidelines"
    assert chunk["page_number"] == 1
    assert chunk["section_title"] == "3.5 Management of Asthma"
    assert chunk["chunk_word_count"] >= 20
    assert len(chunk["text"].split()) == chunk["chunk_word_count"]
