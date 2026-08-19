"""Unit tests for asthma_rag.cleaning.section.extract_section_title."""

from asthma_rag.cleaning.section import extract_section_title


def test_extract_section_title_numbered_heading() -> None:
    text = "Some preamble text.\n3.5 Management of Asthma\nBody follows."
    assert extract_section_title(text) == "3.5 Management of Asthma"


def test_extract_section_title_numbered_multi_part_heading() -> None:
    text = "5.2.1 Severe Asthma\nBody follows."
    assert extract_section_title(text) == "5.2.1 Severe Asthma"


def test_extract_section_title_all_caps_heading() -> None:
    text = "INTRODUCTION\nBody text here."
    assert extract_section_title(text) == "INTRODUCTION"


def test_extract_section_title_skips_copyright_footer() -> None:
    text = "Body text here.\nCOPYRIGHTED MATERIAL - DO NOT COPY OR DISTRIBUTE"
    assert extract_section_title(text) is None


def test_extract_section_title_bold_line_fallback() -> None:
    text = "Body text here.\nTreatment Goals\nMore text."
    assert extract_section_title(text) == "Treatment Goals"


def test_extract_section_title_returns_none() -> None:
    text = (
        "This is just a long paragraph of regular body text without any heading. "
        "It continues."
    )
    assert extract_section_title(text) is None
