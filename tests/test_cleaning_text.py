"""Unit tests for asthma_rag.cleaning.text.clean_text."""

from asthma_rag.cleaning.text import clean_text


def test_clean_text_normalizes_line_endings() -> None:
    text = "line1\r\nline2\rline3\nline4"
    assert clean_text(text) == "line1\nline2\nline3\nline4"


def test_clean_text_removes_standalone_page_numbers() -> None:
    text = "Header\n\n42\n\nBody"
    assert clean_text(text) == "Header\n\nBody"


def test_clean_text_removes_copyright_footer() -> None:
    text = "Text\nCOPYRIGHTED MATERIAL - DO NOT COPY OR DISTRIBUTE\nMore"
    assert clean_text(text) == "Text\n\nMore"


def test_clean_text_removes_standalone_bullets_and_hyphens() -> None:
    text = "Item list:\n•\n●\n▪\n◦\n-\nDone"
    assert clean_text(text) == "Item list:\n\nDone"


def test_clean_text_strips_trailing_whitespace() -> None:
    text = "hello   \nworld\t\n  end"
    assert clean_text(text) == "hello\nworld\n  end"


def test_clean_text_collapses_blank_lines() -> None:
    text = "a\n\n\n\n\nb\n\n\nc"
    assert clean_text(text) == "a\n\nb\n\nc"


def test_clean_text_removes_toc_leader_dots() -> None:
    text = "Acknowledgements .... 5\nIntroduction .... 10\n"
    assert clean_text(text) == "Acknowledgements\nIntroduction"


def test_clean_text_removes_repeated_all_caps_headers() -> None:
    text = (
        "GLOBAL STRATEGY FOR ASTHMA MANAGEMENT\n"
        "Introduction\n"
        "GLOBAL STRATEGY FOR ASTHMA MANAGEMENT\n"
        "Body text here.\n"
    )
    assert clean_text(text) == "Introduction\n\nBody text here."


def test_clean_text_removes_repeated_title_case_headers() -> None:
    text = (
        "Global Strategy For Asthma\n"
        "Some body content\n"
        "Global Strategy For Asthma\n"
        "More content.\n"
    )
    assert clean_text(text) == "Some body content\n\nMore content."
