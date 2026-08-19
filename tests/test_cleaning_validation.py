"""Unit tests for asthma_rag.cleaning.validation.validate_pages."""

from asthma_rag.cleaning.validation import validate_pages


def test_validate_pages_warns_on_low_word_count() -> None:
    pages = [
        {"page_number": 0, "text": "just a few words"},
        {
            "page_number": 1,
            "text": "A fully formed page with many words that should not trigger "
            "any warning at all.",
        },
    ]
    warnings = validate_pages(pages)
    assert len(warnings) == 1
    assert "Page 0" in warnings[0]
    assert "fewer than 10 words" in warnings[0]


def test_validate_pages_warns_on_control_char_dominance() -> None:
    text = "\x00" * 200 + "a b c d e f g h i j k"
    pages = [{"page_number": 0, "text": text}]
    warnings = validate_pages(pages)
    assert len(warnings) == 1
    assert "control" in warnings[0].lower()


def test_validate_pages_warns_on_section_title_streak() -> None:
    pages = [
        {
            "page_number": i,
            "text": "This page contains some regular body text that should not "
            "trigger a low word warning at all.",
        }
        for i in range(6)
    ]
    warnings = validate_pages(pages)
    assert len(warnings) == 1
    assert "section title" in warnings[0].lower()


def test_validate_pages_no_warnings_for_clean_pages() -> None:
    pages = [
        {
            "page_number": i,
            "text": "Section One\nLots of body text that fills the page with "
            "enough words to avoid every possible warning in this validator.",
        }
        for i in range(4)
    ]
    assert validate_pages(pages) == []
