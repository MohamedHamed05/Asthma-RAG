"""Page-level validation for extracted guideline pages."""

from __future__ import annotations

from asthma_rag.cleaning.section import extract_section_title

_LOW_WORD_THRESHOLD = 10
_CONTROL_RATIO_THRESHOLD = 0.9
_STREAK_THRESHOLD = 5


def _word_count(text: str) -> int:
    return len(text.split())


def _control_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    control = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\t")
    return control / len(text)


def _page_label(page: dict, index: int) -> str:
    label = str(page.get("page_number", index))
    doc_name = page.get("doc_name")
    return f"{doc_name}:{label}" if doc_name else label


def _streak_warning(pages: list[dict], start: int, end: int) -> str:
    count = end - start + 1
    start_label = _page_label(pages[start], start)
    end_label = _page_label(pages[end], end)
    if start_label == end_label:
        return f"Page {start_label}: no section title in {count} consecutive pages"
    return (
        f"Pages {start_label}-{end_label}: "
        f"{count} consecutive pages without a section title"
    )


def validate_pages(pages: list[dict]) -> list[str]:
    """Return human-readable warnings for low-quality pages.

    Warns when a page has fewer than 10 words, when more than 90% of its
    characters are control characters, and when 5+ consecutive pages lack a
    section title.
    """
    warnings: list[str] = []
    run_start: int | None = None

    for index, page in enumerate(pages):
        label = _page_label(page, index)
        text = page.get("text", "")

        words = _word_count(text)
        if words < _LOW_WORD_THRESHOLD:
            warnings.append(
                f"Page {label}: fewer than {_LOW_WORD_THRESHOLD} words ({words})"
            )

        if _control_char_ratio(text) > _CONTROL_RATIO_THRESHOLD:
            warnings.append(f"Page {label}: >90% control characters")

        if extract_section_title(text) is not None:
            if run_start is not None and index - run_start >= _STREAK_THRESHOLD:
                warnings.append(_streak_warning(pages, run_start, index - 1))
            run_start = None
        elif run_start is None:
            run_start = index

    if run_start is not None and len(pages) - run_start >= _STREAK_THRESHOLD:
        warnings.append(_streak_warning(pages, run_start, len(pages) - 1))

    return warnings
