"""Text cleaning for extracted asthma guideline pages.

This module preserves the original per-line regex pipeline from the
prototype notebook (``hackathon (2).py``) and adds two heuristics:

* removal of running headers/footers (short all-caps or title-case lines
  that repeat within the page text),
* removal of TOC leader dots (``.... 25`` patterns).
"""

from __future__ import annotations

import re

_BULLET_CHARS = "\u2022\u25cf\u25aa\u25e6"  # • ● ▪ ◦
_PAGE_NUMBER = re.compile(r"(?m)^\s*\d+\s*$")
_COPYRIGHT_FOOTER = re.compile(
    r"(?mi)^\s*COPYRIGHTED MATERIAL\s*-\s*DO NOT COPY OR DISTRIBUTE\s*$"
)
_STANDALONE_BULLET = re.compile(rf"(?m)^\s*[{_BULLET_CHARS}]\s*$")
_STANDALONE_HYPHEN = re.compile(r"(?m)^\s*-\s*$")
_TOC_LEADER = re.compile(r"(?m)\.{2,}\s*\d+\s*$")
_TRAILING_WHITESPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")

_MAX_HEADER_LEN = 60
_MIN_HEADER_REPEATS = 2


def _is_header_like(line: str) -> bool:
    """True when a line looks like a running header/footer."""
    candidate = line.strip()
    if not candidate or len(candidate) > _MAX_HEADER_LEN:
        return False
    if not any(ch.isalpha() for ch in candidate):
        return False
    return candidate.isupper() or candidate.istitle()


def _remove_running_headers(lines: list[str]) -> list[str]:
    """Blank every occurrence of a repeated header-like line."""
    counts: dict[str, int] = {}
    for line in lines:
        if _is_header_like(line):
            key = line.strip()
            counts[key] = counts.get(key, 0) + 1
    repeated = {key for key, count in counts.items() if count >= _MIN_HEADER_REPEATS}
    if not repeated:
        return lines
    return ["" if line.strip() in repeated else line for line in lines]


def clean_text(text: str) -> str:
    """Clean raw text extracted from a guideline PDF page."""
    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove standalone page numbers.
    text = _PAGE_NUMBER.sub("", text)

    # Remove the copyright footer line.
    text = _COPYRIGHT_FOOTER.sub("", text)

    # Remove standalone bullet characters and hyphens.
    text = _STANDALONE_BULLET.sub("", text)
    text = _STANDALONE_HYPHEN.sub("", text)

    # Remove TOC leader dots ("Introduction .... 10" -> "Introduction").
    text = _TOC_LEADER.sub("", text)

    # Remove repeated running headers/footers.
    lines = _remove_running_headers(text.split("\n"))
    text = "\n".join(lines)

    # Strip trailing whitespace per line.
    text = _TRAILING_WHITESPACE.sub("", text)

    # Collapse runs of three or more blank lines down to two.
    text = _BLANK_RUN.sub("\n\n", text)

    return text.strip()
