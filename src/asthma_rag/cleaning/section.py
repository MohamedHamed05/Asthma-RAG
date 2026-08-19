"""Section-heading detection for cleaned guideline page text.

Strategies are tried in priority order:

1. a numbered heading such as ``3.5 Management of Asthma``,
2. an ALL-CAPS line,
3. a short line that ends without punctuation (bold-line fallback).
"""

from __future__ import annotations

import re

_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*\s+[A-Z]")
_MAX_FALLBACK_LEN = 60
_END_PUNCTUATION = ". , ; : ! ?"
_COPYRIGHT_MARKERS = ("COPYRIGHTED MATERIAL", "DO NOT COPY OR DISTRIBUTE")


def _is_copyright_footer(line: str) -> bool:
    upper = line.strip().upper()
    return all(marker in upper for marker in _COPYRIGHT_MARKERS)


def _is_all_caps(line: str) -> bool:
    return line.isupper() and any(ch.isalpha() for ch in line)


def extract_section_title(text: str) -> str | None:
    """Return the most likely section heading in ``text``, or None."""
    if not text:
        return None
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    for line in lines:
        if _is_copyright_footer(line):
            continue
        if _NUMBERED_HEADING.search(line):
            return line

    for line in lines:
        if _is_copyright_footer(line):
            continue
        if _is_all_caps(line):
            return line

    for line in lines:
        if _is_copyright_footer(line):
            continue
        if len(line) <= _MAX_FALLBACK_LEN and not line.endswith(tuple(_END_PUNCTUATION)):
            return line

    return None
