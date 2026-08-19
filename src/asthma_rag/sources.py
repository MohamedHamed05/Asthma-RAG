"""Authoritative asthma sources registry.

Loads ``sources.yaml`` into typed ``Source`` records and provides filters
that separate open-access (fetchable) sources from rights-reserved
(cite-only) sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class Source:
    """A single authoritative asthma guideline source."""

    id: str
    title: str
    publisher: str
    year: int
    url: str
    license: str
    fetch: bool
    note: str | None = None


def load_sources(path: Path = Path("sources.yaml")) -> list[Source]:
    """Load and validate all sources from a YAML registry file.

    Raises ``TypeError``/``ValueError`` when an entry is missing a required
    field or the file does not contain a list of entries.
    """
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of sources in {path}, got {type(raw).__name__}")
    return [Source(**entry) for entry in raw]


def get_open_access_urls(sources: list[Source]) -> list[Source]:
    """Return sources that are open-access and safe to fetch (``fetch=true``)."""
    return [s for s in sources if s.fetch]


def get_cite_only_sources(sources: list[Source]) -> list[Source]:
    """Return sources that are rights-reserved or paywalled (``fetch=false``)."""
    return [s for s in sources if not s.fetch]