"""Tests for the authoritative asthma sources registry.

Locks the contract between ``sources.yaml`` and ``asthma_rag.sources``:
the registry must distinguish open-access (fetchable) sources from
rights-reserved (cite-only) sources, and reject malformed entries.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from asthma_rag.sources import (
    Source,
    get_cite_only_sources,
    get_open_access_urls,
    load_sources,
)

SOURCES_YAML = Path(__file__).resolve().parents[1] / "sources.yaml"

ALL_SOURCE_IDS = {
    "gina_2026",
    "nhlbi_2020",
    "nhlbi_focused_2020",
    "who_2026",
    "nice_ng245",
    "bts_sign_158",
    "ats_ers_severe_2014",
    "ers_ats_severe_2020",
    "global_asthma_report_2022",
}

OPEN_ACCESS_IDS = ALL_SOURCE_IDS - {"ers_ats_severe_2020", "global_asthma_report_2022"}
CITE_ONLY_IDS = {"ers_ats_severe_2020", "global_asthma_report_2022"}


def _load() -> list[Source]:
    return load_sources(SOURCES_YAML)


def test_load_sources_returns_all_nine_authoritative_sources() -> None:
    """Given sources.yaml, when loaded, then all 9 authoritative sources are present."""
    sources = _load()

    assert len(sources) == 9
    assert {s.id for s in sources} == ALL_SOURCE_IDS


def test_open_access_urls_excludes_rights_reserved_sources() -> None:
    """Given all sources, when filtering open access, then rights-reserved (fetch=false) sources are excluded."""
    open_access = get_open_access_urls(_load())

    open_ids = {s.id for s in open_access}
    assert open_ids == OPEN_ACCESS_IDS
    assert CITE_ONLY_IDS.isdisjoint(open_ids)
    for s in open_access:
        assert s.fetch is True


def test_cite_only_sources_includes_only_rights_reserved_sources() -> None:
    """Given all sources, when filtering cite-only, then only fetch=false rights-reserved sources are returned."""
    cite_only = get_cite_only_sources(_load())

    assert {s.id for s in cite_only} == CITE_ONLY_IDS
    for s in cite_only:
        assert s.fetch is False


def test_load_sources_raises_when_required_field_missing(tmp_path: Path) -> None:
    """Given a yaml entry missing the required 'title' field, when loaded, then an error is raised."""
    bad_yaml = tmp_path / "missing_title.yaml"
    bad_yaml.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "bad_source",
                    "publisher": "Example Publisher",
                    "year": 2026,
                    "url": "https://example.com",
                    "license": "open-access",
                    "fetch": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises((TypeError, ValueError)):
        load_sources(bad_yaml)
