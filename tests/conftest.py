"""Shared test fixtures.

The real .env contains live Langfuse keys, so observability is force-disabled
for every test — otherwise pytest runs would export traces to the cloud.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from asthma_rag import observability


@pytest.fixture(autouse=True)
def _disable_langfuse(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make get_langfuse() return None for the duration of each test."""
    observability.reset_langfuse_cache()
    monkeypatch.setattr(observability, "_initialized", True)
    monkeypatch.setattr(observability, "_client", None)
    yield
    observability.reset_langfuse_cache()
