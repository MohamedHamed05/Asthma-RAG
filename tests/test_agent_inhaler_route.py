"""Tests for the inhaler route node in the LangGraph agent.

RED phase: these tests should FAIL before ``inhaler_route`` is defined in
``src/asthma_rag/agent/inhaler.py``.
"""

from __future__ import annotations

from asthma_rag.agent.inhaler import inhaler_route


# ---------------------------------------------------------------------------
# Inhaler-technique queries
# ---------------------------------------------------------------------------

class TestInhalerRoute:
    """inhaler_route should dispatch technique queries to the inhaler branch."""

    def test_inhaler_query_routes_to_inhaler(self) -> None:
        state = {"query": "How do I use my inhaler?"}
        result = inhaler_route(state)

        assert result["route"] == "inhaler"

    def test_inhaler_query_has_explanation(self) -> None:
        state = {"query": "inhaler technique"}
        result = inhaler_route(state)

        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 0

    def test_inhaler_query_has_youtube_embed(self) -> None:
        state = {"query": "How do I use a spacer?"}
        result = inhaler_route(state)

        assert "youtube.com/embed/" in result["video_html"]

    def test_state_passes_through_extra_keys(self) -> None:
        state = {"query": "inhaler demonstration", "user_id": 42}
        result = inhaler_route(state)

        assert result["route"] == "inhaler"


# ---------------------------------------------------------------------------
# Non-inhaler queries
# ---------------------------------------------------------------------------

class TestRetrieveRoute:
    """Non-inhaler queries should fall through to retrieval."""

    def test_non_inhaler_query_routes_to_retrieve(self) -> None:
        state = {"query": "What is asthma?"}
        result = inhaler_route(state)

        assert result == {"route": "retrieve"}

    def test_inhaler_query_without_technique_routes_to_retrieve(self) -> None:
        state = {"query": "inhaler side effects"}
        result = inhaler_route(state)

        assert result == {"route": "retrieve"}

    def test_negated_technique_routes_to_retrieve(self) -> None:
        state = {"query": "I will not use the inhaler"}
        result = inhaler_route(state)

        assert result == {"route": "retrieve"}
