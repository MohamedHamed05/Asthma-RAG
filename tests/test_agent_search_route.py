"""Tests for the route_query node in the LangGraph agent.

Precedence: inhaler > video_search > search > retrieve.
"""

from __future__ import annotations

from typing import Any

from asthma_rag.agent.search_route import route_query


def _state(query: str, **extra: Any) -> dict[str, Any]:
    return {"query": query, **extra}


# ---------------------------------------------------------------------------
# Inhaler precedence
# ---------------------------------------------------------------------------


class TestInhalerPrecedence:
    """An inhaler-technique query must always route to inhaler, overriding other signals."""

    def test_inhaler_technique_routes_to_inhaler(self) -> None:
        result = route_query(_state("How do I use my inhaler?"))
        assert result["route"] == "inhaler"

    def test_inhaler_query_also_mentioning_price_routes_to_inhaler(self) -> None:
        """Drug-price keywords must NOT hijack an inhaler-technique query."""
        result = route_query(_state("How do I use my inhaler? price in Egypt"))
        assert result["route"] == "inhaler"

    def test_inhaler_query_also_mentioning_video_routes_to_inhaler(self) -> None:
        """Video keywords must NOT hijack an inhaler-technique query either."""
        result = route_query(_state("spacer demonstration step by step"))
        assert result["route"] == "inhaler"

    def test_inhaler_query_does_not_contain_explanation_or_video_html(self) -> None:
        """The route_query layer must NOT duplicate the inhaler payload; that belongs in inhaler_route."""
        result = route_query(_state("inhaler technique demo"))
        # route_query just sets the route name; inhaler_route handles payload building.
        assert "explanation" not in result
        assert "video_html" not in result


# ---------------------------------------------------------------------------
# video_search
# ---------------------------------------------------------------------------


class TestVideoSearchRoute:
    """VIDEO_KW alone (no inhaler intent) should route to video_search."""

    def test_tutorial_keyword_routes_to_video(self) -> None:
        result = route_query(_state("tutorial on asthma"))
        assert result["route"] == "video_search"

    def test_demo_keyword_routes_to_video(self) -> None:
        result = route_query(_state("show me how to manage asthma"))
        assert result["route"] == "video_search"

    def test_nebulizer_keyword_routes_to_video(self) -> None:
        result = route_query(_state("nebulizer treatment"))
        assert result["route"] == "video_search"

    def test_step_by_step_routes_to_video(self) -> None:
        result = route_query(_state("step by step breathing exercises"))
        assert result["route"] == "video_search"

    def test_spacer_keyword_routes_to_video(self) -> None:
        result = route_query(_state("spacer for child"))
        assert result["route"] == "video_search"

    def test_video_keyword_routes_to_video(self) -> None:
        result = route_query(_state("video on asthma triggers"))
        assert result["route"] == "video_search"


# ---------------------------------------------------------------------------
# search (drug-price)
# ---------------------------------------------------------------------------


class TestSearchRoute:
    """Drug-price keywords + asthma drug keywords => search."""

    def test_price_with_ventolin_routes_to_search(self) -> None:
        result = route_query(_state("ventolin price in Egypt"))
        assert result["route"] == "search"

    def test_cost_with_seretide_routes_to_search(self) -> None:
        result = route_query(_state("seretide cost"))
        assert result["route"] == "search"

    def test_how_much_with_symbicort_routes_to_search(self) -> None:
        result = route_query(_state("how much is symbicort"))
        assert result["route"] == "search"

    def test_egp_with_salmeterol_routes_to_search(self) -> None:
        result = route_query(_state("salmeterol EGP"))
        assert result["route"] == "search"

    def test_arabic_currency_keyword_with_inhaler_routes_to_search(self) -> None:
        result = route_query(_state("inhaler جنيه"))
        assert result["route"] == "search"

    def test_arabic_country_keyword_with_budesonide_routes_to_search(self) -> None:
        result = route_query(_state("budesonide مصر"))
        assert result["route"] == "search"


class TestSearchNotTriggered:
    """Search must require BOTH a price signal and a drug signal."""

    def test_price_alone_without_drug_routes_to_retrieve(self) -> None:
        result = route_query(_state("how much does asthma treatment cost in general"))
        assert result["route"] == "retrieve"

    def test_drug_alone_without_price_routes_to_retrieve(self) -> None:
        result = route_query(_state("salbutamol side effects"))
        assert result["route"] == "retrieve"

    def test_no_drug_no_price_routes_to_retrieve(self) -> None:
        result = route_query(_state("What is asthma?"))
        assert result["route"] == "retrieve"


# ---------------------------------------------------------------------------
# Default retrieve fallback
# ---------------------------------------------------------------------------


class TestRetrieveFallback:
    """Anything that doesn't match earlier routes must fall through to retrieve."""

    def test_unrelated_query_routes_to_retrieve(self) -> None:
        result = route_query(_state("What are the symptoms of asthma?"))
        assert result["route"] == "retrieve"

    def test_state_with_extra_keys_still_routes(self) -> None:
        result = route_query(_state("ventolin price", user_id=42, retrieved={}))
        assert result["route"] == "search"


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------


class TestReturnShape:
    """route_query must always return a dict with at least a 'route' key."""

    def test_returns_only_route_key(self) -> None:
        result = route_query(_state("What is asthma?"))
        assert set(result.keys()) == {"route"}
