"""Tests for the inhaler intent detector and payload builder.

RED phase: these tests should all FAIL before src/asthma_rag/agent/inhaler.py exists.
"""

from __future__ import annotations

import pytest

from asthma_rag.agent.inhaler import (
    INHALER_KEYWORDS,
    TECHNIQUE_KEYWORDS,
    build_inhaler_payload,
    detect_inhaler_intent,
)


# ---------------------------------------------------------------------------
# Regex sanity checks
# ---------------------------------------------------------------------------

class TestRegexPatterns:
    """Verify the compiled regex patterns exist and are usable."""

    def test_inhaler_keywords_is_compiled(self) -> None:
        assert INHALER_KEYWORDS.search("inhaler") is not None

    def test_technique_keywords_is_compiled(self) -> None:
        assert TECHNIQUE_KEYWORDS.search("technique") is not None


# ---------------------------------------------------------------------------
# True-positive cases: inhaler + technique present
# ---------------------------------------------------------------------------

class TestDetectInhalerIntentTruePositives:
    """Queries that SHOULD trigger the inhaler intent (both keyword sets present)."""

    @pytest.mark.parametrize(
        "query",
        [
            "How do I use my inhaler?",
            "inhaler technique",
            "how to use a spacer",
            "puffer demonstration",
            "MDI steps",
            "dry powder inhaler use",
            "nebulizer use",
            "How should I properly use my puffer?",
            "Show me the technique for my inhaler",
            "How do you use a metered dose inhaler correctly?",
            "Demonstrate how to use a nebulizer",
            "Use of dry powder inhaler properly",
        ],
    )
    def test_true_positive(self, query: str) -> None:
        assert detect_inhaler_intent(query) is True, (
            f"Expected True for query: {query!r}"
        )


# ---------------------------------------------------------------------------
# True-negative cases: missing inhaler keyword, technique keyword, or both
# ---------------------------------------------------------------------------

class TestDetectInhalerIntentTrueNegatives:
    """Queries that should NOT trigger the inhaler intent."""

    @pytest.mark.parametrize(
        "query",
        [
            "What is asthma?",
            "What triggers asthma?",
            "asthma symptoms",
            "COPD inhaler",            # technique keyword absent
            "inhaler side effects",     # technique keyword absent
            "Define puffer",            # technique keyword absent
            "Can you explain asthma control?",
            "What are the side effects of albuterol?",
            "How does the spacer work?",
            "Tell me about pulmonary function tests",
        ],
    )
    def test_true_negative(self, query: str) -> None:
        assert detect_inhaler_intent(query) is False, (
            f"Expected False for query: {query!r}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Negation, partial matches, and casing should not create false positives."""

    def test_negation_does_not_trigger(self) -> None:
        assert detect_inhaler_intent("I will not use the inhaler") is False

    def test_partial_inhaler_word_no_trigger(self) -> None:
        # "inhaler" is a full word match; partial substrings like "inhal" shouldn't count.
        assert detect_inhaler_intent("inhal technique") is False

    def test_case_insensitive(self) -> None:
        assert detect_inhaler_intent("HOW DO I USE MY INHALER?") is True

    def test_empty_string(self) -> None:
        assert detect_inhaler_intent("") is False


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

class TestBuildInhalerPayload:
    """build_inhaler_payload should return explanation + video_html."""

    def test_returns_dict_with_required_keys(self) -> None:
        payload = build_inhaler_payload("How do I use my inhaler?")
        assert isinstance(payload, dict)
        assert "explanation" in payload
        assert "video_html" in payload

    def test_explanation_mentions_inhaler(self) -> None:
        payload = build_inhaler_payload("inhaler technique")
        assert "inhaler" in payload["explanation"].lower()

    def test_explanation_mentions_demonstration(self) -> None:
        payload = build_inhaler_payload("inhaler technique")
        assert "demonstration" in payload["explanation"].lower()

    def test_video_html_is_iframe(self) -> None:
        payload = build_inhaler_payload("inhaler technique")
        assert "<iframe" in payload["video_html"]
        assert "youtube.com/embed/" in payload["video_html"]
