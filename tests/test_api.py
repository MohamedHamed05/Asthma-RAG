"""Tests for the FastAPI backend integration endpoint."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

import main


class _StubPipeline:
    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result

    def query(self, _question: str) -> dict[str, Any]:
        return self._result


def test_ask_endpoint_returns_answer_and_citations(
    monkeypatch,
) -> None:
    """Given a real answer text, the API returns answer plus parsed citations."""
    monkeypatch.setattr(
        main,
        "get_pipeline",
        lambda: _StubPipeline(
            {
                "final_answer": (
                    "Asthma is chronic.\n\n"
                    "**Sources**\n"
                    "- GINA-2026-Strategy-Report-WMS.pdf, p. 25\n"
                    "- NICE-NG80.pdf, p. 14\n"
                ),
                "route": "retrieve",
                "video_html": "",
            }
        ),
    )
    client = TestClient(main.app)

    response = client.post("/api/ask", json={"question": "What is asthma?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": (
            "Asthma is chronic.\n\n"
            "**Sources**\n"
            "- GINA-2026-Strategy-Report-WMS.pdf, p. 25\n"
            "- NICE-NG80.pdf, p. 14"
        ),
        "citations": [
            "GINA-2026-Strategy-Report-WMS.pdf, p. 25",
            "NICE-NG80.pdf, p. 14",
        ],
        "route": "retrieve",
        "video_html": "",
    }


def test_ask_endpoint_requires_non_empty_question() -> None:
    """Given an empty question payload, FastAPI validation rejects it."""
    client = TestClient(main.app)

    response = client.post("/api/ask", json={"question": ""})

    assert response.status_code == 422


def test_health_endpoint_returns_ok() -> None:
    """Health endpoint responds with status ok."""
    client = TestClient(main.app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
