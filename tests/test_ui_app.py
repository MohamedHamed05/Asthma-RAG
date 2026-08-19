"""Tests for the Gradio UI app."""

from __future__ import annotations

from typing import Any

import gradio as gr

from asthma_rag.ui.app import create_app, respond


class _MockPipeline:
    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result

    def query(self, _question: str) -> dict[str, Any]:
        return self._result


def test_respond_returns_answer_and_video() -> None:
    """Given a result with answer and video HTML, respond returns both."""
    pipeline = _MockPipeline(
        {
            "final_answer": "Use a spacer and rinse your mouth after.",
            "video_html": "<iframe src='youtube.com/embed/x'></iframe>",
        }
    )

    answer, video = respond("How do I use my inhaler?", pipeline)

    assert answer == "Use a spacer and rinse your mouth after."
    assert video == "<iframe src='youtube.com/embed/x'></iframe>"


def test_respond_returns_empty_video_when_missing() -> None:
    """Given a result without video HTML, respond returns an empty video string."""
    pipeline = _MockPipeline({"final_answer": "Asthma is chronic."})

    answer, video = respond("What is asthma?", pipeline)

    assert answer == "Asthma is chronic."
    assert video == ""


def test_create_app_returns_gradio_blocks() -> None:
    """Given a pipeline, create_app returns a Gradio Blocks instance."""
    pipeline = _MockPipeline({"final_answer": "Answer.", "video_html": ""})

    app = create_app(pipeline)

    assert isinstance(app, gr.Blocks)


def test_respond_returns_generic_error_on_pipeline_failure() -> None:
    """Given a pipeline that raises, respond returns a generic message without leaking details."""
    class _FailingPipeline:
        def query(self, _question: str) -> dict[str, Any]:
            raise RuntimeError("boom at C:\\secret\\path")

    answer, video = respond("What is asthma?", _FailingPipeline())

    assert answer.startswith("An error occurred")
    assert "boom" not in answer
    assert "secret" not in answer
    assert video == ""


def test_respond_empty_query_returns_prompt_without_calling_pipeline() -> None:
    """Given a blank query, respond returns the prompt message and skips the pipeline."""
    class _ExplodingPipeline:
        def query(self, _question: str) -> dict[str, Any]:
            raise AssertionError("pipeline must not be called for an empty query")

    answer, video = respond("   ", _ExplodingPipeline())

    assert answer == "Please enter an asthma-related question."
    assert video == ""
