"""Gradio UI for the asthma RAG agent.

The interface exposes a single question box, an answer panel, and an optional
embedded demonstration video (returned for inhaler-technique queries).
"""

from __future__ import annotations

import gradio as gr

from asthma_rag.pipeline import Pipeline

_DISCLAIMER = (
    "**Disclaimer:** This tool is an educational prototype for exploring "
    "retrieval-augmented generation over public asthma guideline documents. "
    "It is **not** a medical device and does **not** provide medical advice. "
    "Always consult a qualified healthcare professional for diagnosis or "
    "treatment decisions."
)


def respond(query: str, pipeline: Pipeline) -> tuple[str, str]:
    """Ask the pipeline a question and return (answer_markdown, video_html)."""
    if not query.strip():
        return ("Please enter an asthma-related question.", "")
    try:
        result = pipeline.query(query)
    except Exception:
        # Never surface raw tracebacks (paths, dependency details) to the UI.
        return (
            "An error occurred while processing your question. Please try again.",
            "",
        )
    answer: str = result.get("final_answer", "")
    video_html: str = result.get("video_html", "") or ""
    return answer, video_html


def create_app(pipeline: Pipeline | None = None) -> gr.Blocks:
    """Build and return the Gradio Blocks app wired to ``pipeline``."""
    pipeline = pipeline or Pipeline()

    with gr.Blocks(title="Asthma Guidelines RAG") as app:
        gr.Markdown("# Asthma Guidelines RAG")
        gr.Markdown(
            "Ask a question about asthma diagnosis, management, or inhaler technique."
        )

        with gr.Row():
            query_box = gr.Textbox(
                label="Question",
                placeholder=(
                    "e.g. What is the first-line controller therapy for asthma?"
                ),
                max_length=2000,
                scale=4,
            )
            submit_btn = gr.Button("Ask", scale=1, variant="primary")

        gr.Examples(
            examples=[
                ["What is asthma?"],
                ["How is asthma diagnosed?"],
                ["What is the first-line controller therapy for asthma?"],
                ["How do I use my inhaler?"],
            ],
            inputs=query_box,
            label="Try an example question",
        )

        answer_md = gr.Markdown(label="Answer")
        video_html = gr.HTML(label="Demonstration video")

        gr.Markdown(_DISCLAIMER)

        submit_btn.click(
            fn=lambda q: respond(q, pipeline),
            inputs=query_box,
            outputs=[answer_md, video_html],
            api_name="query",
        )

    return app


def main() -> None:
    """Launch the Gradio app with the default pipeline."""
    create_app().launch()


if __name__ == "__main__":
    main()
