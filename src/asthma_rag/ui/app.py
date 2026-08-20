"""Gradio UI for the asthma RAG agent with voice support."""

from __future__ import annotations

import traceback

import gradio as gr

from asthma_rag.pipeline import Pipeline
from asthma_rag.voice import GroqVoice, VOICE_LABELS

_DISCLAIMER = (
    "**Disclaimer:** This tool is an educational prototype for exploring "
    "retrieval-augmented generation over public asthma guideline documents. "
    "It is **not** a medical device and does **not** provide medical advice. "
    "Always consult a qualified healthcare professional for diagnosis or "
    "treatment decisions."
)

_WELCOME = (
    "👋 Welcome! I'm your Asthma Guidelines assistant. "
    "Ask me anything about asthma diagnosis, management, or inhaler technique. "
    "You can **type** your question or use the **microphone** to speak."
)


def _msg(role: str, content: str) -> dict:
    """Build a Gradio 6.0 chat message dict."""
    return {"role": role, "content": content}


def create_app(pipeline: Pipeline | None = None) -> gr.Blocks:
    """Build and return the enhanced Gradio Blocks app with voice I/O."""
    pipeline = pipeline or Pipeline()
    voice = GroqVoice()

    def respond(
        text_input: str,
        audio_input: str | None,
        history: list,
        voice_choice: str,
        speak_answer: bool,
    ) -> tuple[list, str | None, str, dict, dict]:
        """Handle a user turn (text or voice) and update the chat UI."""
        # ------------------------------------------------------------------
        # Resolve query
        # ------------------------------------------------------------------
        if text_input and text_input.strip():
            query = text_input.strip()
            display_query = query
        elif audio_input:
            try:
                query = voice.transcribe(audio_input)
            except Exception as exc:
                tb = traceback.format_exc()
                print("[ASR ERROR]\n", tb)
                history = history + [_msg("assistant", f"⚠️ Transcription failed: {exc}")]
                return history, None, "", gr.update(value=""), gr.update(value=None)
            display_query = f"🎤 {query}"
        else:
            return history, None, "", gr.update(value=""), gr.update(value=None)

        # ------------------------------------------------------------------
        # Run RAG pipeline
        # ------------------------------------------------------------------
        try:
            result = pipeline.query(query)
        except Exception:
            tb = traceback.format_exc()
            print("[RAG ERROR]\n", tb)
            short_err = tb.split("\n")[-2] if "\n" in tb else "Unknown error"
            history = history + [
                _msg("user", display_query),
                _msg(
                    "assistant",
                    f"❌ **Pipeline error:** `{short_err}`\n\n"
                    f"Check the terminal for the full traceback.",
                ),
            ]
            return history, None, "", gr.update(value=""), gr.update(value=None)

        answer: str = result.get("final_answer", "")
        video_html: str = result.get("video_html", "") or ""

        # ------------------------------------------------------------------
        # Optional TTS
        # ------------------------------------------------------------------
        audio_out: str | None = None
        tts_error_msg: str = ""
        if speak_answer and answer:
            try:
                audio_out = voice.speak(answer, voice=voice_choice)
            except Exception as exc:
                tb = traceback.format_exc()
                print("[TTS ERROR]\n", tb)
                tts_error_msg = f"\n\n_⚠️ TTS failed: {exc}_"

        if tts_error_msg:
            answer = answer + tts_error_msg

        history = history + [
            _msg("user", display_query),
            _msg("assistant", answer),
        ]
        return history, audio_out, video_html, gr.update(value=""), gr.update(value=None)

    # ==================================================================
    # UI layout  —  clean two-column design
    # ==================================================================
    with gr.Blocks(title="Asthma Guidelines RAG") as app:
        gr.Markdown("# 🫁 Asthma Guidelines RAG")
        gr.Markdown(
            "Ask a question about asthma via **text** or **voice**. "
            "Enable **🔊 Speak answer** to hear the response."
        )

        with gr.Row():
            # ==================== LEFT: Chat + Controls ====================
            with gr.Column(scale=3, min_width=640):
                chatbot = gr.Chatbot(
                    value=[_msg("assistant", _WELCOME)],
                    height=540,
                    show_label=False,
                    container=False,
                )

                with gr.Row():
                    msg_text = gr.Textbox(
                        placeholder="Type your question here…",
                        show_label=False,
                        container=False,
                        scale=4,
                        max_length=2000,
                    )
                    msg_audio = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        show_label=False,
                        container=False,
                        scale=1,
                        min_width=80,
                    )

                with gr.Row():
                    voice_select = gr.Dropdown(
                        choices=[(label, val) for val, label in VOICE_LABELS.items()],
                        value="hannah",
                        label="Voice",
                        scale=1,
                        min_width=120,
                    )
                    speak_check = gr.Checkbox(
                        label="🔊 Speak answer",
                        value=False,
                        scale=1,
                        min_width=120,
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=1, min_width=80)
                    clear_btn = gr.Button("Clear", scale=1, min_width=80)

                gr.Examples(
                    examples=[
                        ["What is asthma?"],
                        ["How is asthma diagnosed?"],
                        ["What is the first-line controller therapy for asthma?"],
                        ["How do I use my inhaler?"],
                    ],
                    inputs=msg_text,
                    label="Try an example",
                )

            # ==================== RIGHT: Media ====================
            with gr.Column(scale=1, min_width=320):
                video_panel = gr.HTML(
                    label="Demonstration video",
                    show_label=False,
                )
                tts_player = gr.Audio(
                    label="🔊 Spoken answer",
                    autoplay=False,
                    interactive=False,
                    show_label=False,
                )

                with gr.Accordion("About", open=False):
                    gr.Markdown(_DISCLAIMER)

        # ==================================================================
        # Event wiring
        # ==================================================================
        all_inputs = [msg_text, msg_audio, chatbot, voice_select, speak_check]
        all_outputs = [chatbot, tts_player, video_panel, msg_text, msg_audio]

        submit_btn.click(fn=respond, inputs=all_inputs, outputs=all_outputs)
        msg_text.submit(fn=respond, inputs=all_inputs, outputs=all_outputs)

        clear_btn.click(
            fn=lambda: (
                [_msg("assistant", _WELCOME)],
                None,
                "",
                gr.update(value=""),
                gr.update(value=None),
            ),
            outputs=all_outputs,
        )

    return app


def main() -> None:
    """Launch the Gradio app with the default pipeline."""
    create_app().launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()