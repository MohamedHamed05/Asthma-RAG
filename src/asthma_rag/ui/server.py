"""FastAPI backend for the Aspira web UI.

Wraps the existing ``Pipeline`` (LangGraph agent) and ``GroqVoice`` (ASR/TTS)
in a small JSON API, and serves the static frontend in ``static/``. This is
an additional, independent front door — the original Gradio app in
``asthma_rag.ui.app`` is untouched and still works.

Run with:

    uv run uvicorn asthma_rag.ui.server:app --reload --port 8000

Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import re
import tempfile
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from asthma_rag.config import get_settings
from asthma_rag.pipeline import Pipeline
from asthma_rag.voice import VOICE_LABELS, GroqVoice

STATIC_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="Aspira Asthma Guidelines Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Pipeline | None = None
_voice: GroqVoice | None = None

_CONFIDENCE_FOOTER_RE = re.compile(
    r"\n\n---\n\*\*Confidence:\*\*\s*(?P<confidence>[^\n]+)\n\*\*Reason:\*\*\s*(?P<reason>.+)$",
    re.DOTALL,
)

_ROUTE_LABELS: dict[str, str] = {
    "retrieve": "Guideline retrieval",
    "inhaler": "Inhaler technique",
    "search": "Drug-price search",
    "video_search": "Video search",
    "out_of_scope": "Out of scope",
}


def _get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


def _get_voice() -> GroqVoice:
    global _voice
    if _voice is None:
        _voice = GroqVoice()
    return _voice


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str


class SpeakRequest(BaseModel):
    text: str
    voice: str = "hannah"


class Source(BaseModel):
    doc_name: str
    page: str | int
    chunk_id: str | int
    rerank_score: float | None = None


class WebSource(BaseModel):
    title: str
    url: str
    published_date: str | None = None


class ChatResponse(BaseModel):
    answer: str
    route: str
    route_label: str
    confidence: str | None = None
    judge_reason: str | None = None
    refused: bool = False
    video_html: str | None = None
    sources: list[Source] = []
    web_sources: list[WebSource] = []


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _split_confidence_footer(answer: str) -> tuple[str, str | None, str | None]:
    """Strip the ``\\n\\n---\\n**Confidence:** ...`` footer the judge appends.

    Returns ``(clean_answer, confidence, reason)``.
    """
    match = _CONFIDENCE_FOOTER_RE.search(answer)
    if not match:
        return answer, None, None
    clean = answer[: match.start()]
    return clean, match.group("confidence").strip(), match.group("reason").strip()


def _extract_sources(retrieved: dict[str, Any] | None) -> list[Source]:
    if not retrieved:
        return []
    metadatas = (retrieved.get("metadatas") or [[]])
    metadatas = metadatas[0] if metadatas else []
    sources: list[Source] = []
    seen: set[tuple[str, str]] = set()
    for meta in metadatas:
        doc_name = str(meta.get("doc_name", "Unknown"))
        page = meta.get("page_number", "Unknown")
        key = (doc_name, str(page))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            Source(
                doc_name=doc_name,
                page=page,
                chunk_id=meta.get("chunk_id", "Unknown"),
                rerank_score=meta.get("rerank_score"),
            )
        )
    return sources


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    missing = [
        name
        for name, val in (
            ("GROQ_API_KEY", settings.groq_api_key),
            ("COHERE_API_KEY", settings.cohere_api_key),
        )
        if not val
    ]
    return {
        "status": "ok" if not missing else "missing_keys",
        "missing_keys": missing,
        "exa_configured": bool(settings.exa_api_key),
    }


@app.get("/api/voices")
def voices() -> dict[str, Any]:
    return {"voices": [{"id": k, "label": v} for k, v in VOICE_LABELS.items()]}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    question = req.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="message must not be empty")

    try:
        result = _get_pipeline().query(question)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="The pipeline failed to answer that question.")

    raw_answer = result.get("final_answer") or ""
    clean_answer, footer_confidence, footer_reason = _split_confidence_footer(raw_answer)

    judge_result = result.get("judge_result") or {}
    confidence = judge_result.get("confidence") or footer_confidence
    judge_reason = judge_result.get("reason") or footer_reason

    route = result.get("route") or "retrieve"
    refused = bool(result.get("refusal_reason")) and route != "out_of_scope"

    web_hits = result.get("search_results") or []
    web_sources = [
        WebSource(
            title=hit.get("title", "Untitled"),
            url=hit.get("url", ""),
            published_date=hit.get("published_date"),
        )
        for hit in web_hits
    ]

    return ChatResponse(
        answer=clean_answer.strip(),
        route=route,
        route_label=_ROUTE_LABELS.get(route, route.replace("_", " ").title()),
        confidence=confidence,
        judge_reason=judge_reason,
        refused=refused or route == "out_of_scope",
        video_html=result.get("video_html") or None,
        sources=_extract_sources(result.get("retrieved")),
        web_sources=web_sources,
    )


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> dict[str, str]:
    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    data = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        text = _get_voice().transcribe(tmp_path)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"text": text}


@app.post("/api/speak")
def speak(req: SpeakRequest) -> FileResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    try:
        wav_path = _get_voice().speak(text, voice=req.voice)  # type: ignore[arg-type]
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {exc}")
    return FileResponse(wav_path, media_type="audio/wav", filename="answer.wav")


# --------------------------------------------------------------------------
# Static frontend
# --------------------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.exception_handler(404)
    async def spa_fallback(request, exc):  # noqa: ANN001, ARG001
        if request.url.path.startswith("/api"):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        return FileResponse(STATIC_DIR / "index.html")
