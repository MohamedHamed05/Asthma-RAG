"""FastAPI app exposing the asthma RAG pipeline for browser frontends."""

from __future__ import annotations

import os
import re
from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from asthma_rag.pipeline import Pipeline

_DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]


class AskRequest(BaseModel):
    """Frontend request payload."""

    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    """Backend response payload returned to the frontend."""

    answer: str
    citations: list[str]
    route: str | None = None
    video_html: str | None = None


def _parse_citations(answer: str) -> list[str]:
    """Extract citation bullets listed under a Sources section."""
    lines = answer.splitlines()
    source_header = re.compile(r"^\s*\*{0,2}\s*sources\s*\*{0,2}\s*$", re.IGNORECASE)
    bullet = re.compile(r"^\s*-\s+(.*\S)\s*$")

    in_sources = False
    citations: list[str] = []
    for line in lines:
        if source_header.match(line):
            in_sources = True
            continue
        if not in_sources:
            continue
        if not line.strip() and citations:
            break
        match = bullet.match(line)
        if match:
            citations.append(match.group(1))
    return citations


def _cors_allow_origins() -> list[str]:
    """Return CORS origins from env or local-dev defaults."""
    env_value = os.getenv("CORS_ALLOW_ORIGINS", "")
    if not env_value.strip():
        return _DEFAULT_CORS_ORIGINS
    return [origin.strip() for origin in env_value.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    """Create and cache the pipeline once for API requests."""
    return Pipeline()


app = FastAPI(title="Asthma RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Simple readiness endpoint for frontend/backed checks."""
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    """Run a real pipeline query and return answer metadata for the frontend."""
    result = get_pipeline().query(payload.question)
    answer = (result.get("final_answer") or "").strip()
    return AskResponse(
        answer=answer,
        citations=_parse_citations(answer),
        route=result.get("route"),
        video_html=result.get("video_html"),
    )
