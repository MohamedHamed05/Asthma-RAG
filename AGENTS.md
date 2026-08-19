# AGENTS.md — Asthma RAG Pipeline

## Goal
Refactor a prototype Jupyter notebook (`hackathon (2).py`) into a modular, agentic RAG pipeline that answers questions from asthma guidelines (PDFs). All implementation waves are complete; the pipeline is verified end-to-end against real data.

## Stack
- **LLM**: Groq (`openai/gpt-oss-120b`, key via env; note: `llama-3.3-70b-versatile` is retired)
- **Reranking**: Cohere (`rerank-v4.0-pro`, `COHERE_API_KEY`)
- **Embeddings**: Qwen3-Embedding-0.6B via a GPU Ollama server (`EMBEDDING_BACKEND=ollama`, set in this project's `.env`) or local sentence-transformers on CPU (`EMBEDDING_BACKEND=local`, the code default; `HF_HOME` cache)
- **Vector store**: Chroma (`chroma_db/`)
- **UI**: Gradio
- **Agent orchestration**: LangGraph

## Key folders
| Path | Purpose |
|------|---------|
| `src/asthma_rag/cleaning/` | PDF → clean text extraction, section titles, validation |
| `src/asthma_rag/llm/` | Groq chat client |
| `src/asthma_rag/rerank/` | Cohere reranking |
| `src/asthma_rag/agent/` | LangGraph state, nodes (grader/rewriter/inhaler), graph wiring |
| `src/asthma_rag/ui/` | Gradio interface + YouTube embed helper |
| `src/asthma_rag/pipeline.py` | High-level ingest → index → agent orchestrator |
| `src/asthma_rag/ollama_embeddings.py` | Ollama-backed Chroma embedding function |
| `src/asthma_rag/embeddings.py` | Embedding factory (local vs ollama dispatch) |
| `src/asthma_rag/vectorstore.py` | Chroma wrapper |
| `src/asthma_rag/retrieval.py` | Retrieve, rerank, context formatting |
| `src/asthma_rag/ingest.py` | PDF → chunks |
| `data/raw/` | Source PDFs (gitignored) |
| `data/chunks/` | Chunked documents (gitignored; derived from copyrighted PDFs) |
| `chroma_db/` | Vector index (gitignored) |
| `tests/` | Test suite (147 tests) |
| `scripts/` | Pipeline scripts (`run_pipeline.py`, `ask.py`) |

## Environment variables
- `GROQ_API_KEY` — Groq LLM
- `COHERE_API_KEY` — Cohere rerank
- `HF_HOME` — optional, local model cache path (local backend only)
- `EMBEDDING_BACKEND` — `ollama` (this project's `.env`) or `local` (code default)
- `OLLAMA_BASE_URL` — Ollama server URL (default `http://localhost:11434`)
- `OLLAMA_EMBEDDING_MODEL` — default `qwen3-embedding:0.6b`
- `GROQ_MODEL` — optional, overrides the chat model

## Rules
- **Never commit PDFs** or raw source documents. `data/raw/`, `*.pdf`, and `hackathon.zip` (contains the raw PDFs) are gitignored.
- Never commit `.env` (secrets).
- Generated artifacts (`chroma_db/`, `data/chunks/`, `data/cleaned/`, `.omo/`, `.codegraph/`) are gitignored.
- Keep the pipeline modular: each package must not import from sibling packages' internals.
- The clinical system prompt (`src/asthma_rag/prompts.py`) is locked by SHA-256 in `tests/test_prompts.py`; any edit must be deliberate and update the hash.

## Workflow
1. `uv run python scripts/run_pipeline.py` to rebuild the index from `data/raw/`.
2. `uv run python scripts/ask.py "question"` for CLI queries.
3. `uv run python -m asthma_rag.ui.app` to launch the Gradio UI.
4. `uv run pytest` for tests.
5. Add sources to `sources.yaml`; respect `fetch: false` (cite-only, do not download).
