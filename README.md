# Asthma RAG Pipeline

Retrieval-augmented generation over authoritative asthma guidelines:

**PDFs → clean text → chunks → local Qwen3 embeddings → Chroma → retrieve → Cohere rerank → Groq answer via a LangGraph agent, exposed through Gradio.**

## Features

- Modular agentic RAG pipeline with separate cleaning, embedding, retrieval, reranking, LLM, and UI packages.
- Embeddings via `Qwen3-Embedding-0.6B`: GPU Ollama server (set in this project's `.env`) or local CPU (sentence-transformers).
- Cloud LLM via Groq (`openai/gpt-oss-120b`).
- Cohere `rerank-v4.0-pro` reranking.
- LangGraph agent with document grading, one-shot query rewrite, and a special inhaler-technique route that returns an embedded American Lung Association demonstration video.
- Gradio web UI with clinical-safety disclaimer.
- Full test suite (147 tests) covering every module and end-to-end scenario contracts.

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Configure secrets
cp .env.example .env
# Edit .env and add GROQ_API_KEY and COHERE_API_KEY
```

## Usage

### 1. Ingest PDFs and build the vector index

Place source PDFs in `data/raw/` (never committed), then run:

```bash
uv run python scripts/run_pipeline.py --pdf-dir data/raw --chroma-path chroma_db
```

Use `--use-hash-embeddings` for a fast, deterministic dev index that does not download Qwen3:

```bash
uv run python scripts/run_pipeline.py --use-hash-embeddings
```

Embeddings run on a GPU Ollama server when `EMBEDDING_BACKEND=ollama` is set in `.env` (requires `qwen3-embedding:0.6b` on the server). The default `local` backend runs Qwen3 on CPU.

### 2. Ask a question from the CLI

```bash
uv run python scripts/ask.py "What is asthma?"
```

### 3. Launch the Gradio UI

```bash
uv run python -m asthma_rag.ui.app
```

Then open the printed local URL and ask a question.

### 4. Run tests

```bash
uv run pytest
```

## Project structure

```
src/asthma_rag/
  cleaning/        # PDF text cleaning, section extraction, validation
  llm/             # Groq chat client
  rerank/          # Cohere rerank wrapper
  agent/           # LangGraph state, nodes, and graph wiring
  ui/              # Gradio app + YouTube embed helper
  config.py        # Environment-driven settings
  embeddings.py    # Local embedding factory
  ingest.py        # PDF → chunks CLI
  pipeline.py      # High-level ingest → index → agent orchestrator
  prompts.py       # Locked clinical system prompt
  retrieval.py     # Retrieve, rerank, and context formatting
  sources.py       # sources.yaml loader
  vectorstore.py   # Chroma wrapper

tests/             # Test suite
scripts/           # One-off pipeline scripts
sources.yaml       # Authoritative source registry
data/raw/          # Source PDFs (gitignored)
chroma_db/         # Vector index (gitignored)
```

## Sources

Source documents are tracked in `sources.yaml`. Only open-access (`fetch: true`) sources should be downloaded and indexed; rights-reserved entries are cite-only.

## Safety disclaimer

This tool is an educational prototype for exploring RAG over public asthma guideline documents. It is **not** a medical device and does **not** provide medical advice. Always consult a qualified healthcare professional for diagnosis or treatment decisions.