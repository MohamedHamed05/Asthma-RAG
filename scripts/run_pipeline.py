"""One-off script: ingest PDFs and rebuild the Chroma index.

Example::

    uv run python scripts/run_pipeline.py --pdf-dir data/raw --chroma-path chroma_db
"""

from __future__ import annotations

import argparse
from pathlib import Path

from asthma_rag.config import Settings
from asthma_rag.embeddings import get_embedding_function
from asthma_rag.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest asthma guideline PDFs and rebuild the Chroma index.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Directory containing source PDFs (default: data/raw).",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=None,
        help="Path to write chunks.json (default: data/chunks/chunks.json).",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path("chroma_db"),
        help="Path to the Chroma persistent database (default: chroma_db).",
    )
    parser.add_argument(
        "--use-hash-embeddings",
        action="store_true",
        help="Use deterministic hash embeddings instead of the configured Qwen3 model.",
    )
    args = parser.parse_args()

    settings = Settings(chroma_path=args.chroma_path)
    embedding_function = None if args.use_hash_embeddings else get_embedding_function()
    pipeline = Pipeline(settings=settings, embedding_function=embedding_function)

    chunks = pipeline.run(pdf_dir=args.pdf_dir, chunks_path=args.chunks_path)
    print(f"Indexed {len(chunks)} chunks in {settings.chroma_path}")


if __name__ == "__main__":
    main()
