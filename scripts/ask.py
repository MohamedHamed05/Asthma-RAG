"""Ask the asthma RAG agent a question from the command line.

Example::

    uv run python scripts/ask.py "What is asthma?"
"""

from __future__ import annotations

import argparse
import sys

from asthma_rag.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask the asthma RAG agent a question.",
    )
    parser.add_argument("question", help="The asthma question to ask.")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    pipeline = Pipeline()
    result = pipeline.query(args.question)

    print(result.get("final_answer", ""))

    route = result.get("route")
    if route:
        print(f"\n[route: {route}]")
    if result.get("video_html"):
        print("[video embed returned]")


if __name__ == "__main__":
    main()
