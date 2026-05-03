"""
main.py
=======
Light CLI chat — allows you to test the assistant even if Streamlit is not installed.

Usage
--------
    python main.py                  # interactive chat loop
    python main.py "What did Marie Curie discover?"

Commands (in interactive mode)
---------------------------
    /sources    show source chunks of last answer
    /stats      index statistics
    /reset      clear ChromaDB (CAUTION)
    /quit       exit
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from src.rag_pipeline import RAGAnswer, RAGPipeline


def _print_answer(ans: RAGAnswer) -> None:
    print()
    print("=" * 72)
    print(f"Q: {ans.query}")
    print(
        f"  intent: {ans.intent.query_type.value} "
        f"(people={ans.intent.mentioned_people}, "
        f"places={ans.intent.mentioned_places}, "
        f"compare={ans.intent.is_comparison})"
    )
    print(f"  latency: {ans.latency_ms:.0f} ms | model: {ans.model}")
    print("-" * 72)
    print(ans.answer)
    print("=" * 72)


def _print_sources(ans: Optional[RAGAnswer]) -> None:
    if not ans or not ans.chunks:
        print("(no sources)")
        return
    for i, c in enumerate(ans.chunks, start=1):
        m = c["metadata"]
        sim = c.get("similarity", 0.0)
        print(f"[{i}] {m['title']} ({m['type']})  sim={sim:.3f}")
        snippet = c["text"][:240].replace("\n", " ")
        print(f"    {snippet}...")
        print(f"    {m.get('url', '')}")
        print()


def _interactive_loop(pipeline: RAGPipeline) -> None:
    last: Optional[RAGAnswer] = None
    print("Local Wikipedia RAG — CLI. Exit with /quit.\n")
    while True:
        try:
            q = input("you ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not q:
            continue
        if q in ("/quit", "/exit", "exit", "quit"):
            return
        if q == "/sources":
            _print_sources(last)
            continue
        if q == "/stats":
            print(pipeline.stats())
            continue
        if q == "/reset":
            confirm = input("Are you sure you want to reset ChromaDB? (y/N) ")
            if confirm.strip().lower() == "y":
                pipeline.reset_index()
                print("Reset complete. Re-index with `python -m scripts.run_ingest`.")
            continue

        try:
            last = pipeline.ask(q)
        except RuntimeError as exc:
            print(f"[error] {exc}")
            continue
        _print_answer(last)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pipeline = RAGPipeline()

    if pipeline.store.count() == 0:
        print(
            "WARNING: Vector store is empty. Run this command first:\n"
            "    python -m scripts.run_ingest\n"
        )

    if len(sys.argv) > 1:
        # Single-shot mode
        query = " ".join(sys.argv[1:])
        try:
            ans = pipeline.ask(query)
        except RuntimeError as exc:
            print(f"[error] {exc}")
            return 1
        _print_answer(ans)
        return 0

    _interactive_loop(pipeline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
