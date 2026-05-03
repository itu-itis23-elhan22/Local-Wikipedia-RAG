"""
scripts/run_ingest.py
=====================
CLI girişi: Wikipedia ingest + chunk + embed + store.

Kullanım
--------
    python -m scripts.run_ingest                # eksikleri çek
    python -m scripts.run_ingest --force        # yeniden çek (cache'i yok say)
    python -m scripts.run_ingest --reset        # ChromaDB'yi sıfırla, sonra tekrar yaz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Project root'u path'e ekle (PYTHONPATH ayarına bağımlılığı kaldırır)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_pipeline import RAGPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wikipedia RAG ingest")
    p.add_argument(
        "--force",
        action="store_true",
        help="Cache'i yok say, sayfaları Wikipedia'dan yeniden çek.",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="ChromaDB koleksiyonunu sıfırla, sonra ingest+index uygula.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="DEBUG seviyesinde log üret.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    pipeline = RAGPipeline()

    if args.reset:
        pipeline.reset_index()

    stats = pipeline.ingest_and_index(force=args.force)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
