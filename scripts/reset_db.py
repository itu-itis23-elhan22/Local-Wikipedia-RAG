"""
scripts/reset_db.py
===================
SQLite + ChromaDB tüm yerel state'i siler.

Bunu sadece "temiz başlamak" istediğinde çalıştır.

    python -m scripts.reset_db
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CHROMA_DIR, SQLITE_PATH  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger("reset_db")

    if SQLITE_PATH.exists():
        SQLITE_PATH.unlink()
        log.info("SQLite silindi: %s", SQLITE_PATH)
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        log.info("ChromaDB klasörü silindi: %s", CHROMA_DIR)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Sıfırlama tamamlandı.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
