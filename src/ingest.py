"""
src/ingest.py
=============
Wikipedia ingest katmanı.

Sorumluluklar
-------------
1. Önceden tanımlanmış kişi/mekan listelerini Wikipedia'dan çeker.
2. Her sayfanın özet, tam metin, URL ve "type" (person/place) bilgisini
   yerel bir SQLite veritabanında saklar (yeniden indirmeyi önlemek için).
3. İdempotent çalışır: ikinci çağrıda sadece eksik kayıtları getirir.

Tasarım notları
---------------
- `wikipedia` kütüphanesi yer yer DisambiguationError fırlatır; bu durumda
  aday başlık listesinin ilkini deneriz.
- Tam metin yerine sadece "summary" kullansaydık RAG retrieval'ın
  sürdürülebilirliği zayıf olurdu; tüm içerik daha sonra chunk'lanır.
- Veritabanı = SQLite çünkü tek-process'te zaten eşzamanlı yazma
  ihtiyacımız yok ve 40 küçük belge için fazlasıyla yeterli.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import wikipedia

from config import PEOPLE, PLACES, SQLITE_PATH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------
@dataclass
class WikiDocument:
    """Tek bir Wikipedia sayfasını temsil eder."""

    title: str           # bizim sorgulayıp etiketlediğimiz başlık
    canonical_title: str # Wikipedia'nın asıl döndürdüğü başlık
    type: str            # "person" veya "place"
    url: str
    summary: str
    content: str

    def to_row(self) -> tuple:
        return (
            self.title,
            self.canonical_title,
            self.type,
            self.url,
            self.summary,
            self.content,
        )


# ---------------------------------------------------------------------------
# SQLite katmanı
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    title             TEXT PRIMARY KEY,
    canonical_title   TEXT NOT NULL,
    type              TEXT NOT NULL CHECK (type IN ('person', 'place')),
    url               TEXT NOT NULL,
    summary           TEXT NOT NULL,
    content           TEXT NOT NULL,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(type);
"""


def _get_conn(db_path: Path = SQLITE_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = SQLITE_PATH) -> None:
    """Tabloları (yoksa) oluşturur."""
    with closing(_get_conn(db_path)) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()
    logger.info("SQLite şeması hazır: %s", db_path)


def get_existing_titles(db_path: Path = SQLITE_PATH) -> set[str]:
    with closing(_get_conn(db_path)) as conn:
        rows = conn.execute("SELECT title FROM documents").fetchall()
    return {r["title"] for r in rows}


def upsert_document(doc: WikiDocument, db_path: Path = SQLITE_PATH) -> None:
    sql = """
    INSERT INTO documents
        (title, canonical_title, type, url, summary, content)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(title) DO UPDATE SET
        canonical_title = excluded.canonical_title,
        type            = excluded.type,
        url             = excluded.url,
        summary         = excluded.summary,
        content         = excluded.content;
    """
    with closing(_get_conn(db_path)) as conn:
        conn.execute(sql, doc.to_row())
        conn.commit()


def load_all_documents(db_path: Path = SQLITE_PATH) -> List[WikiDocument]:
    with closing(_get_conn(db_path)) as conn:
        rows = conn.execute(
            "SELECT title, canonical_title, type, url, summary, content "
            "FROM documents"
        ).fetchall()
    return [
        WikiDocument(
            title=r["title"],
            canonical_title=r["canonical_title"],
            type=r["type"],
            url=r["url"],
            summary=r["summary"],
            content=r["content"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Wikipedia fetch
# ---------------------------------------------------------------------------
def _fetch_one(title: str, doc_type: str) -> Optional[WikiDocument]:
    """Tek bir Wikipedia sayfasını dirençli (resilient) şekilde çeker."""
    try:
        # auto_suggest=False  → "Marie Curie" gibi tam başlıkları başka
        # sayfalara yönlendirip kafanı karıştırmaz.
        page = wikipedia.page(title=title, auto_suggest=False, redirect=True)
    except wikipedia.DisambiguationError as e:
        # İlk adayı dene
        if not e.options:
            logger.warning("Disambiguation aday yok: %s", title)
            return None
        candidate = e.options[0]
        logger.warning("Disambiguation: %s → %s denenecek", title, candidate)
        try:
            page = wikipedia.page(title=candidate, auto_suggest=False)
        except Exception as exc:  # noqa: BLE001
            logger.error("Disambiguation sonrası başarısız (%s): %s", title, exc)
            return None
    except wikipedia.PageError:
        # auto_suggest açıp tekrar dene
        try:
            page = wikipedia.page(title=title, auto_suggest=True)
        except Exception as exc:  # noqa: BLE001
            logger.error("PageError, sonra da başarısız (%s): %s", title, exc)
            return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Wikipedia fetch hata (%s): %s", title, exc)
        return None

    return WikiDocument(
        title=title,
        canonical_title=page.title,
        type=doc_type,
        url=page.url,
        summary=page.summary,
        content=page.content,
    )


def ingest_entities(
    titles: Iterable[str],
    doc_type: str,
    db_path: Path = SQLITE_PATH,
    *,
    force: bool = False,
    polite_delay_sec: float = 0.4,
) -> List[WikiDocument]:
    """Verilen başlıkları çeker ve veritabanına yazar.

    Parameters
    ----------
    titles : entity başlıkları (örn. ["Albert Einstein", ...])
    doc_type : "person" | "place"
    force : True → cache'i yok say, baştan indir
    polite_delay_sec : Wikipedia'ya nazik olmak için her istek arası bekleme
    """
    init_db(db_path)
    existing = set() if force else get_existing_titles(db_path)
    fetched: List[WikiDocument] = []

    for title in titles:
        if title in existing:
            logger.info("Cache hit: %s", title)
            continue
        logger.info("Çekiliyor (%s): %s", doc_type, title)
        doc = _fetch_one(title, doc_type)
        if doc is None:
            logger.warning("Atlandı: %s", title)
            continue
        upsert_document(doc, db_path)
        fetched.append(doc)
        time.sleep(polite_delay_sec)

    logger.info("Yeni indirilen %s sayısı: %d", doc_type, len(fetched))
    return fetched


def ingest_all(
    *,
    people: Optional[List[str]] = None,
    places: Optional[List[str]] = None,
    force: bool = False,
) -> dict:
    """Hem people hem places ingest eder; özet istatistik döndürür."""
    people = people or PEOPLE
    places = places or PLACES

    new_people = ingest_entities(people, "person", force=force)
    new_places = ingest_entities(places, "place", force=force)

    total = len(load_all_documents())
    return {
        "new_people": len(new_people),
        "new_places": len(new_places),
        "total_in_db": total,
    }


# ---------------------------------------------------------------------------
# CLI giriş noktası (modül çalıştırılarak da kullanılabilir)
# ---------------------------------------------------------------------------
def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    stats = ingest_all()
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
