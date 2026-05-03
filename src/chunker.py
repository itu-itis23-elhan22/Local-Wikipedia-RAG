"""
src/chunker.py
==============
Metin parçalama (chunking) katmanı.

Stratejimiz: **Karakter-bazlı, cümle-aware, sabit boyutlu, örtüşmeli (sliding-window).**

Neden bu strateji?
------------------
- *Sabit boyut*  → embedding modelinin context limitine garanti uyar.
- *Örtüşme*      → bir bilgi parçası iki chunk arasında bölünürse, en az
  birinde tam görünür ve retrieval kalitesi düşmez.
- *Cümle-aware* → chunk sınırlarını cümle sonlarına yaklaştırarak yarım
  kalmış cümleleri minimize eder; recall için önemli.
- *Karakter*    → tokenizer'a bağımlı değil; embedder değişse bile çalışır.

Ödevde "language native functionality" istendiği için harici
RecursiveCharacterTextSplitter (LangChain) yerine kendi sade
implementasyonumuzu yazıyoruz.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

from config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """Tek bir metin parçası + metadata."""

    text: str
    source_title: str       # ait olduğu Wikipedia başlığı (canonical)
    source_type: str        # "person" | "place"
    source_url: str
    chunk_index: int        # belge içindeki sıralama
    char_start: int
    char_end: int

    def chunk_id(self) -> str:
        """ChromaDB için stabil, tekrarlanabilir kimlik."""
        # ":" ChromaDB tarafından kabul edilir, ID stable.
        safe_title = re.sub(r"\s+", "_", self.source_title)
        return f"{safe_title}::{self.chunk_index}"

    def metadata(self) -> dict:
        """ChromaDB'ye yazılacak metadata sözlüğü.

        Sadece JSON-uyumlu, primitif tipleri kullanırız (Chroma kuralı).
        """
        return {
            "title": self.source_title,
            "type": self.source_type,        # ← Option B'nin filtre anahtarı
            "url": self.source_url,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")
# Wikipedia "== Bölüm Başlığı ==" şeklinde başlıklar koyar; metni okuyan
# için faydalıdır ama embedder'a fazla "===" gürültüsü iyi gelmez.
_WIKI_HEADING_RE = re.compile(r"={2,}\s*([^=]+?)\s*={2,}")


def clean_text(raw: str) -> str:
    """Hafif normalizasyon: fazla boşluk, başlık delimleri."""
    text = _WIKI_HEADING_RE.sub(r"\n\1\n", raw)        # "== X ==" → "X"
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTIPLE_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def _find_sentence_boundary(text: str, target: int, look_back: int = 200) -> int:
    """`target` etrafında en yakın cümle sonunu bul; bulamazsa target dön.

    Tipik Wikipedia metinlerinde "." / "!" / "?" cümle sonu işareti olur.
    """
    if target >= len(text):
        return len(text)

    window_start = max(0, target - look_back)
    window = text[window_start:target]
    # Pencerede son nokta/?/! konumu (tabandan)
    for i in range(len(window) - 1, -1, -1):
        if window[i] in ".!?":
            # Cümle sonundan SONRAKİ karakter chunk başlangıcı olur.
            return window_start + i + 1
    return target  # cümle sonu yoksa olduğu yerde böl


# ---------------------------------------------------------------------------
# Ana API
# ---------------------------------------------------------------------------
def chunk_document(
    text: str,
    *,
    source_title: str,
    source_type: str,
    source_url: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """Bir Wikipedia belgesini örtüşmeli chunk'lara böler.

    Algoritma
    ---------
    pos = 0
    repeat:
        end = pos + chunk_size
        end = en yakın cümle sonuna kaydır (geri doğru)
        chunk = text[pos:end]
        pos = end - chunk_overlap   ← örtüşme penceresi
    until pos >= len(text)
    """
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size, chunk_overlap'tan büyük olmalı")

    text = clean_text(text)
    n = len(text)
    if n == 0:
        return []

    chunks: List[Chunk] = []
    pos = 0
    idx = 0
    step = chunk_size - chunk_overlap

    while pos < n:
        target_end = min(pos + chunk_size, n)
        if target_end < n:
            end = _find_sentence_boundary(text, target_end)
            # Eğer aynı yere düştüysek (boyutsuz progress) zorla ilerlet
            if end <= pos:
                end = target_end
        else:
            end = n

        piece = text[pos:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    text=piece,
                    source_title=source_title,
                    source_type=source_type,
                    source_url=source_url,
                    chunk_index=idx,
                    char_start=pos,
                    char_end=end,
                )
            )
            idx += 1

        if end >= n:
            break
        pos = max(end - chunk_overlap, pos + 1)  # progress garantisi

    logger.debug("Chunked '%s' → %d parça", source_title, len(chunks))
    return chunks


def chunk_documents(documents: list, **kwargs) -> List[Chunk]:
    """Birden fazla WikiDocument'i flat chunk listesine indirger."""
    out: List[Chunk] = []
    for d in documents:
        out.extend(
            chunk_document(
                d.content,
                source_title=d.canonical_title,
                source_type=d.type,
                source_url=d.url,
                **kwargs,
            )
        )
    return out
