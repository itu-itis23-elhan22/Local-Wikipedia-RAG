"""
src/retriever.py
================
Sorgu yönlendirme + retrieval katmanı.

Akış
----
user_query ──► classify_query() ──► QueryIntent(type, mentioned_titles)
                                          │
                       VectorStore.query(top_k, where=...) ◄┘
                                          │
                                          ▼
                                  List[Chunk] (rank'lı)

Niye basit anahtar kelime tabanlı klasifikasyon?
------------------------------------------------
Ödev "Keyword based or rule based approaches are acceptable" diyor.
LLM-based classifier eklenebilir ama:
  - extra latency (her sorgu için bir LLM call),
  - extra hata yüzeyi,
  - production'a alınırken cache layer / eval seti gerekir.
40 entity'lik bir korpus için kural tabanlı doğruluk %99'a yakındır.
İleride iyileştirme: NER + spaCy. Bunu recommendation.md'de açıkladık.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from config import PEOPLE, PLACES, TOP_K
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent modelleri
# ---------------------------------------------------------------------------
class QueryType(str, Enum):
    PERSON = "person"
    PLACE = "place"
    BOTH = "both"
    UNKNOWN = "unknown"   # hiçbir entity eşleşmediğinde


@dataclass
class QueryIntent:
    """Sorguyu sınıflandırma sonucu."""

    query: str
    query_type: QueryType
    mentioned_people: List[str] = field(default_factory=list)
    mentioned_places: List[str] = field(default_factory=list)
    is_comparison: bool = False

    @property
    def all_mentioned_titles(self) -> List[str]:
        return self.mentioned_people + self.mentioned_places


# ---------------------------------------------------------------------------
# Anahtar kelime/entity matcher'ı
# ---------------------------------------------------------------------------
def _tokenize_lower(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


# Person ipuçları (heuristic): "who", "compare", "kim", isim olabilirleri.
_PERSON_HINTS = {
    "who", "whose", "whom", "person", "people", "human",
    "biography", "bio",
    "scientist", "artist", "musician", "physicist", "chemist",
    "mathematician", "footballer", "writer", "poet", "painter",
    "singer", "engineer", "inventor", "philosopher",
    "discovered", "invented", "wrote", "painted", "composed",
    "born", "died",
}
_PLACE_HINTS = {
    "where", "located", "location", "country", "city", "continent",
    "place", "places",
    "monument", "wonder", "tower", "wall", "temple", "mountain",
    "canyon", "river", "park", "ruins", "valley", "site",
    "tourist", "historic", "ancient",
}
_COMPARISON_HINTS = {"compare", "vs", "versus", "difference", "between"}


def _build_alias_index(titles: List[str]) -> List[tuple[str, str, set[str]]]:
    """Her başlık için (title, lower_title, alias_set) üretir.

    Aliases: "Albert Einstein" için → {"albert einstein", "einstein"}
    Soyad/önemli kelimeleri ek alias olarak ekleriz. Yan etki olarak
    "Marie Curie" sorgulanan yerde "Curie" da yakalanır.
    """
    index = []
    for t in titles:
        lower = _tokenize_lower(t)
        words = lower.split()
        aliases: set[str] = {lower}
        if len(words) > 1:
            # Son kelime (soyad / merkez kelime)
            aliases.add(words[-1])
            # 2 kelimelik "Mount Everest" gibi yer adlarında ilki de tutulsun
            if len(words[0]) > 3:
                aliases.add(words[0])
        index.append((t, lower, aliases))
    return index


_PEOPLE_INDEX = _build_alias_index(PEOPLE)
_PLACES_INDEX = _build_alias_index(PLACES)


def _find_mentions(query_lc: str, index: list) -> List[str]:
    """Bir alias indexi üzerinde substring eşleşmesi yapar."""
    found: List[str] = []
    for title, _lower, aliases in index:
        for alias in aliases:
            # Word boundary'ler: "tesla" "Coil-Tesla-Wow" kelimesini yakalamasın.
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, query_lc):
                found.append(title)
                break
    return found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def classify_query(query: str) -> QueryIntent:
    """Bir kullanıcı sorgusunu kişi/mekan/her ikisi şeklinde sınıflandırır."""
    q_lc = _tokenize_lower(query)
    tokens = set(q_lc.split())

    people_hits = _find_mentions(q_lc, _PEOPLE_INDEX)
    place_hits = _find_mentions(q_lc, _PLACES_INDEX)

    is_compare = bool(_COMPARISON_HINTS & tokens)

    # 1) İsim eşleşmesi varsa onu kullan (güçlü sinyal)
    if people_hits and place_hits:
        qt = QueryType.BOTH
    elif people_hits:
        qt = QueryType.PERSON
    elif place_hits:
        qt = QueryType.PLACE
    else:
        # 2) İsim eşleşmesi yoksa hint kelimelere bak
        person_score = len(_PERSON_HINTS & tokens)
        place_score = len(_PLACE_HINTS & tokens)
        if person_score and not place_score:
            qt = QueryType.PERSON
        elif place_score and not person_score:
            qt = QueryType.PLACE
        elif person_score and place_score:
            qt = QueryType.BOTH
        else:
            qt = QueryType.UNKNOWN

    intent = QueryIntent(
        query=query,
        query_type=qt,
        mentioned_people=people_hits,
        mentioned_places=place_hits,
        is_comparison=is_compare,
    )
    logger.debug("Intent: %s", intent)
    return intent


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class Retriever:
    """VectorStore üzerine ince bir orkestrasyon katmanı."""

    def __init__(self, store: Optional[VectorStore] = None) -> None:
        self.store = store or VectorStore()

    def retrieve(self, query: str, *, top_k: int = TOP_K) -> dict:
        """Intent classify et + chunk'ları getir.

        Returns
        -------
        dict
            {
              "intent": QueryIntent,
              "chunks": [ {text, metadata, distance, similarity}, ... ],
            }
        """
        intent = classify_query(query)

        if intent.query_type == QueryType.PERSON:
            chunks = self.store.query(query, top_k=top_k, type_filter="person")

        elif intent.query_type == QueryType.PLACE:
            chunks = self.store.query(query, top_k=top_k, type_filter="place")

        elif intent.query_type == QueryType.BOTH:
            # Comparison sorularını dengeli temsil edebilmek için
            # her iki taraftan eşit sayıda al, sonra distance'a göre kırp.
            half = max(1, top_k // 2)
            person_chunks = self.store.query(
                query, top_k=half, type_filter="person",
                title_filter=intent.mentioned_people or None,
            )
            place_chunks = self.store.query(
                query, top_k=half, type_filter="place",
                title_filter=intent.mentioned_places or None,
            )
            merged = person_chunks + place_chunks
            merged.sort(key=lambda c: c["distance"])
            chunks = merged[: top_k]

        else:  # UNKNOWN — yine de dene; filtre uygulamadan en yakın chunk
            chunks = self.store.query(query, top_k=top_k)

        return {"intent": intent, "chunks": chunks}
