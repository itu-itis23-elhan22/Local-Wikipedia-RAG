"""
src/vector_store.py
===================
ChromaDB vector store wrapper'ı.

Tasarım kararı: **Option B — tek koleksiyon + metadata filtreleme**
-----------------------------------------------------------------
Ödev iki seçenek sundu:

A) Kişiler için ayrı, mekanlar için ayrı koleksiyon.
B) Tek koleksiyon; her chunk'ta `type ∈ {"person", "place"}` metadata.

**Bu projede B'yi seçtik. Neden?**

1. **Ölçeklenebilirlik:** İleride yeni domain'ler (örn. "events",
   "organizations") eklemek istersek, ek koleksiyon yaratıp client
   tarafında çatallı kod yazmak yerine sadece yeni bir `type` değeri
   eklemek yeterlidir.

2. **"Both" tipi sorgular:** "Compare Albert Einstein and the Eiffel
   Tower" gibi karma sorgularda iki ayrı koleksiyona ardışık hit atmak
   gerekirdi. Tek koleksiyonda `where` filtresi olmadan tek seferde
   en alakalı sonuçları alabiliyoruz.

3. **Operasyonel basitlik:** Tek bir HNSW indexi → daha az bakım, daha
   az disk fragmantasyonu, tek seferlik backup.

4. **Endüstri normu:** Pinecone, Weaviate, Milvus gibi production
   sistemlerinde tipik patern "namespace yerine metadata" yönündedir;
   filtreleme HNSW post-filter olarak çalışır ve modern motorlarda
   maliyeti düşüktür.

Trade-off: Filtreleme indeksi olmayan vector store'larda metadata
filtreleme tüm aday setini gezer. Chroma 0.4+ bu durumu pre-filter
olarak optimize eder, dolayısıyla 40 belgelik bir veri seti için
performans kaygısı yoktur.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import CHROMA_DIR, COLLECTION_NAME, DISTANCE_METRIC
from src.chunker import Chunk
from src.embedder import LocalEmbedder, get_embedder

logger = logging.getLogger(__name__)


class VectorStore:
    """Persisted ChromaDB collection — Wikipedia RAG için tek nokta."""

    def __init__(
        self,
        persist_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedder: Optional[LocalEmbedder] = None,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedder = embedder or get_embedder()

        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # NOT: ChromaDB'nin built-in "embedding_function"ı yerine
        # embed'leri elimizle gönderiyoruz; LocalEmbedder'ın tek doğruluk
        # kaynağı olmasını sağlar.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": DISTANCE_METRIC},
        )
        logger.info(
            "Vector store hazır: %s (count=%d)",
            collection_name,
            self.collection.count(),
        )

    # ------------------------------------------------------------------
    # Yazma
    # ------------------------------------------------------------------
    def add_chunks(self, chunks: Sequence[Chunk], *, batch_size: int = 64) -> int:
        """Chunk'ları gömüp koleksiyona yazar. Tekrar eden ID'leri günceller."""
        if not chunks:
            return 0

        n_added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id() for c in batch]
            docs = [c.text for c in batch]
            metas = [c.metadata() for c in batch]
            embeddings = self.embedder.embed_texts(docs)

            # `upsert` mevcut ID'leri günceller, yenileri ekler — idempotent.
            self.collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embeddings,
            )
            n_added += len(batch)
            logger.info("Vector store: %d / %d chunk yazıldı", n_added, len(chunks))
        return n_added

    # ------------------------------------------------------------------
    # Okuma
    # ------------------------------------------------------------------
    def query(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        type_filter: Optional[str] = None,   # "person" | "place" | None
        title_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Top-k benzer chunk'ları döndürür.

        Returns
        -------
        List[ {text, metadata, distance} ]
        """
        where: Dict[str, Any] = {}
        if type_filter and title_filter:
            where = {
                "$and": [
                    {"type": type_filter},
                    {"title": {"$in": title_filter}},
                ]
            }
        elif type_filter:
            where = {"type": type_filter}
        elif title_filter:
            where = {"title": {"$in": title_filter}}

        query_embedding = self.embedder.embed_query(query_text)

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where or None,
        )

        # ChromaDB cevabı paralel listeler döner; düzeltip dict listesine çevirelim.
        out: List[Dict[str, Any]] = []
        if not result["ids"] or not result["ids"][0]:
            return out

        for doc, meta, dist, _id in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            result["ids"][0],
        ):
            out.append(
                {
                    "id": _id,
                    "text": doc,
                    "metadata": meta,
                    "distance": float(dist),
                    # 1 - cosine_distance ≈ similarity
                    "similarity": 1.0 - float(dist),
                }
            )
        return out

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------
    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Koleksiyonu yere indirir; ingest sıfırdan yapılmalı."""
        logger.warning("Koleksiyon sıfırlanıyor: %s", self.collection_name)
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": DISTANCE_METRIC},
        )

    def list_titles(self) -> List[str]:
        """Tüm benzersiz belge başlıklarını döndürür (debug için)."""
        # Chroma'da `get` ile tüm metaları çekebiliriz.
        items = self.collection.get(include=["metadatas"])
        titles = {m["title"] for m in items["metadatas"] if m and "title" in m}
        return sorted(titles)
