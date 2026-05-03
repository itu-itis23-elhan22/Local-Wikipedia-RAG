"""
src/rag_pipeline.py
===================
Tüm RAG akışını orkestre eden facade.

  ingest → chunk → embed → store  (offline, run_ingest.py ile)
  ─────────────────────────────────
  query → classify → retrieve → generate  (online, app.py / CLI)

Bu modül UI'yi içsel detaylardan soyutlar; tek bir nokta üzerinden
"sor → cevap" çağrısı yapılır.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.chunker import chunk_documents
from src.generator import GenerationResult, OllamaGenerator
from src.ingest import ingest_all, load_all_documents
from src.retriever import QueryIntent, Retriever
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    query: str
    answer: str
    intent: QueryIntent
    chunks: List[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    model: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class RAGPipeline:
    """Üst düzey RAG arayüzü."""

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[OllamaGenerator] = None,
    ) -> None:
        self.store = store or VectorStore()
        self.retriever = retriever or Retriever(self.store)
        self.generator = generator or OllamaGenerator()

    # ------------------------------------------------------------------
    # Offline pipeline
    # ------------------------------------------------------------------
    def ingest_and_index(self, *, force: bool = False) -> dict:
        """Wikipedia'dan indir → SQLite'a yaz → chunk'la → ChromaDB'ye yaz."""
        t0 = time.time()
        ingest_stats = ingest_all(force=force)

        documents = load_all_documents()
        chunks = chunk_documents(documents)
        n_added = self.store.add_chunks(chunks)
        elapsed = time.time() - t0

        stats = {
            **ingest_stats,
            "chunks_added": n_added,
            "vector_count": self.store.count(),
            "elapsed_sec": round(elapsed, 2),
        }
        logger.info("Ingest+index tamamlandı: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # Online pipeline
    # ------------------------------------------------------------------
    def ask(self, query: str, *, top_k: Optional[int] = None) -> RAGAnswer:
        t0 = time.time()
        retrieval = self.retriever.retrieve(query, top_k=top_k or 5)
        intent: QueryIntent = retrieval["intent"]
        chunks: list = retrieval["chunks"]

        # Hiç chunk gelmediyse model'e boş context göndermek yerine kestir.
        if not chunks:
            return RAGAnswer(
                query=query,
                answer="I don't know.",
                intent=intent,
                chunks=[],
                latency_ms=(time.time() - t0) * 1000,
                model=self.generator.model,
            )

        gen: GenerationResult = self.generator.generate(query, chunks)
        return RAGAnswer(
            query=query,
            answer=gen.answer,
            intent=intent,
            chunks=chunks,
            latency_ms=(time.time() - t0) * 1000,
            model=gen.model,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
        )

    def stream(self, query: str, *, top_k: Optional[int] = None):
        """Streaming yanıt — Streamlit `st.write_stream` ile uyumlu."""
        retrieval = self.retriever.retrieve(query, top_k=top_k or 5)
        intent: QueryIntent = retrieval["intent"]
        chunks: list = retrieval["chunks"]

        if not chunks:
            def _empty():
                yield "I don't know."
            return _empty(), intent, []

        return self.generator.stream(query, chunks), intent, chunks

    # ------------------------------------------------------------------
    # Reset / utility
    # ------------------------------------------------------------------
    def reset_index(self) -> None:
        self.store.reset()

    def stats(self) -> dict:
        return {
            "vector_count": self.store.count(),
            "titles_indexed": self.store.list_titles(),
            "model": self.generator.model,
            "host": self.generator.host,
        }
