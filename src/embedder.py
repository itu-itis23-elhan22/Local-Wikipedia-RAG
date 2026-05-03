"""
src/embedder.py
===============
Yerel embedding katmanı.

Neden ChromaDB'nin built-in embed_function'ını DOĞRUDAN kullanmıyoruz?
---------------------------------------------------------------------
- Hangi modelle gömme yaptığımızı, batch boyutunu, cihaz seçimini açıkça
  görebilelim diye.
- Ödev "yerel embedding" zorunlu kıldığı için no-op davranan default'a
  düşmek istemiyoruz.
- Test yazılabilirlik: Embedder bir interface ardındadır; ileride
  Ollama nomic-embed-text'e geçmek istersek tek satır değişir.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """Sentence-Transformers tabanlı yerel embedder."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        *,
        device: str | None = None,        # None → otomatik (cuda > mps > cpu)
        normalize: bool = True,            # cosine ile uyumlu
        batch_size: int = 32,
    ) -> None:
        logger.info("Embedding modeli yükleniyor: %s", model_name)
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        # `device=None` → SentenceTransformer otomatik seçer.
        self.model = SentenceTransformer(model_name, device=device)
        self.dim: int = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding boyutu: %d", self.dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Liste -> embedding matrisi (List[List[float]])."""
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # ChromaDB JSON-friendly tipler bekler.
        return [v.astype(np.float32).tolist() for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        """Tek bir query'i embedler."""
        return self.embed_texts([text])[0]


# Singleton — uygulamada tek bir model yüklensin diye.
_INSTANCE: LocalEmbedder | None = None


def get_embedder() -> LocalEmbedder:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = LocalEmbedder()
    return _INSTANCE
