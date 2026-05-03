"""
src/generator.py
================
Yerel Ollama üzerinden LLM üretim katmanı.

Sözleşme
--------
- Model parametreleri config.py'den okunur (OLLAMA_MODEL, sıcaklık, vb.).
- Bağlam (context) chunk listesi olarak verilir.
- System prompt katı: yalnızca context'e güven, bilmediğinde "I don't know".

Hata durumları
--------------
- Ollama daemon kapalıysa anlamlı bir mesajla yükselir; UI bunu yakalar.
- Model yüklenmemişse aynı şekilde anlaşılır error string döndürür.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator, Iterable, List, Optional

import ollama

from config import (
    GENERATION_NUM_PREDICT,
    GENERATION_TEMPERATURE,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Yardımcı tip
# ---------------------------------------------------------------------------
@dataclass
class GenerationResult:
    answer: str
    used_chunks: List[dict]
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


# ---------------------------------------------------------------------------
# Prompt birleştirici
# ---------------------------------------------------------------------------
def _format_context(chunks: Iterable[dict]) -> str:
    """Chunk listesini LLM'in okuyabileceği numaralı pasaj formatına çevirir."""
    lines: List[str] = []
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {}) or {}
        title = meta.get("title", "Unknown")
        ctype = meta.get("type", "unknown")
        text = (c.get("text") or "").strip()
        lines.append(f"[{i}] (type={ctype}, source={title})\n{text}")
    return "\n\n".join(lines) if lines else "(no context)"


def build_user_prompt(query: str, chunks: Iterable[dict]) -> str:
    """User mesajına gömülecek nihai prompt."""
    context = _format_context(chunks)
    return (
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{query}\n\n"
        "Answer using ONLY the CONTEXT above. "
        "If the answer is not present, reply exactly: I don't know."
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class OllamaGenerator:
    """Ollama HTTP API'si üzerinden Llama 3.2 (veya seçilen model) ile üretim."""

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        host: str = OLLAMA_HOST,
        *,
        temperature: float = GENERATION_TEMPERATURE,
        num_predict: int = GENERATION_NUM_PREDICT,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.host = host
        self.temperature = temperature
        self.num_predict = num_predict
        self.system_prompt = system_prompt
        # Ollama Python istemcisi `host` parametresi ile özel sunucuya bağlanır.
        self._client = ollama.Client(host=host)

    # ------------------------------------------------------------------
    # Tek seferlik generate
    # ------------------------------------------------------------------
    def generate(
        self,
        query: str,
        chunks: List[dict],
        *,
        extra_system: str | None = None,
    ) -> GenerationResult:
        user_prompt = build_user_prompt(query, chunks)
        system = self.system_prompt
        if extra_system:
            system = system + "\n\n" + extra_system

        try:
            resp = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.num_predict,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ollama chat hatası")
            raise RuntimeError(
                f"Ollama'ya bağlanılamadı ({self.host}). "
                "`ollama serve` çalışıyor mu? Hata: " + str(exc)
            ) from exc

        answer = (resp.get("message") or {}).get("content", "").strip()
        return GenerationResult(
            answer=answer or "I don't know.",
            used_chunks=chunks,
            model=self.model,
            prompt_tokens=resp.get("prompt_eval_count"),
            completion_tokens=resp.get("eval_count"),
        )

    # ------------------------------------------------------------------
    # Streaming generate (Streamlit için yararlı)
    # ------------------------------------------------------------------
    def stream(
        self,
        query: str,
        chunks: List[dict],
    ) -> Generator[str, None, None]:
        user_prompt = build_user_prompt(query, chunks)
        try:
            stream = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.num_predict,
                },
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001
            yield f"\n\n[error] Ollama'ya bağlanılamadı: {exc}"
            return

        for part in stream:
            token = (part.get("message") or {}).get("content", "")
            if token:
                yield token
