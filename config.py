"""
config.py
=========
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CHROMA_DIR: Path = DATA_DIR / "chroma"
SQLITE_PATH: Path = DATA_DIR / "wiki_cache.sqlite"
RAW_PAGES_DIR: Path = DATA_DIR / "raw_pages"

# Ensure folders are ready on first import
for _p in (DATA_DIR, CHROMA_DIR, RAW_PAGES_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# homework dataset 
# ---------------------------------------------------------------------------
PEOPLE: List[str] = [
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    # 10 extra
    "Isaac Newton",
    "Charles Darwin",
    "Stephen Hawking",
    "Mahatma Gandhi",
    "Nelson Mandela",
    "Vincent van Gogh",
    "Pablo Picasso",
    "Wolfgang Amadeus Mozart",
    "Mustafa Kemal Atatürk",
    "Steve Jobs",
]

PLACES: List[str] = [
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Giza pyramid complex",   # Wikipedia  subtitle for "Pyramids of Giza"  
    "Mount Everest",
    # 10 extra
    "Stonehenge",
    "Petra",
    "Acropolis of Athens",
    "Niagara Falls",
    "Mount Fuji",
    "Sagrada Família",
    "Burj Khalifa",
    "Cappadocia",
    "Angkor Wat",
    "Sydney Opera House",
]


# ---------------------------------------------------------------------------
# Chunking strategy
# ---------------------------------------------------------------------------
# Character-based; average ~1 token = 4 characters (for English)
# 800 characters ≈ 200 tokens: mostly covers a single paragraph,
# Also easily handled by Llama 3.2 3B context (8K).
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 150  # ~18% overlap; compensates for sentence breaks.


# ---------------------------------------------------------------------------
# Embedding model (local)
# ---------------------------------------------------------------------------
# all-MiniLM-L6-v2: 384-dimensional, very fast (~14k sentences/sec CPU), MTEB score
# sufficient for practical uses. First download ~80 MB.
EMBEDDING_MODEL_NAME: str = os.getenv(
    "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM: int = 384


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
COLLECTION_NAME: str = "wiki_rag"
# ChromaDB uses cosine distance. Reason: by normalizing embeddings
# produces consistent results in terms of semantic similarity. Dot product (ip)
# is sensitive to vector magnitudes; L2 loses information in high dimensions.
DISTANCE_METRIC: str = "cosine"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K: int = 5  # her sorguda dönen chunk sayısı


# ---------------------------------------------------------------------------
# Generation (Ollama)
# ---------------------------------------------------------------------------
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("RAG_LLM_MODEL", "llama3.2:3b")
GENERATION_TEMPERATURE: float = 0.2  # low = less hallucination
GENERATION_NUM_PREDICT: int = 512    # max new tokens


# ---------------------------------------------------------------------------
# System prompt — force the model to context
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = """You are a careful, factual assistant that answers questions \
about famous people and famous places using ONLY the context provided below.

Strict rules:
1. Use ONLY information present in the CONTEXT. Do NOT use outside knowledge.
2. If the answer cannot be found in the CONTEXT, reply exactly: "I don't know."
3. Never invent facts, dates, or numbers. If unsure, say "I don't know."
4. Be concise (3-6 sentences). When useful, cite the entity name like (source: Albert Einstein).
5. If the question asks to compare two entities and only one is in the CONTEXT, \
say what you can about the available one and reply "I don't know" for the missing one.
6. If the user asks about something clearly outside the dataset \
(e.g. "the president of Mars"), reply "I don't know."
7. You MAY combine facts that are explicitly stated in the CONTEXT to \
answer indirect questions. For example, if the CONTEXT says \
"X is located in country Y" and the user asks "which place is in Y?", \
you SHOULD answer X. This is NOT outside knowledge — it is a direct \
inference from the CONTEXT itself.
"""


# ---------------------------------------------------------------------------
# Dataclass — so that other modules can receive a single type "Settings" object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """Read-only runtime config snapshot."""

    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    embedding_model: str = EMBEDDING_MODEL_NAME
    collection_name: str = COLLECTION_NAME
    chroma_dir: Path = CHROMA_DIR
    sqlite_path: Path = SQLITE_PATH
    top_k: int = TOP_K
    ollama_host: str = OLLAMA_HOST
    ollama_model: str = OLLAMA_MODEL
    temperature: float = GENERATION_TEMPERATURE
    num_predict: int = GENERATION_NUM_PREDICT
    system_prompt: str = SYSTEM_PROMPT
    people: List[str] = field(default_factory=lambda: list(PEOPLE))
    places: List[str] = field(default_factory=lambda: list(PLACES))


def get_settings() -> Settings:
    """Returns a single Settings instance (convenient for DI / testing)."""
    return Settings()
