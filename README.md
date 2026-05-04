# Local Wikipedia RAG Assistant

DEMO VİDEO= https://youtu.be/Vrh-Fl35Jx4


A **fully-local** Retrieval-Augmented Generation (RAG) system that answers
questions about famous people and famous places, using Wikipedia as the
knowledge base. No external LLM API is used at any point — every component
(embeddings, vector search, language model) runs on your machine.

This is the deliverable for **BLG483E – Project 3: Build a Local Wikipedia RAG
Assistant**.

---

## 1. Architecture at a glance

```
            ┌──────────────────────────────────────────────────────┐
            │                       Streamlit UI                   │
            │                  (app.py — chat interface)           │
            └────────────────────────┬─────────────────────────────┘
                                     │
                                     ▼
            ┌──────────────────────────────────────────────────────┐
            │                     RAGPipeline                      │
            │  (orchestrates retrieve → generate, also ingest)     │
            └─────┬────────────────────────────────────────┬───────┘
                  │ online                                  │ offline
                  ▼                                         ▼
   ┌──────────────────────────┐           ┌──────────────────────────────┐
   │ Retriever                │           │ Ingest                       │
   │  • classify_query()      │           │  • wikipedia API             │
   │  • metadata-filtered     │           │  • SQLite cache              │
   │    Chroma query          │           │                              │
   └───────────┬──────────────┘           └─────────────┬────────────────┘
               │                                         │
               ▼                                         ▼
   ┌──────────────────────────┐           ┌──────────────────────────────┐
   │ ChromaDB collection      │◄──────────│ Chunker + Embedder           │
   │  type ∈ {person, place}  │           │  (sentence-transformers,     │
   │  cosine HNSW             │           │   overlap chunking)          │
   └───────────┬──────────────┘           └──────────────────────────────┘
               │
               ▼
   ┌──────────────────────────┐
   │ Ollama (Llama 3.2 3B)    │  ← local LLM, strict system prompt
   └──────────────────────────┘
```

**Key design choices** (each is justified inline in the source):

| Concern        | Choice                              | Rationale (one-liner)                                     |
|----------------|-------------------------------------|-----------------------------------------------------------|
| Vector store   | One Chroma collection + metadata    | Option B — simpler ops, scales to more domains            |
| Distance       | Cosine                              | Stable on normalized sentence embeddings                  |
| Chunking       | 800 chars, 150 overlap, sentence-aware | Balances context limit vs. recall                      |
| Embedding      | `all-MiniLM-L6-v2` (384-d)          | Fast on CPU, strong MTEB score, 80 MB                     |
| Routing        | Keyword + alias matching            | Deterministic, no extra LLM call                          |
| Generation     | Llama 3.2 3B via Ollama             | 8K context, ~2 GB RAM, friendly licensing                 |

---

## 2. Project layout

```
RAGasistant/
├── README.md                ← this file
├── product_prd.md           ← Product Requirements Document
├── recommendation.md        ← Production deployment recommendations
├── requirements.txt
├── config.py                ← all hyperparameters, paths, prompt
├── app.py                   ← Streamlit chat UI
├── main.py                  ← lightweight CLI chat
├── scripts/
│   ├── run_ingest.py        ← run ingest + index pipeline
│   └── reset_db.py          ← wipe local SQLite + Chroma
├── src/
│   ├── ingest.py            ← Wikipedia → SQLite
│   ├── chunker.py           ← overlap, sentence-aware chunker
│   ├── embedder.py          ← sentence-transformers wrapper
│   ├── vector_store.py      ← ChromaDB facade (Option B)
│   ├── retriever.py         ← keyword router + filtered retrieval
│   ├── generator.py         ← Ollama Llama 3.2 client
│   └── rag_pipeline.py      ← orchestrator (ingest, ask, stream)
└── data/                    ← created at runtime (gitignored)
    ├── wiki_cache.sqlite
    └── chroma/
```

---

## 3. Prerequisites

- **Python 3.10 +**
- **Ollama** installed locally — <https://ollama.com/download>
- ~3 GB free disk (Llama 3.2 3B model + Chroma index + Wikipedia cache)
- macOS / Linux / Windows (tested on macOS Apple Silicon and Ubuntu 22.04)

> The first run downloads the Llama 3.2 model (~2 GB) and the embedding
> model (~80 MB). Subsequent runs are offline.

---

## 4. Step-by-step setup

### 4.1. Install Ollama and pull the model

```bash
# (one-time) install Ollama from https://ollama.com/download
# then start the local daemon
ollama serve            # leave running in a terminal
# in a second terminal:
ollama pull llama3.2:3b
```

You can verify it works with:

```bash
ollama run llama3.2:3b "Say hi in one sentence."
```

### 4.2. Create a virtual environment and install Python deps

```bash
cd /path/to/RAGasistant
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 4.3. Ingest Wikipedia data and build the index

```bash
python -m scripts.run_ingest
```

What this does:

1. Downloads 20 famous people + 20 famous places from Wikipedia.
2. Stores raw pages in `data/wiki_cache.sqlite` (idempotent).
3. Chunks each article (800 chars / 150 overlap, sentence-aware).
4. Computes local embeddings with `all-MiniLM-L6-v2`.
5. Writes everything into a single Chroma collection with `type ∈
   {person, place}` metadata.

You should see something like:

```json
{
  "new_people": 20,
  "new_places": 20,
  "total_in_db": 40,
  "chunks_added": 873,
  "vector_count": 873,
  "elapsed_sec": 41.7
}
```

### 4.4. Launch the chat UI

```bash
streamlit run app.py
```

Streamlit opens at <http://localhost:8501>.

If you prefer the terminal:

```bash
python main.py                                  # interactive REPL
python main.py "What did Marie Curie discover?" # one-shot
```

---

## 5. Example queries

**People**

- *Who was Albert Einstein and what is he known for?*
- *What did Marie Curie discover?*
- *Why is Nikola Tesla famous?*
- *Compare Lionel Messi and Cristiano Ronaldo.*

**Places**

- *Where is the Eiffel Tower located?*
- *Why is the Great Wall of China important?*
- *What is Machu Picchu?*
- *Where is Mount Everest?*

**Mixed / comparison**

- *Which famous place is located in Turkey?*
- *Which person is associated with electricity?*
- *Compare the Eiffel Tower and the Statue of Liberty.*

**Failure cases (the model should say "I don't know")**

- *Who is the president of Mars?*
- *Tell me about a random unknown person John Doe.*

The Streamlit sidebar exposes a **"Show source chunks"** toggle that prints
the exact passages used to ground every answer, plus the routing intent
(person/place/both/unknown) detected by the retriever.

---

## 6. Configuration

All knobs live in `config.py`:

| Variable                   | Default                                    | Notes                       |
|----------------------------|--------------------------------------------|-----------------------------|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 150                              | character-based             |
| `EMBEDDING_MODEL_NAME`     | `sentence-transformers/all-MiniLM-L6-v2`   | env: `RAG_EMBEDDING_MODEL`  |
| `OLLAMA_MODEL`             | `llama3.2:3b`                              | env: `RAG_LLM_MODEL`        |
| `OLLAMA_HOST`              | `http://localhost:11434`                   | env: `OLLAMA_HOST`          |
| `TOP_K`                    | 5                                          | per-query                   |
| `GENERATION_TEMPERATURE`   | 0.2                                        | low = less hallucination    |

Switch the LLM with one env var:

```bash
RAG_LLM_MODEL=mistral:7b streamlit run app.py
```

---

## 7. Troubleshooting

| Symptom                                              | Fix                                                                      |
|------------------------------------------------------|--------------------------------------------------------------------------|
| `Ollama'ya bağlanılamadı` / Connection refused       | Start `ollama serve` in another terminal, or set `OLLAMA_HOST`           |
| Model "I don't know" everything                      | Index is empty → run `python -m scripts.run_ingest`                      |
| Streamlit shows "Vector store boş"                   | Same as above                                                            |
| Wikipedia DisambiguationError on rare titles         | Edit `config.py` and use the canonical title (e.g. *Giza pyramid complex*) |
| Want to start fresh                                  | `python -m scripts.reset_db && python -m scripts.run_ingest`             |

---

## 8. Demo video

A 5-minute walk-through (system overview, live ingestion + Q&A, technical
decisions, trade-offs, future work) is available at:

**▶︎ Demo video: `<https://youtu.be/Vrh-Fl35Jx4>`**

---

## 9. License

MIT — feel free to use, modify, and learn from this code.
