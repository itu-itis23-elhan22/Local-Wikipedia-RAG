# Production Deployment Recommendations

This document outlines what would change if we wanted to take the Local
Wikipedia RAG Assistant from a laptop demo to a real production service.
The current code is intentionally optimized for **clarity and locality**;
production introduces additional concerns around **scale, reliability,
security, and cost**.

---

## 1. Containerization & reproducibility

### 1.1. Multi-stage Docker image

```dockerfile
# ------------------------------------------------------------
# Stage 1 — base with Ollama + pre-pulled model
# ------------------------------------------------------------
FROM ollama/ollama:0.3.6 AS ollama
RUN ollama serve & sleep 5 && ollama pull llama3.2:3b && pkill ollama

# ------------------------------------------------------------
# Stage 2 — application image
# ------------------------------------------------------------
FROM python:3.11-slim
WORKDIR /app

# OS deps for chromadb & sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding weights so the first request is fast
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 1.2. `docker-compose.yml`

Run Ollama and the Streamlit app side by side:

```yaml
version: "3.9"
services:
  ollama:
    image: ollama/ollama:0.3.6
    volumes:
      - ollama-models:/root/.ollama
    ports: ["11434:11434"]
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 10s
      timeout: 3s
      retries: 5
    deploy:
      resources:
        reservations:
          devices: [{ driver: nvidia, count: 1, capabilities: [gpu] }]

  rag:
    build: .
    environment:
      OLLAMA_HOST: http://ollama:11434
      RAG_LLM_MODEL: llama3.2:3b
    depends_on:
      ollama: { condition: service_healthy }
    ports: ["8501:8501"]
    volumes:
      - rag-data:/app/data

volumes:
  ollama-models:
  rag-data:
```

This guarantees byte-for-byte identical environments across developer
laptops, CI, and production.

---

## 2. GPU acceleration

The current default (`llama3.2:3b`, 4-bit quantized) runs on CPU at ~12
tokens/sec on an M2. For latency-sensitive deployments:

| Tier              | Hardware           | Tokens/sec (Llama 3.2 3B) | Latency (p50) |
|-------------------|--------------------|---------------------------|---------------|
| Laptop (CPU)      | Apple M2 / Ryzen 7 | 8 – 14                    | ~2.5 s        |
| Mid-range GPU     | RTX 3060 (12 GB)   | 70 – 110                  | ~0.5 s        |
| Server GPU        | A10 / L4           | 150 – 250                 | ~0.25 s       |
| High-end          | A100 / H100        | 400 +                     | < 0.15 s      |

**Recommendations**

- **Embeddings on GPU** — `SentenceTransformer(..., device="cuda")` is a
  ~10× speedup at ingest time.
- **Pin a CUDA driver** in the Docker image (`nvidia/cuda:12.4.1-base`).
- **Quantization** — for production, switch to a fp16 / 4-bit GGUF build
  (`llama3.2:3b-instruct-q4_K_M`). Quality loss is small; throughput
  doubles.
- **Speculative decoding** with a 1B draft model (`llama3.2:1b`) gives
  ~2× speedup with no quality loss.

---

## 3. Scalability & data layer

### 3.1. Vector store

Chroma is excellent up to ~10⁶ vectors. Beyond that, swap the
`VectorStore` facade to:

- **Qdrant** — Rust core, gRPC, native quantization, payload filtering.
- **Weaviate** — multi-tenant, hybrid (BM25+vector) out of the box.
- **PGVector** — when ops already runs Postgres and prefers one fewer
  service.
- **Milvus / Zilliz** — for ≥ 10⁸ vectors.

Because all DB access is hidden behind `src/vector_store.py`, the swap is
a one-file change.

### 3.2. Ingest pipeline

For real corpora (millions of articles):

- Move from synchronous `wikipedia` calls to a **batch dump** of the
  Wikipedia XML (`pages-articles-multistream.xml.bz2`).
- Stream chunks into the vector DB via a **task queue** (Celery + Redis,
  or Temporal). Idempotency via deterministic chunk IDs is already in
  place.
- Add **content-hash dedup** so re-ingest of unchanged articles is a
  no-op.

### 3.3. Caching

- **Embedding cache**: hash(text) → vector, persisted (Redis or RocksDB).
  Saves repeat embedding cost at chunk-overlap boundaries.
- **Answer cache**: `hash(query, top_chunks)` → answer with TTL. Cuts
  cost ~30 % in real traffic where users repeat popular questions.

---

## 4. Retrieval quality

| Improvement              | Estimated lift on recall@5 | Effort |
|--------------------------|---------------------------|--------|
| Hybrid BM25 + vector      | +6 to +10 pp              | Low    |
| Cross-encoder re-ranker   | +8 to +15 pp              | Medium |
| Multi-query expansion     | +3 to +6 pp               | Low    |
| Domain-tuned embedder     | +5 to +8 pp               | High   |

**Re-ranker recipe**

1. Retrieve `top_k = 30` from Chroma.
2. Score each (query, chunk) pair with `cross-encoder/ms-marco-MiniLM-L-6-v2`
   (still local, ~30 ms / pair on CPU).
3. Keep top 5; pass to LLM.

---

## 5. Observability

- **Tracing** — wrap retrieve/generate with OpenTelemetry; each span
  records latency, top_k, intent, and token counts.
- **Metrics** — Prometheus counters: `queries_total`,
  `i_dont_know_total`, `latency_seconds_bucket`.
- **Logs** — structured JSON, redacting raw user content if PII is a
  concern.
- **Eval suite** — version-control a `tests/eval_questions.jsonl` with
  ~50 grader-style prompts; CI computes faithfulness, answer relevance,
  and "I don't know" rate using `ragas` (local mode).

---

## 6. Reliability & ops

- **Liveness/readiness** — `/healthz` for Streamlit, `/api/tags` for
  Ollama. Use the Docker healthcheck above.
- **Graceful degradation** — if the LLM is unreachable, surface a "RAG
  search worked but the writer is offline" message and still show the
  retrieved chunks.
- **Rate limiting** — front the service with Caddy or Traefik with a
  token-bucket per IP. Local LLM throughput is finite; protect it.
- **Backup** — `data/wiki_cache.sqlite` and `data/chroma/` to S3 nightly.

---

## 7. Security & privacy

- **Network egress**: lock the container to allow only Wikipedia REST
  during ingest, and zero outbound traffic at query time.
- **Input sanitization**: cap query length (e.g. 1000 chars), strip
  control bytes; user content is fed verbatim into the prompt, so the
  system prompt MUST reiterate refusal rules.
- **Model jailbreaks**: log prompts that bypass "I don't know" thresholds;
  retrain the system prompt against the leaked patterns.
- **Tenant isolation**: when serving multiple users, use a `tenant_id`
  metadata field and always inject a `where={"tenant_id": ...}` filter.

---

## 8. Cost model (informal)

For a single-tenant deployment serving ~1 000 queries/day:

| Component        | Hosting                | ~ monthly cost |
|------------------|------------------------|----------------|
| GPU server (L4)  | Hetzner / OVH / GCP    | $250 – 500     |
| Object storage   | S3 (≤ 50 GB)           | $1             |
| Egress           | (none from LLM)        | ~$0            |
| Ops/monitoring   | Grafana Cloud free     | $0 – 50        |

Compared to GPT-4-class APIs at similar volume (~$300+ in pure inference
costs alone), running locally pays back the GPU within a few months and
gives full data sovereignty.

---

## 9. Roadmap (prioritized)

1. **v1.1 — Quality**
   - Add BM25 hybrid retrieval and a cross-encoder reranker.
   - Multilingual embedder (Turkish queries).
   - Eval harness in CI.

2. **v1.2 — Scale**
   - Switch to Qdrant.
   - Batch ingest from Wikipedia dumps.
   - Async generation with token streaming over WebSockets.

3. **v1.3 — Productization**
   - SSO (OIDC) on the chat UI.
   - Per-user chat history (SQLite → Postgres).
   - Fine-grained metadata: language, last_modified, popularity score.

4. **v1.4 — Multimodal**
   - Image-of-the-page retrieval (CLIP-based).
   - Whisper-based voice queries → text → RAG.

---

## 10. Closing notes

The current local-only architecture is **not a toy**: it is a deliberate
slice of a production system. By keeping each layer (ingest, chunk, embed,
store, retrieve, generate, UI) behind a thin, replaceable interface, the
upgrade path described above is incremental — none of it requires
re-architecting the codebase.
