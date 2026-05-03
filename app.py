"""
app.py
======
Streamlit UI — Local Wikipedia RAG Assistant.

To run:
    streamlit run app.py

Features
----------
- Modern, clean chat interface.
- Streams the answer token by token (streaming).
- "Show source chunks" toggle: which documents were used to generate the answer?
- Sidebar:
    * System status (vector count, model, host)
    * "Reset chat" — clears chat history.
    * "Reset index" — deletes ChromaDB collection.
    * top_k slider.
- Hot reload compatible cache (RAGPipeline instantiated once).
"""

from __future__ import annotations

import logging
from typing import Iterable, List

import streamlit as st

from config import OLLAMA_HOST, OLLAMA_MODEL, TOP_K
from src.rag_pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Local Wikipedia RAG",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Pipeline (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models and vector store...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # List[ {role, content, sources, intent} ]
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "top_k" not in st.session_state:
        st.session_state.top_k = TOP_K


_init_state()
pipeline = get_pipeline()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title(":books: RAG Settings")

    stats = pipeline.stats()
    st.metric("Indexed chunks", stats["vector_count"])
    st.caption(f"Model: `{stats['model']}`")
    st.caption(f"Host: `{stats['host']}`")
    st.caption(f"Indexed titles: {len(stats['titles_indexed'])}")

    if stats["vector_count"] == 0:
        st.warning(
            "Vector store is empty. In a terminal:\n\n"
            "```bash\npython -m scripts.run_ingest\n```"
        )

    with st.expander("Indexed entities", expanded=False):
        st.write(stats["titles_indexed"])

    st.divider()
    st.subheader("Retrieval")
    st.session_state.top_k = st.slider(
        "Top-k chunks", min_value=1, max_value=12,
        value=st.session_state.top_k,
        help="Her sorguda LLM'e verilecek pasaj sayısı.",
    )
    st.session_state.show_sources = st.toggle(
        "Show source chunks",
        value=st.session_state.show_sources,
        help="Cevabın altında hangi Wikipedia pasajlarının kullanıldığını göster.",
    )

    st.divider()
    st.subheader("Maintenance")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    danger = st.expander(":warning: Danger zone", expanded=False)
    with danger:
        st.caption(
            "Bu butonlar kalıcıdır. Index sıfırlandıktan sonra tekrar "
            "ingest etmeniz gerekir."
        )
        if st.button("Reset vector index"):
            pipeline.reset_index()
            st.success("Index sıfırlandı. Şimdi `python -m scripts.run_ingest` çalıştır.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Local Wikipedia RAG Assistant")
st.caption(
    "ChatGPT-style answers about famous people and places — "
    f"powered by **{OLLAMA_MODEL}** on `{OLLAMA_HOST}`. "
    "All retrieval and generation runs on your laptop."
)


# ---------------------------------------------------------------------------
# Chat history render
# ---------------------------------------------------------------------------
def _render_sources(sources: List[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources", expanded=False):
        for i, c in enumerate(sources, start=1):
            m = c["metadata"]
            sim = c.get("similarity", 0.0)
            st.markdown(
                f"**[{i}] {m['title']}** "
                f"({m['type']}) — similarity `{sim:.3f}`  \n"
                f"<{m.get('url', '')}>"
            )
            st.code(c["text"], language="markdown")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and st.session_state.show_sources:
            _render_sources(msg.get("sources", []))


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
prompt = st.chat_input(
    "Ask about a famous person or place "
    "(e.g. 'What did Marie Curie discover?')"
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bağımsız bir retrieve+generate cycle çalıştır.
    with st.chat_message("assistant"):
        try:
            stream, intent, chunks = pipeline.stream(
                prompt, top_k=st.session_state.top_k
            )
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

        intent_line = (
            f":mag: **intent**: `{intent.query_type.value}`"
            f" — people={intent.mentioned_people or '∅'}"
            f", places={intent.mentioned_places or '∅'}"
        )
        st.caption(intent_line)

        # Streaming yanıt
        def _to_strings(it: Iterable[str]):
            for tok in it:
                yield tok

        if not chunks:
            full_answer = "I don't know."
            st.markdown(full_answer)
        else:
            full_answer = st.write_stream(_to_strings(stream))

        if st.session_state.show_sources:
            _render_sources(chunks)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_answer or "I don't know.",
            "sources": chunks,
            "intent": intent.query_type.value,
        }
    )
