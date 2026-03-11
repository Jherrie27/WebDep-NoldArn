"""
pipeline.py — Knightsbridge Family Planning Handbook RAG pipeline
"""

from __future__ import annotations

import os
import re
import time
import json
import pickle
import gc

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")
FAISS_PATH  = os.path.join(DATA_DIR, "faiss.index")
BM25_PATH   = os.path.join(DATA_DIR, "bm25.pkl")

EMBED_ID  = "nomic-ai/nomic-embed-text-v1.5"
RERANK_ID = "BAAI/bge-reranker-base"

QWEN_MODEL  = "Qwen/Qwen3-4B-Instruct-2507"
JUDGE_MODEL = "llama-3.1-8b-instant"

MODEL_DISPLAY_NAME = "Qwen3-4B + Llama-3.1-8B (Hybrid)"

# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────

QA_SYSTEM = """You are a highly capable clinical extraction AI.

STRICT RULES:
1. Use ONLY the provided Context.
2. Do NOT hallucinate.
3. If information is missing say:
"This information was not found in the provided context."
"""

CONDENSE_SYSTEM = """Rewrite the new user question into a standalone search query.
Output ONLY the query."""

EVAL_SYSTEM = """Score the answer from 1-5.

Faithfulness: <score>
Answer Relevancy: <score>
Context Relevancy: <score>
"""

# ─────────────────────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────────────────────

def _get_clients():
    from openai import OpenAI

    hf_key   = ""
    groq_key = ""

    try:
        import streamlit as st
        hf_raw   = st.secrets["HF_TOKEN"]
        groq_raw = st.secrets["GROQ_API_KEY"]
        hf_key   = hf_raw if isinstance(hf_raw, str) else str(hf_raw)
        groq_key = groq_raw if isinstance(groq_raw, str) else str(groq_raw)
    except Exception:
        pass

    if not hf_key:
        hf_key = os.getenv("HF_TOKEN", "")
    if not groq_key:
        groq_key = os.getenv("GROQ_API_KEY", "")

    if not hf_key:
        raise ValueError("HF_TOKEN is missing")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is missing")

    hf_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_key,
    )
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_key,
    )

    return hf_client, groq_client

# ─────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────

_splits     = None
_faiss      = None
_bm25       = None
_embeddings = None
_reranker   = None

# ─────────────────────────────────────────────────────────────
# RETRIEVER
# ─────────────────────────────────────────────────────────────

def _parse_chunk(c, index: int):
    """
    Safely parse a chunk regardless of whether chunks.json stores
    dicts  {"text": "...", "page": N}
    or plain strings "..."
    """
    from langchain_core.documents import Document

    if isinstance(c, dict):
        text = c.get("text") or c.get("content") or c.get("page_content") or str(c)
        page = c.get("page") or c.get("page_number") or "?"
    elif isinstance(c, str):
        text = c
        page = "?"
    else:
        text = str(c)
        page = "?"

    return Document(page_content=text, metadata={"page": page, "index": index})


def _build_retriever():
    global _splits, _faiss, _bm25, _embeddings

    import faiss
    from langchain_huggingface import HuggingFaceEmbeddings

    # ── sanity checks ──────────────────────────────────────────
    missing = []
    for label, path in [
        ("chunks.json", CHUNKS_PATH),
        ("faiss.index", FAISS_PATH),
        ("bm25.pkl",    BM25_PATH),
    ]:
        if not os.path.exists(path):
            missing.append(f"{label} not found at {path}")

    if missing:
        raise FileNotFoundError(
            "Missing data files:\n" + "\n".join(missing) +
            f"\n\ndata/ contents: {os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else 'DATA_DIR MISSING'}"
        )

    # ── load chunks ────────────────────────────────────────────
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # raw can be a list of dicts OR a list of strings OR a dict wrapper
    if isinstance(raw, dict):
        # e.g. {"chunks": [...]}
        candidates = raw.get("chunks") or raw.get("data") or list(raw.values())[0]
    elif isinstance(raw, list):
        candidates = raw
    else:
        raise ValueError(f"Unexpected chunks.json format: {type(raw)}")

    _splits = [_parse_chunk(c, i) for i, c in enumerate(candidates)]

    if not _splits:
        raise ValueError("chunks.json parsed to 0 documents — check the file format.")

    # ── embeddings ─────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_ID,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── FAISS ──────────────────────────────────────────────────
    _faiss = faiss.read_index(FAISS_PATH)

    # ── BM25 ───────────────────────────────────────────────────
    with open(BM25_PATH, "rb") as f:
        _bm25 = pickle.load(f)

    gc.collect()


def _get_retriever():
    if _splits is None:
        _build_retriever()
    return _splits


def _get_reranker():
    global _reranker

    if _reranker is None:
        from sentence_transformers import CrossEncoder
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        _reranker = CrossEncoder(RERANK_ID, device=device)

    return _reranker

# ─────────────────────────────────────────────────────────────
# RETRIEVE + RERANK
# ─────────────────────────────────────────────────────────────

def retrieve_and_rerank(query: str):
    splits   = _get_retriever()
    reranker = _get_reranker()

    bm25_scores  = _bm25.get_scores(query.split())
    top_idx      = np.argsort(bm25_scores)[::-1][:10]
    initial_docs = [splits[i] for i in top_idx]

    scores   = reranker.predict([[query, d.page_content] for d in initial_docs])
    ranked   = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in ranked[:3]]

    context    = "\n\n---\n\n".join([d.page_content for d in top_docs])
    chunk_list = [
        {"page": d.metadata.get("page", "?"), "text": d.page_content}
        for d in top_docs
    ]

    return context, chunk_list

# ─────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────

def _generate(client, model, system, user, max_tokens=2048):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()

# ─────────────────────────────────────────────────────────────
# JUDGE
# ─────────────────────────────────────────────────────────────

def _last_score(pattern: str, text: str) -> int:
    matches = re.findall(pattern, text)
    return int(matches[-1]) if matches else 0


def _judge(groq_client, query, context, answer):
    user_eval = f"""Context:
{context}

Question:
{query}

Answer:
{answer}
"""
    result = _generate(groq_client, JUDGE_MODEL, EVAL_SYSTEM, user_eval, 80)

    return {
        "faithfulness":      _last_score(r"Faithfulness:\s*(\d)",      result),
        "answer_relevancy":  _last_score(r"Answer Relevancy:\s*(\d)",  result),
        "context_relevancy": _last_score(r"Context Relevancy:\s*(\d)", result),
    }

# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def generate_answer(query: str, history=None):
    hf_client, groq_client = _get_clients()

    start        = time.time()
    search_query = query

    if history:
        hist = "\n".join(
            [f"User:{h['user']}\nBot:{h['assistant']}" for h in history[-3:]]
        )
        condensed    = _generate(
            hf_client,
            QWEN_MODEL,
            CONDENSE_SYSTEM,
            f"{hist}\n\nQuestion:{query}",
            50,
        )
        search_query = condensed.strip()

    context, chunks = retrieve_and_rerank(search_query)

    qa_prompt = f"""Context:
{context}

Question:
{query}
"""

    answer  = _generate(hf_client, QWEN_MODEL, QA_SYSTEM, qa_prompt)
    latency = round(time.time() - start, 2)
    scores  = _judge(groq_client, query, context, answer)

    return {
        "answer":      answer,
        "context":     context,
        "chunks":      chunks,
        "latency_sec": latency,
        **scores,
    }

# ─────────────────────────────────────────────────────────────
# WARMUP / HEALTHCHECK
# ─────────────────────────────────────────────────────────────

def warmup():
    _get_retriever()
    _get_reranker()


def healthcheck():
    return {
        "retriever_ready": _splits   is not None,
        "reranker_ready":  _reranker is not None,
        "chunks_loaded":   len(_splits) if _splits else 0,
        "model":           MODEL_DISPLAY_NAME,
    }
