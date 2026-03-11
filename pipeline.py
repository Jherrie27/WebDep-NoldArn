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

    try:
        import streamlit as st
        hf_key   = str(st.secrets["HF_TOKEN"])
        groq_key = str(st.secrets["GROQ_API_KEY"])
    except Exception:
        hf_key   = os.getenv("HF_TOKEN", "")
        groq_key = os.getenv("GROQ_API_KEY", "")

    if not hf_key:
        raise ValueError("HF_TOKEN secret is missing or empty")
    if not groq_key:
        raise ValueError("GROQ_API_KEY secret is missing or empty")

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

_splits    = None
_faiss     = None
_bm25      = None
_embeddings = None
_reranker  = None

# ─────────────────────────────────────────────────────────────
# RETRIEVER
# ─────────────────────────────────────────────────────────────

def _build_retriever():
    global _splits, _faiss, _bm25, _embeddings

    import faiss
    from rank_bm25 import BM25Okapi
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings

    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"chunks.json missing from {DATA_DIR}")
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(f"faiss.index missing from {DATA_DIR}")
    if not os.path.exists(BM25_PATH):
        raise FileNotFoundError(f"bm25.pkl missing from {DATA_DIR}")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    _splits = [
        Document(
            page_content=c["text"],
            metadata={"page": c["page"]}
        )
        for c in chunks
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_ID,
        model_kwargs={
            "device": device,
            "trust_remote_code": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    _faiss = faiss.read_index(FAISS_PATH)

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

        device = "cuda" if torch.cuda.is_available() else "cpu"

        _reranker = CrossEncoder(RERANK_ID, device=device)

    return _reranker

# ─────────────────────────────────────────────────────────────
# RETRIEVE + RERANK
# ─────────────────────────────────────────────────────────────

def retrieve_and_rerank(query: str):
    splits   = _get_retriever()
    reranker = _get_reranker()

    bm25_scores = _bm25.get_scores(query.split())
    top_idx     = np.argsort(bm25_scores)[::-1][:10]
    initial_docs = [splits[i] for i in top_idx]

    scores = reranker.predict([[query, d.page_content] for d in initial_docs])
    ranked  = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in ranked[:3]]

    context = "\n\n---\n\n".join([d.page_content for d in top_docs])

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
