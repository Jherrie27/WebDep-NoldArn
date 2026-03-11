"""
pipeline.py — Knightsbridge Family Planning Handbook RAG pipeline
Extracted and adapted from Copy_of_CM1_Knightsbridge.ipynb
"""

from __future__ import annotations

import gc
import os
import re
import time

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PDF_PATH   = os.path.join(os.path.dirname(__file__), "data", "phfphandbook-compressed.pdf")
EMBED_ID   = "nomic-ai/nomic-embed-text-v1.5"
RERANK_ID  = "BAAI/bge-reranker-base"
QWEN_MODEL  = "Qwen/Qwen3-4B-Instruct-2507"
LLAMA_MODEL = "llama-3.1-8b-instant"
JUDGE_MODEL = "llama-3.1-8b-instant"

MODEL_DISPLAY_NAME = "Qwen3-4B + Llama-3.1-8B (Hybrid)"

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS  (identical to notebook)
# ─────────────────────────────────────────────────────────────────────────────

QA_SYSTEM = """You are a highly capable and intelligent clinical extraction AI.
STRICT RULES:
1. USE ONLY the information provided in the [Context] section to formulate your answer.
2. INTENT & SYNONYM MATCHING (CRITICAL): Do not be overly rigid with keyword matching. Users may use imprecise words (e.g., asking for "methods" when they mean "stages", "steps", "types", or "components"). If the exact word isn't found, analyze the context to see if it describes the underlying concept under a different term, and answer using the terminology from the context.
3. PARTIAL ANSWERS ONLY — NO INFERENCE: If the context contains SOME but not all details, present ONLY what is explicitly stated. Do NOT reconstruct, infer, or logically derive steps that are not directly written in the context. If a step is missing from the context, omit it entirely rather than filling it in.
4. CONTEXT FRAGMENTATION WARNING: The context comes from a 2-column PDF. Lists and acronyms (like G-A-T-H-E-R) are often physically split apart by tables or column breaks.
5. ACRONYM RULE: If asked for an acronym, you MUST scan the entire context to find every single letter. Do not stop early. Jump over unrelated text until you find the remaining letters.
6. If the topic is completely missing from the context, respond ONLY with: "This information was not found in the provided context." """

CONDENSE_SYSTEM = """You are a search query formulation assistant for a family planning handbook.
Given the chat history and a new user question, rewrite the new question into a STANDALONE search query.

CRITICAL RULES:
1. The standalone query must be based on the NEW question's topic, not the previous answers.
2. If the new question introduces a DIFFERENT topic than the chat history, IGNORE the history entirely and just rewrite the new question directly.
3. Do NOT blend topics. Do NOT answer the question. Output ONLY the rewritten query.

Examples:
- History: discussed female sterilization. New question: "Who cannot have a vasectomy?" → Output: "vasectomy contraindications who cannot have"
- History: discussed COCs. New question: "What is LAM?" → Output: "Lactational Amenorrhea Method LAM conditions"
"""

EVAL_SYSTEM = """You are a strict RAG evaluation judge. Score the Answer from 1 to 5.
Do not explain your reasoning. Output scores only.

You are evaluating an EXTERNAL chatbot's answer, not your own output.
Apply the rubric objectively. If the answer contains the correct information
and it is supported by the Context, give a high score even if the wording
differs from the Context. Do NOT penalize for paraphrasing or reordering.

1. Faithfulness      : Is every claim in the Answer supported by the Context?
                       (5 = fully grounded, 1 = hallucinated or contradicts context)
2. Answer Relevancy  : Does the Answer directly address the Question?
                       (5 = fully answers the question, 1 = off-topic or irrelevant)
3. Context Relevancy : Does the Context contain information needed to answer the Question?
                       (5 = context contains exact answer, 1 = context is irrelevant)

SCORING GUIDE:
- Score 5: Answer is complete and all claims trace back to the Context
- Score 4: Answer is mostly correct with minor omissions
- Score 3: Answer is partially correct or partially supported
- Score 2: Answer has significant unsupported claims
- Score 1: Answer contradicts or ignores the Context entirely

OUTPUT FORMAT — EXACTLY AS SHOWN, NO OTHER TEXT:
Faithfulness: <score>
Answer Relevancy: <score>
Context Relevancy: <score>"""

# ─────────────────────────────────────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────────────────────────────────────

def _get_clients():
    """Lazily build API clients from Streamlit secrets or env vars."""
    import streamlit as st
    from openai import OpenAI

    hf_key   = st.secrets.get("HF_TOKEN",    os.environ.get("HF_TOKEN", ""))
    groq_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

    hf_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_key,
    )
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_key,
    )
    return hf_client, groq_client


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVER (built once, cached in module-level variable)
# ─────────────────────────────────────────────────────────────────────────────

_splits        = None
_hybrid_retriever = None
_reranker      = None
_embeddings    = None

def _build_retriever():
    global _splits, _hybrid_retriever, _embeddings

    import pymupdf4llm
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_huggingface import HuggingFaceEmbeddings

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found at {PDF_PATH}. "
            "Please place 'phfphandbook-compressed.pdf' inside the data/ folder."
        )

    md_pages = pymupdf4llm.to_markdown(PDF_PATH, page_chunks=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=1000,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )

    splits = []
    for page_data in md_pages:
        text     = page_data.get("text", "")
        page_num = page_data.get("metadata", {}).get("page", 0) + 1
        for chunk in splitter.split_text(text):
            splits.append(Document(page_content=chunk, metadata={"page": page_num}))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_ID,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )

    faiss_retriever = FAISS.from_documents(splits, _embeddings).as_retriever(
        search_kwargs={"k": 10}
    )
    bm25_retriever      = BM25Retriever.from_documents(splits)
    bm25_retriever.k    = 10

    _hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.6, 0.4],
    )
    _splits = splits

    # Free indexing memory
    del md_pages
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def _get_retriever():
    if _hybrid_retriever is None:
        _build_retriever()
    return _hybrid_retriever


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _reranker = CrossEncoder(RERANK_ID, device=device)
    return _reranker


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVE + RERANK  (identical logic to notebook)
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_and_rerank(query: str) -> tuple[str, list[dict]]:
    """
    Returns (context_text, chunk_list) where chunk_list is a list of dicts
    with keys: page, text (for display in the UI).
    """
    retriever = _get_retriever()
    reranker  = _get_reranker()
    splits    = _splits

    initial_docs = retriever.invoke(query)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker.model.to(device)
    try:
        with torch.no_grad():
            scores = reranker.predict([[query, d.page_content] for d in initial_docs])
    finally:
        reranker.model.to("cpu")
        if device == "cuda":
            torch.cuda.empty_cache()

    ranked   = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in ranked[:3]]

    expanded_docs = []
    for top_doc in top_docs:
        try:
            idx = next(i for i, d in enumerate(splits) if d.page_content == top_doc.page_content)
            for offset in [-1, 0, 1]:
                n_idx = idx + offset
                if 0 <= n_idx < len(splits) and splits[n_idx] not in expanded_docs:
                    expanded_docs.append(splits[n_idx])
        except StopIteration:
            if top_doc not in expanded_docs:
                expanded_docs.append(top_doc)

    expanded_docs.sort(key=lambda x: x.metadata.get("page", 0))

    context_text = "\n\n---\n\n".join([d.page_content for d in expanded_docs])
    chunk_list   = [
        {"page": d.metadata.get("page", "?"), "text": d.page_content}
        for d in expanded_docs
    ]
    return context_text, chunk_list


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _generate_qwen(hf_client, system: str, user: str, max_tokens: int = 2048) -> str:
    resp = hf_client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _generate_llama(groq_client, system: str, user: str, max_tokens: int = 2048) -> str:
    resp = groq_client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _generate_judge(groq_client, system: str, user: str, max_tokens: int = 80) -> str:
    resp = groq_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# EVAL SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _last_score(pattern: str, text: str) -> int:
    matches = re.findall(pattern, text)
    return int(matches[-1]) if matches else 0


def _compute_judge_scores(groq_client, query: str, context: str, answer: str) -> dict:
    """Run LLM-as-judge and return Faithfulness, Answer Relevancy, Context Relevancy (1–5)."""
    truncated = "\n---\n".join([c[:800] for c in context.split("\n\n---\n\n")])[:6000]
    user_eval = (
        f"[Context]\n{truncated}\n\n"
        f"[Question]\n{query}\n\n"
        f"[Answer]\n{answer}"
    )

    eval_result = ""
    for attempt in range(4):
        try:
            eval_result = _generate_judge(groq_client, EVAL_SYSTEM, user_eval)
            break
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(8 * (attempt + 1))
            else:
                break

    is_refusal = "not found in the provided context" in answer.lower()

    if is_refusal:
        faithfulness      = 5
        answer_relevancy  = 1
        context_relevancy = 1
    else:
        faithfulness      = _last_score(r"Faithfulness:\s*(\d)",      eval_result)
        answer_relevancy  = _last_score(r"Answer Relevancy:\s*(\d)",  eval_result)
        context_relevancy = _last_score(r"Context Relevancy:\s*(\d)", eval_result)

    return {
        "faithfulness":      faithfulness,
        "answer_relevancy":  answer_relevancy,
        "context_relevancy": context_relevancy,
        "is_refusal":        is_refusal,
        "raw_eval":          eval_result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(query: str, history: list[dict] | None = None) -> dict:
    """
    Full pipeline: condense → retrieve+rerank → generate → judge-score.

    Returns dict with keys:
        answer, context, chunks, latency_sec,
        faithfulness, answer_relevancy, context_relevancy, is_refusal
    """
    hf_client, groq_client = _get_clients()

    start = time.time()

    # 1. Condense query using recent history
    search_query = query
    if history:
        recent = history[-3:]
        hist_str = "\n".join([f"User: {h['user']}\nBot: {h['assistant']}" for h in recent])
        condense_prompt = (
            f"[Chat History]\n{hist_str}\n\n"
            f"[New Question]\n{query}\n\nStandalone Query:"
        )
        try:
            search_query = _generate_qwen(
                hf_client, CONDENSE_SYSTEM, condense_prompt, max_tokens=50
            ).strip()
        except Exception:
            search_query = query  # fallback

    # 2. Retrieve + rerank
    context_text, chunk_list = retrieve_and_rerank(search_query)

    # 3. Build history string for QA prompt
    hist_str = "No history yet."
    if history:
        recent   = history[-3:]
        hist_str = "\n".join([f"User: {h['user']}\nBot: {h['assistant']}" for h in recent])

    qa_user = (
        f"[Chat History]\n{hist_str}\n\n"
        f"[Context]\n{context_text}\n\n"
        f"[Question]\n{query}"
    )

    # 4. Generate answer (Qwen as primary — matches notebook's generate_text)
    answer = _generate_qwen(hf_client, QA_SYSTEM, qa_user)

    latency = round(time.time() - start, 2)

    # 5. LLM-as-judge scoring
    scores = _compute_judge_scores(groq_client, query, context_text, answer)

    return {
        "answer":            answer,
        "context":           context_text,
        "chunks":            chunk_list,
        "latency_sec":       latency,
        "faithfulness":      scores["faithfulness"],
        "answer_relevancy":  scores["answer_relevancy"],
        "context_relevancy": scores["context_relevancy"],
        "is_refusal":        scores["is_refusal"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# WARMUP / HEALTHCHECK
# ─────────────────────────────────────────────────────────────────────────────

def warmup():
    """Pre-load retriever on startup so first query isn't slow."""
    _get_retriever()
    _get_reranker()


def healthcheck() -> dict:
    return {
        "retriever_ready": _hybrid_retriever is not None,
        "reranker_ready":  _reranker is not None,
        "chunks_loaded":   len(_splits) if _splits else 0,
        "pdf_exists":      os.path.exists(PDF_PATH),
        "model":           MODEL_DISPLAY_NAME,
    }
