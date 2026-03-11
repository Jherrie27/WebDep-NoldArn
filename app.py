"""
app.py — Streamlit UI for Knightsbridge Family Planning Handbook RAG chatbot
Adapted from Copy_of_CM1_Knightsbridge.ipynb
"""

from __future__ import annotations

import traceback

import streamlit as st

from pipeline import MODEL_DISPLAY_NAME, generate_answer, healthcheck, warmup

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="FP Handbook Bot",
    page_icon="🌿",
    layout="wide",
)

# =============================================================================
# SESSION STATE
# =============================================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "example_query" not in st.session_state:
    st.session_state.example_query = ""

if "input_value" not in st.session_state:
    st.session_state.input_value = ""

# =============================================================================
# STARTUP
# =============================================================================

startup_error = None
try:
    warmup()
except Exception as e:
    startup_error = str(e)

# =============================================================================
# HELPERS
# =============================================================================

def reset_app() -> None:
    st.session_state.chat_history = []
    st.session_state.example_query = ""
    st.session_state.input_value = ""


def score_color(score: int) -> str:
    """Return a color hex based on 1–5 score."""
    if score >= 4:
        return "🟢"
    elif score == 3:
        return "🟡"
    else:
        return "🔴"


def render_eval_scores(turn: dict) -> None:
    """Render the eval metric scores panel below each answer."""
    faith = turn.get("faithfulness", 0)
    ansrel = turn.get("answer_relevancy", 0)
    ctxrel = turn.get("context_relevancy", 0)
    is_refusal = turn.get("is_refusal", False)

    st.markdown("**📊 Evaluation Scores**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        label=f"{score_color(faith)} Faithfulness",
        value=f"{faith} / 5",
        help="Is every claim in the answer supported by the retrieved context? (LLM-as-Judge, 1–5)"
    )
    col2.metric(
        label=f"{score_color(ansrel)} Answer Relevancy",
        value=f"{ansrel} / 5",
        help="Does the answer directly address the question? (LLM-as-Judge, 1–5)"
    )
    col3.metric(
        label=f"{score_color(ctxrel)} Context Relevancy",
        value=f"{ctxrel} / 5",
        help="Does the retrieved context contain what's needed to answer the question? (LLM-as-Judge, 1–5)"
    )
    col4.metric(
        label="⏱️ Latency",
        value=f"{turn.get('latency_sec', 0.0):.2f}s",
        help="Wall-clock time from query submission to answer display."
    )

    if is_refusal:
        st.info(
            "ℹ️ Out-of-scope query detected — the answer was correctly refused. "
            "Faithfulness is set to 5 (correct refusal), Answer Relevancy is 1 (no direct answer).",
            icon="🚫",
        )

    # Score legend
    st.caption("🟢 4–5 · 🟡 3 · 🔴 1–2  |  Judge model: llama-3.1-8b-instant (independent, via Groq)")


def render_chunks(chunks: list[dict]) -> None:
    if not chunks:
        st.info("No retrieved chunks available.")
        return
    for i, row in enumerate(chunks, start=1):
        with st.expander(f"Chunk {i} — Page {row.get('page', '?')}", expanded=False):
            st.write(row.get("text", ""))


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("🌿 FP Handbook Bot")
    st.caption(f"Model: {MODEL_DISPLAY_NAME}")

    st.markdown("### About")
    st.write(
        "RAG chatbot for the **Philippine Family Planning Handbook (2023 Edition)**. "
        "Answers are grounded in retrieved handbook passages and scored automatically "
        "after every response."
    )

    if st.button("Clear conversation", use_container_width=True):
        reset_app()
        st.rerun()

    with st.expander("System status", expanded=False):
        try:
            st.json(healthcheck())
        except Exception as e:
            st.error(f"Healthcheck failed: {e}")

    st.markdown("---")
    st.markdown("### Example questions")

    examples = [
        ("GATHER approach steps",       "What are the steps in the GATHER approach?"),
        ("COC contraindications",        "Who should not use combined oral contraceptives?"),
        ("IUD heavy bleeding",           "What should a provider do if a client experiences heavy bleeding after IUD insertion?"),
        ("Mini laparotomy procedure",    "Give me the Mini laparotomy procedure."),
        ("Yuzpe method dosage",          "Give me the recommended dose of acceptable brands for the Yuzpe Method."),
        ("LAM effectiveness",            "Can LAM be an effective method of family planning?"),
        ("FP Outreach setup",            "What are the steps in setting up and implementing FP Outreach?"),
    ]

    for label, query in examples:
        if st.button(label, use_container_width=True):
            st.session_state.example_query = query
            st.session_state.input_value   = query


# =============================================================================
# MAIN
# =============================================================================

st.title("🌿 Philippine Family Planning Handbook Bot")
st.caption("Powered by Qwen3-4B · Hybrid BM25 + FAISS · BGE Reranker · Groq")

if startup_error:
    st.error(
        "Startup failed. Check your secrets and that the PDF is in data/.\n\n"
        f"Details: {startup_error}"
    )
    st.stop()

st.markdown(
    """
Ask any question about the **Philippine Family Planning Handbook (2023 Edition)**.

This chatbot uses:
- `nomic-embed-text-v1.5` embeddings (FAISS dense retrieval)
- BM25 sparse retrieval with 60/40 ensemble fusion
- `BAAI/bge-reranker-base` cross-encoder reranking
- `Qwen3-4B-Instruct-2507` answer generation (HuggingFace Router)
- `llama-3.1-8b-instant` independent judge scoring (Groq)

Every answer is automatically scored on **Faithfulness**, **Answer Relevancy**, and **Context Relevancy** (1–5 scale).
"""
)

# =============================================================================
# INPUT FORM
# =============================================================================

default_query = st.session_state.input_value or st.session_state.example_query

with st.form("ask_form", clear_on_submit=False):
    user_query = st.text_area(
        "Question",
        value=default_query,
        height=100,
        placeholder="e.g. What are the steps in the GATHER approach?",
    )
    submitted = st.form_submit_button("Ask", use_container_width=True)

# =============================================================================
# RUN QUERY
# =============================================================================

if submitted:
    clean_query = user_query.strip()

    if not clean_query:
        st.warning("Please enter a valid question.")
    elif len(clean_query) < 5:
        st.warning("Question too short.")
    else:
        st.session_state.input_value   = ""
        st.session_state.example_query = ""

        try:
            with st.spinner("Retrieving context, generating answer, and scoring..."):
                result = generate_answer(
                    clean_query,
                    history=st.session_state.chat_history,
                )

            st.session_state.chat_history.append(
                {
                    "user":              clean_query,
                    "assistant":         result["answer"],
                    "latency_sec":       result.get("latency_sec", 0.0),
                    "faithfulness":      result.get("faithfulness", 0),
                    "answer_relevancy":  result.get("answer_relevancy", 0),
                    "context_relevancy": result.get("context_relevancy", 0),
                    "is_refusal":        result.get("is_refusal", False),
                    "context":           result.get("context", ""),
                    "chunks":            result.get("chunks", []),
                }
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc(), language="python")

# =============================================================================
# CHAT HISTORY
# =============================================================================

if st.session_state.chat_history:
    for turn in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(turn["user"])

        with st.chat_message("assistant"):
            st.write(turn["assistant"])

            # ── EVAL SCORES ──────────────────────────────────────────
            render_eval_scores(turn)

            # ── RETRIEVED CONTEXT ─────────────────────────────────────
            with st.expander("📄 Retrieved context", expanded=False):
                ctx = turn.get("context", "")
                if ctx:
                    st.text(ctx)
                else:
                    st.info("No context returned.")

            with st.expander("🔍 Retrieved chunks (with page numbers)", expanded=False):
                render_chunks(turn.get("chunks", []))

            st.divider()

else:
    st.info("Ask a question above to begin.")
