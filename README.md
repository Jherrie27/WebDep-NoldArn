# Knightsbridge FP Handbook Bot — Streamlit Deployment

## Folder Structure

```
knightsbridge_streamlit/
├── app.py                         # Streamlit UI
├── pipeline.py                    # RAG pipeline (retrieval, generation, scoring)
├── requirements.txt
├── data/
│   └── phfphandbook-compressed.pdf   ← PUT YOUR PDF HERE
└── .streamlit/
    ├── config.toml
    └── secrets.toml               ← PUT YOUR API KEYS HERE
```

## Steps to Deploy

### 1. Add your PDF
Place `phfphandbook-compressed.pdf` in the `data/` folder.

### 2. Fill in secrets
Edit `.streamlit/secrets.toml`:
```toml
HF_TOKEN     = "hf_..."    # HuggingFace token (for Qwen3-4B via HF Router)
GROQ_API_KEY = "gsk_..."   # Groq API key (for Llama judge + Llama generation fallback)
```

### 3. Deploy to Streamlit Cloud
1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your repo, set main file to `app.py`
4. Add your secrets in the Streamlit Cloud secrets panel (same keys as secrets.toml)
5. Deploy

### 4. Run locally (for testing)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What the App Does

- Accepts natural language questions about the Philippine Family Planning Handbook
- Retrieves relevant passages using BM25 + FAISS hybrid retrieval (60/40 ensemble)
- Reranks using BGE cross-encoder
- Generates answers using Qwen3-4B via HuggingFace Router
- **Automatically scores every answer** using an independent LLM judge (Llama-3.1-8B via Groq):
  - Faithfulness (1–5): are all claims grounded in the retrieved context?
  - Answer Relevancy (1–5): does the answer address the question?
  - Context Relevancy (1–5): does the retrieved context contain the needed information?
- Displays latency, retrieved chunks with page numbers, and full context
- Supports multi-turn conversation with query condensation

## Models Used
| Role | Model | Provider |
|---|---|---|
| Embeddings | nomic-ai/nomic-embed-text-v1.5 | HuggingFace |
| Reranker | BAAI/bge-reranker-base | HuggingFace |
| Answer generation | Qwen3-4B-Instruct-2507 | HuggingFace Router |
| Judge scoring | llama-3.1-8b-instant | Groq |
