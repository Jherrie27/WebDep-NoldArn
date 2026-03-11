# PH Family Planning Handbook AI — Streamlit Deployment

RAG chatbot grounded on the Philippine Family Planning Handbook (2023 Edition).  
Uses hybrid BM25 + FAISS retrieval, BGE cross-encoder reranking, and Qwen3-4B via HuggingFace Inference API.

---

## File Structure

```
fp-handbook-chatbot/
├── app.py                        # Main Streamlit app
├── requirements.txt              # Python dependencies
├── data/
│   └── phfphandbook-compressed.pdf   # Knowledge base PDF
└── .streamlit/
    ├── secrets.toml              # API keys (DO NOT commit to GitHub)
    └── config.toml               # Streamlit theme and server config
```

---

## Deployment: Streamlit Community Cloud

### Step 1 — Push to GitHub
1. Create a new GitHub repository (public or private)
2. Upload all files **keeping the folder structure above**
3. Make sure `data/phfphandbook-compressed.pdf` is included
4. **Do NOT upload `.streamlit/secrets.toml`** — add it to `.gitignore`

```
# .gitignore
.streamlit/secrets.toml
__pycache__/
*.pyc
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select your GitHub repo and set **Main file path** to `app.py`
4. Click **Deploy**

### Step 3 — Add Secrets
In the Streamlit Cloud dashboard:
1. Open your deployed app → **Settings** → **Secrets**
2. Add the following:

```toml
HF_TOKEN = "hf_your_token_here"
GROQ_API_KEY = "gsk_your_key_here"
```

#### Getting your API keys:
- **HF_TOKEN**: Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token → Read access
- **GROQ_API_KEY**: Go to [console.groq.com](https://console.groq.com) → API Keys → Create new key

---

## How It Works

| Component | What it does |
|---|---|
| `pymupdf4llm` | Extracts PDF as Markdown, preserving structure |
| `RecursiveCharacterTextSplitter` | Splits into 4000-char chunks with 1000-char overlap |
| `nomic-embed-text-v1.5` | Embeds chunks for semantic (FAISS) search |
| `BM25Retriever` | Keyword-based sparse retrieval |
| `EnsembleRetriever` | Combines BM25 (60%) + FAISS (40%) |
| `bge-reranker-base` | Cross-encoder reranks top results |
| `Qwen3-4B` (HF API) | Generates grounded answers |
| Context expansion | Includes neighboring chunks for continuity |

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure `.streamlit/secrets.toml` has your API keys filled in.

---

## Notes
- First load takes ~2 minutes (builds FAISS index + loads reranker)
- After first load, everything is cached — responses are fast
- The app runs on CPU only (no GPU required for deployment)
