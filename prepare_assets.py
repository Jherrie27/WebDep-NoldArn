import os
import json
import faiss
import pickle

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import fitz  # PyMuPDF

DATA_DIR = "data"
PDF_PATH = os.path.join(DATA_DIR, "phfphandbook-compressed.pdf")

print("Loading PDF...")

doc = fitz.open(PDF_PATH)
text = ""

for page in doc:
    text += page.get_text()

print("Splitting into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))

with open(os.path.join(DATA_DIR, "chunks.json"), "w", encoding="utf-8") as f:
    json.dump(chunks, f)

print("Generating embeddings...")


model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

embeddings = model.encode(chunks, show_progress_bar=True)

print("Building FAISS index...")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, os.path.join(DATA_DIR, "faiss.index"))

print("Building BM25 index...")

tokenized_chunks = [chunk.split(" ") for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

with open(os.path.join(DATA_DIR, "bm25.pkl"), "wb") as f:
    pickle.dump(bm25, f)

print("Assets generated successfully!")