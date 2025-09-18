"""
retriever.py
- Builds embeddings from support_docs.txt
- Creates / loads a FAISS index
- Provides function `get_relevant_docs(query, k=3)`
"""

import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from pathlib import Path

DATA_DIR = Path("data")
DOCS_PATH = DATA_DIR / "support_docs.txt"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
METADATA_PATH = DATA_DIR / "faiss_meta.json"
EMB_DIM = 384  # for all-MiniLM-L6-v2

model = SentenceTransformer(EMB_MODEL_NAME)

def _read_docs():
    text = DOCS_PATH.read_text(encoding="utf-8")
    # split into paragraphs by blank line or newline headings, keep simple
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    # as fallback, split by line if paragraphs small
    if len(paragraphs) < 2:
        paragraphs = [line.strip() for line in text.split("\n") if line.strip()]
    return paragraphs

def build_index(force_rebuild=False):
    DATA_DIR.mkdir(exist_ok=True)
    if INDEX_PATH.exists() and METADATA_PATH.exists() and not force_rebuild:
        print("Index and metadata found — skipping rebuild.")
        return

    docs = _read_docs()
    embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=True)
    index = faiss.IndexFlatL2(EMB_DIM)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(INDEX_PATH))

    meta = {"docs": docs}
    METADATA_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built index with {len(docs)} documents.")

def load_index():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        build_index()
    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return index, meta

def get_relevant_docs(query, k=3):
    index, meta = load_index()
    docs = meta["docs"]
    q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        if idx < len(docs):
            results.append(docs[idx])
    return results

