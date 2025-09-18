# AI Customer Support Chatbot — Delivery Hero (Demo)

A small demo showing how to build a retrieval-augmented LLM chatbot for delivery support queries.

## Features
- Retrieval-Augmented Generation (RAG) with SentenceTransformers + FAISS
- Simple LLM wrapper: use Hugging Face Inference API (recommended) or a mock fallback
- Streamlit UI for quick demoing
- Mock `orders.csv` to simulate order lookups

## Quick start (local)
1. Create virtualenv and install:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
