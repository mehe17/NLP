"""
chatbot.py
- LLM wrapper (supports HF Inference API or a mock fallback)
- Assemble RAG prompt (retrieve docs + order lookup)
- Expose `answer_query(user_query, order_id=None, mode='hf')`
"""

import os
import requests
import pandas as pd
from pathlib import Path
from retriever import get_relevant_docs

DATA_DIR = Path("data")
ORDERS_PATH = DATA_DIR / "orders.csv"

# Basic prompt template
PROMPT_TEMPLATE = """You are a helpful customer support assistant for Memo Hero Delivery.
Use the policy documents and the user's order data (if provided) to answer succinctly and politely.
Cite (briefly) when you use policy text.

Policy excerpts:
{policy_texts}

Order info:
{order_text}

User question:
{user_question}

Answer in 2-4 concise sentences. If you don't have enough info, ask for the order id or more details.
"""

def load_order(order_id):
    if not ORDERS_PATH.exists():
        return None
    df = pd.read_csv(ORDERS_PATH, dtype=str)
    row = df[df['order_id'] == str(order_id)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()

def _format_order_text(order):
    if not order:
        return "No order provided."
    pieces = [f"{k}: {v}" for k,v in order.items() if pd.notna(v)]
    return "\n".join(pieces)

class LLMClient:
    def __init__(self, mode="hf"):
        """
        mode: 'hf' for Hugging Face Inference API, 'mock' for fallback rule-based
        If mode == 'hf', requires env var HF_API_TOKEN
        """
        self.mode = mode
        self.hf_token = os.environ.get("HF_API_TOKEN")

    def generate(self, prompt, max_length=512):
        if self.mode == "hf" and self.hf_token:
            return self._call_hf(prompt, max_length)
        else:
            return self._mock_generate(prompt)

    def _call_hf(self, prompt, max_length):
        model = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0")
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "temperature": 0.2}}
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            return f"(HF API error {resp.status_code}) {resp.text}"
        data = resp.json()
        if isinstance(data, list) and 'generated_text' in data[0]:
            return data[0]['generated_text']
        if isinstance(data, dict) and 'generated_text' in data:
            return data['generated_text']
        return str(data)

    def _mock_generate(self, prompt):
        text = prompt.lower()
        if "cancel" in text or "cancellation" in text:
            return "You can cancel orders within 5 minutes if the restaurant hasn't confirmed. If already accepted, cancellation may not be possible."
        if "refund" in text:
            return "Refunds are available for missing or incorrect items and non-delivery. Please provide your order id to check further."
        if "where is" in text or "status" in text:
            return "Please provide your order id so I can look up the status."
        return "Thanks for your question — could you share the order id or clarify your request?"

def answer_query(user_query, order_id=None, llm_mode="hf"):
    snippets = get_relevant_docs(user_query, k=3)
    policy_texts = "\n\n".join(snippets)

    order = None
    order_text = "No order provided."
    if order_id:
        order = load_order(order_id)
        if order:
            order_text = _format_order_text(order)
        else:
            order_text = f"No order found with id {order_id}."

    prompt = PROMPT_TEMPLATE.format(
        policy_texts=policy_texts,
        order_text=order_text,
        user_question=user_query
    )

    client = LLMClient(mode=llm_mode)
    resp = client.generate(prompt)
    return resp.strip()
