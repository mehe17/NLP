import streamlit as st
from chatbot import answer_query
from retriever import build_index
import os

# Page config
st.set_page_config(page_title="Delivery Hero — Support Chatbot (Demo)", layout="wide")

# Title
st.title("🟠 Delivery Hero — LLM Support Chatbot (Demo)")

# Setup & Notes
with st.expander("Setup & Notes"):
    st.markdown(
        """
- This demo uses **RAG** (FAISS + SentenceTransformers) to retrieve policy excerpts and a simple LLM wrapper.
- LLM modes:
  - **HF**: Use Hugging Face Inference API. Set `HF_API_TOKEN` in your environment (or enter below).
  - **Mock**: Fast local fallback (rule-based) for demos without any API keys.
- If you want to use a local model (e.g., Ollama), you can swap the `LLMClient._call_hf` implementation to call your local endpoint.
"""
    )

# Sidebar configuration
st.sidebar.header("Configuration")
mode = st.sidebar.selectbox("LLM Mode", ["hf", "mock"])
hf_token_input = st.sidebar.text_input("Hugging Face Token (optional)", type="password")
hf_model = st.sidebar.text_input("HF Model (optional)", value="mistralai/Mistral-7B-Instruct-v0")

if hf_token_input:
    os.environ["HF_API_TOKEN"] = hf_token_input
if hf_model:
    os.environ["HF_MODEL"] = hf_model

# Build RAG index button
if st.sidebar.button("(Re)build RAG index"):
    with st.spinner("Building index..."):
        build_index(force_rebuild=True)
    st.success("Index built.")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

if "rerun_flag" not in st.session_state:
    st.session_state.rerun_flag = False

# Layout
col1, col2 = st.columns([1, 2])

# Left column: Quick actions
with col1:
    st.header("Quick actions")
    if st.button("Show sample questions"):
        st.write(
            """
- Can I cancel my order?
- How do I request a refund?
- Where is order #21345?
- My order is missing an item — what should I do?
"""
        )

    st.markdown("---")
    st.write("Provide an order id to allow order-specific lookups.")
    order_id = st.text_input("Order ID (optional)", value="21345")

# Right column: Chat
with col2:
    st.header("Chat")
    query = st.text_input(
        "Ask the bot anything about orders, cancellations, refunds, delivery times...",
        key="query_input"
    )
    submit = st.button("Send")

    if submit and query:
        with st.spinner("Thinking..."):
            resp = answer_query(
                query,
                order_id=order_id if order_id else None,
                llm_mode=mode
            )
        st.session_state.history.append({"user": query, "bot": resp})
        st.session_state.rerun_flag = True  # safe rerun flag

    # Safe rerun
    if st.session_state.rerun_flag:
        st.session_state.rerun_flag = False
        st.experimental_rerun()

    # Display last 10 chat turns
    for i, turn in enumerate(reversed(st.session_state.history[-10:])):
        st.markdown(f"**You:** {turn['user']}")
        st.markdown(f"**Bot:** {turn['bot']}")
        st.markdown("---")
