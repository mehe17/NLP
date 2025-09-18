import streamlit as st
from chatbot import answer_query
from retriever import build_index
import os

# -----------------------
# Page configuration
# -----------------------
st.set_page_config(page_title="Memo Hero Delivery — Support Chatbot (Demo)", layout="wide")
st.title("🟠 Memo Hero Delivery — LLM Support Chatbot (Demo)")

# -----------------------
# Setup & Notes
# -----------------------
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

# -----------------------
# Sidebar configuration
# -----------------------
st.sidebar.header("Configuration")
mode = st.sidebar.selectbox("LLM Mode", ["hf", "mock"])
hf_token_input = st.sidebar.text_input("Hugging Face Token (optional)", type="password")
hf_model = st.sidebar.text_input("HF Model (optional)", value="mistralai/Mistral-7B-Instruct-v0")

if hf_token_input:
    os.environ["HF_API_TOKEN"] = hf_token_input
if hf_model:
    os.environ["HF_MODEL"] = hf_model

# Build RAG index
if st.sidebar.button("(Re)build RAG index"):
    with st.spinner("Building index..."):
        build_index(force_rebuild=True)
    st.success("Index built.")

# -----------------------
# Initialize session state
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# -----------------------
# Mock answer function
# -----------------------
def mock_answer(query):
    q = query.lower()
    if "cancel" in q:
        return "You can cancel orders within 5 minutes after placing the order if the restaurant hasn't confirmed. If the restaurant accepted it, cancellation may not be possible or may incur a fee. Please provide your order id to check further."
    elif "refund" in q:
        return "Refunds are available for missing or incorrect items and non-delivery. Contact Memo Hero Delivery support within 24 hours of delivery with your order id and details. Refund processing can take up to 7 business days."
    elif "track" in q or "status" in q:
        return "You can track your order using the tracking number sent to your email or app."
    else:
        return "I'm sorry, I didn't understand that. Can you rephrase?"

# -----------------------
# Layout: Columns
# -----------------------
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

# Right column: Chat interface
with col2:
    st.header("Chat")
    query = st.text_input(
        "Ask the Memo Hero Delivery bot anything about orders, cancellations, refunds, delivery times...",
        key="query_input",
        value=st.session_state.query_input
    )
    submit = st.button("Send")

    if submit and query:
        with st.spinner("Thinking..."):
            if mode == "mock":
                resp = mock_answer(query)
            else:
                resp = answer_query(query, order_id=order_id if order_id else None, llm_mode=mode)
        # Append to history
        st.session_state.history.append({"user": query, "bot": resp})
        st.session_state.query_input = ""  # clear input after send

    # Keep only last 20 messages
    st.session_state.history = st.session_state.history[-20:]

    # Display chat history
    for turn in reversed(st.session_state.history):
        st.markdown(f"**You:** {turn['user']}")
        st.markdown(f"**Bot:** {turn['bot']}")
        st.markdown("---")
