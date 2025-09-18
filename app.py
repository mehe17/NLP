import streamlit as st
import os
from openai import OpenAI

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
- This demo uses a simple LLM wrapper for FAQ/chat.
- LLM modes:
  - **HF**: Hugging Face Inference API. Token is stored in Streamlit Secrets.
  - **Mock**: Fast local fallback (rule-based).
"""
    )

# -----------------------
# Sidebar configuration
# -----------------------
st.sidebar.header("Configuration")
mode = st.sidebar.selectbox("LLM Mode", ["hf", "mock"])
hf_model = st.sidebar.text_input("HF Model (optional)", value="moonshotai/Kimi-K2-Instruct")
os.environ["HF_MODEL"] = hf_model  # override model if needed

# -----------------------
# Initialize Hugging Face client (online LLM)
# -----------------------
if mode == "hf":
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]  # load token from Secrets
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )

# -----------------------
# Initialize session state
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------
# Mock answer function
# -----------------------
def mock_answer(query, order_id=None):
    q = query.lower()
    if "cancel" in q:
        msg = (
            "You can cancel orders within 5 minutes after placing the order "
            "if the restaurant hasn't confirmed. If the restaurant accepted it, "
            "cancellation may not be possible or may incur a fee."
        )
    elif "refund" in q:
        msg = (
            "Refunds are available for missing or incorrect items and non-delivery. "
            "Contact Memo Hero Delivery support within 24 hours of delivery."
        )
    elif "track" in q or "status" in q:
        msg = "You can track your order using the tracking number sent to your email or app."
    elif "delivery time" in q or "delivery times" in q:
        msg = (
            "Typical delivery times range from 20 to 45 minutes depending on the restaurant and location."
        )
    elif "missing item" in q or "item missing" in q:
        msg = (
            "If an item is missing from your order, please contact Memo Hero Delivery support "
            "with your order id and the missing item details."
        )
    else:
        msg = "I'm sorry, I didn't understand that. Can you rephrase or ask about cancellations, refunds, tracking, or delivery times?"

    if order_id:
        msg += f" (Order ID: {order_id})"
    return msg

# -----------------------
# Function to query online LLM
# -----------------------
def answer_query(query, order_id=None):
    if mode == "mock":
        return mock_answer(query, order_id)

    messages = [{"role": "user", "content": query}]
    if order_id:
        messages.append({"role": "system", "content": f"Order ID: {order_id}"})

    # Call Hugging Face LLM via OpenAI-compatible API
    completion = client.chat.completions.create(
        model=os.environ["HF_MODEL"],
        messages=messages
    )
    return completion.choices[0].message

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
- What are the delivery times?
- How do I track my order?
"""
        )
    st.markdown("---")
    st.write("Provide an order id to allow order-specific lookups.")
    order_id = st.text_input("Order ID (optional)", value="21345")

# Right column: Chat interface
with col2:
    st.header("Chat")
    query = st.text_input(
        "Ask the Memo Hero Delivery bot anything about orders, cancellations, refunds, delivery times, tracking, or missing items...",
        key="query_input"
    )
    submit = st.button("Send")

    if submit and query:
        with st.spinner("Thinking..."):
            resp = answer_query(query, order_id=order_id if order_id else None)
        st.session_state.history.append({"user": query, "bot": resp})

    # Keep only last 20 messages
    st.session_state.history = st.session_state.history[-20:]

    # Display chat history
    for turn in reversed(st.session_state.history):
        st.markdown(f"**You:** {turn['user']}")
        st.markdown(f"**Bot:** {turn['bot']}")
        st.markdown("---")
