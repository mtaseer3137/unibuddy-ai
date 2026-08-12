"""
UNIBUDDY AI — WEB VERSION (Streamlit)
--------------------------------------------------
A colorful, modern-looking version of the stats study assistant.

SETUP:
   pip install streamlit

RUN:
   streamlit run app.py
"""

import os
import streamlit as st
import chromadb
from google import genai
from google.genai import types

# ---- PAGE SETUP ----
st.set_page_config(page_title="UniBuddy AI", page_icon="🎓", layout="centered")

# ---- CUSTOM STYLING ----
st.markdown("""
<style>
    /* Gradient header banner */
    .hero-banner {
        background: linear-gradient(135deg, #6C5CE7 0%, #A29BFE 50%, #74B9FF 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(108, 92, 231, 0.35);
    }
    .hero-banner h1 {
        color: white;
        font-size: 2.4rem;
        margin: 0;
        font-weight: 800;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.9);
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Chat message bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.5rem 0.25rem;
    }

    /* Source caption pill */
    .source-pill {
        display: inline-block;
        background: linear-gradient(135deg, #74B9FF, #A29BFE);
        color: white;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Chat input styling */
    [data-testid="stChatInput"] {
        border-radius: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ---- HERO BANNER ----
st.markdown("""
<div class="hero-banner">
    <h1>🎓 UniBuddy AI</h1>
    <p>Your personal study companion — built by Taseer</p>
</div>
""", unsafe_allow_html=True)

# ---- SETUP (runs once per session) ----
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not set. Add it in your app's Secrets settings.")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)

DB_FOLDER = "knowledge_base"
chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
collection = chroma_client.get_or_create_collection(name="stats_notes")

SYSTEM_PROMPT = """You are UniBuddy, a patient, encouraging study assistant for
Taseer, a university student in Lahore studying Artificial Intelligence.

You will be given relevant excerpts from his Statistics course notes before
each question. Use these excerpts as your PRIMARY source of truth when they
are relevant to the question. If the excerpts don't contain the answer,
say so honestly, then you may answer from general knowledge instead.

Your style:
- Explain things simply, like to a beginner.
- Use short examples where helpful.
- If asked to quiz him, ask ONE question at a time and wait for his answer.
- Keep responses focused, not too long.
- Be encouraging.
"""


def search_notes(question, n_results=3):
    results = collection.query(query_texts=[question], n_results=n_results)
    if not results["documents"] or not results["documents"][0]:
        return None, []
    chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]
    combined_text = "\n\n---\n\n".join(chunks)
    unique_sources = list(set(sources))
    return combined_text, unique_sources


# ---- SESSION STATE ----
if "chat" not in st.session_state:
    st.session_state.chat = client.chats.create(
        model="gemini-flash-latest",
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- DISPLAY CHAT HISTORY ----
if not st.session_state.messages:
    st.info("👋 Try asking: *\"What is measures of dispersion?\"* or *\"Quiz me on skewness and kurtosis\"*")

for msg in st.session_state.messages:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            pills = " ".join(
                f'<span class="source-pill">📄 {s}</span>' for s in msg["sources"]
            )
            st.markdown(pills, unsafe_allow_html=True)

# ---- CHAT INPUT ----
user_input = st.chat_input("Ask about your stats coursework...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_input)

    context, sources = search_notes(user_input)
    if context:
        full_prompt = (
            f"Relevant excerpts from Taseer's Stats notes:\n\n{context}\n\n"
            f"---\n\nTaseer's question: {user_input}"
        )
    else:
        full_prompt = user_input

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat.send_message(full_prompt)
            st.markdown(response.text)
            if sources:
                pills = " ".join(
                    f'<span class="source-pill">📄 {s}</span>' for s in sources
                )
                st.markdown(pills, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.text,
        "sources": sources,
    })
