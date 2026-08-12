"""
STATS STUDY ASSISTANT — WEB VERSION (Streamlit)
--------------------------------------------------
This turns your terminal chatbot into a website you (and eventually
your classmates) can use in a browser.

SETUP:
   pip install streamlit

RUN (note: this is NOT "python app.py" — it's a special command):
   streamlit run app.py

This will open a browser tab automatically at http://localhost:8501
"""

import os
import streamlit as st
import chromadb
from google import genai
from google.genai import types

# ---- PAGE SETUP ----
st.set_page_config(page_title="Stats Study Assistant", page_icon="📊")
st.title("📊 Stats Study Assistant")
st.caption("Ask anything about your stats coursework — built by Taseer")

# ---- SETUP (runs once per session) ----
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not set. Set it in your terminal before running this app.")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)

DB_FOLDER = "knowledge_base"
chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
collection = chroma_client.get_or_create_collection(name="stats_notes")

SYSTEM_PROMPT = """You are a patient, encouraging study assistant for Taseer,
a university student in Lahore studying Artificial Intelligence.

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
# Streamlit re-runs the whole script on every interaction, so we need to
# store the chat session and message history in st.session_state to
# persist them between reruns (otherwise memory would reset every message!)
if "chat" not in st.session_state:
    st.session_state.chat = client.chats.create(
        model="gemini-flash-latest",
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- DISPLAY CHAT HISTORY ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"📄 Based on: {', '.join(msg['sources'])}")

# ---- CHAT INPUT ----
user_input = st.chat_input("Ask about your stats coursework...")

if user_input:
    # Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Search notes + build the prompt
    context, sources = search_notes(user_input)
    if context:
        full_prompt = (
            f"Relevant excerpts from Taseer's Stats notes:\n\n{context}\n\n"
            f"---\n\nTaseer's question: {user_input}"
        )
    else:
        full_prompt = user_input

    # Get the bot's reply and show it
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat.send_message(full_prompt)
            st.markdown(response.text)
            if sources:
                st.caption(f"📄 Based on: {', '.join(sources)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.text,
        "sources": sources,
    })
