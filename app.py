"""
UNIBUDDY AI — WEB VERSION (Streamlit)
--------------------------------------------------
A colorful, modern-looking version of the stats study assistant.
RAG still runs behind the scenes (the bot still reads your notes to answer),
but source file names are no longer shown to keep the chat clean.

SETUP:
   pip install streamlit

RUN:
   streamlit run app.py
"""

import os
import time
import streamlit as st
import chromadb
from google import genai
from google.genai import types

# ---- PAGE SETUP ----
st.set_page_config(page_title="UniBuddy AI", page_icon="🎓", layout="centered")

# ---- CUSTOM STYLING ----
st.markdown("""
<style>
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
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.5rem 0.25rem;
    }
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

# ---- SETUP ----
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not set. Add it in your app's Secrets settings.")
    st.stop()

DB_FOLDER = "knowledge_base"
chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
collection = chroma_client.get_or_create_collection(name="stats_notes")


def build_system_prompt(student_name):
    return f"""You are UniBuddy, a patient, encouraging study assistant for
{student_name}, a university student in Lahore studying Artificial Intelligence.

You will be given relevant excerpts from the Statistics course notes before
each question. Use these excerpts as your PRIMARY source of truth when they
are relevant to the question. If the excerpts don't contain the answer,
say so honestly, then you may answer from general knowledge instead.

Your style:
- Explain things simply, like to a beginner.
- Use short examples where helpful.
- If asked to quiz {student_name}, ask ONE question at a time and wait for
  their answer.
- Keep responses focused, not too long.
- Be encouraging.
- Address {student_name} by name occasionally, naturally (not every message).
- Never mention file names, slide numbers, or "your notes/slides" explicitly
  in a way that sounds like a citation — just answer naturally as if you
  simply know the material.
"""


def search_notes(question, n_results=3):
    results = collection.query(query_texts=[question], n_results=n_results)
    if not results["documents"] or not results["documents"][0]:
        return None
    chunks = results["documents"][0]
    return "\n\n---\n\n".join(chunks)


def get_ai_response(full_prompt, student_name):
    """
    Creates a FRESH client and rebuilds the full conversation history on
    every call, instead of reusing a long-lived chat object. This avoids
    the 'client has been closed' crash that happens when Streamlit reruns
    the script between messages.
    """
    client = genai.Client(api_key=GOOGLE_API_KEY)

    contents = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part(text=msg["content"])])
        )
    contents.append(
        types.Content(role="user", parts=[types.Part(text=full_prompt)])
    )

    # Try a couple of model names in case one is temporarily overloaded,
    # and retry briefly before giving up.
    models_to_try = ["gemini-flash-latest", "gemini-2.5-flash"]
    last_error = None

    for model_name in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=build_system_prompt(student_name)
                    ),
                )
                return response.text
            except Exception as e:
                last_error = e
                time.sleep(4)

    raise last_error


# ---- STEP 1: ASK FOR NAME ----
if "user_name" not in st.session_state:
    st.session_state.user_name = None

if st.session_state.user_name is None:
    st.markdown("### 👋 Welcome! What should I call you?")
    name_input = st.text_input("Your name", placeholder="e.g. Ahmed, Sara...", label_visibility="collapsed")
    if st.button("Start studying →") and name_input.strip():
        st.session_state.user_name = name_input.strip()
        st.rerun()
    st.stop()

# ---- STEP 2: MESSAGE HISTORY ----
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info(f"👋 Hey {st.session_state.user_name}! Try asking: *\"What is measures of dispersion?\"* or *\"Quiz me on skewness and kurtosis\"*")

for msg in st.session_state.messages:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ---- CHAT INPUT ----
user_input = st.chat_input("Ask about your stats coursework...")

if user_input:
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_input)

    context = search_notes(user_input)
    if context:
        full_prompt = (
            f"Relevant excerpts from the Stats notes:\n\n{context}\n\n"
            f"---\n\n{st.session_state.user_name}'s question: {user_input}"
        )
    else:
        full_prompt = user_input

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Thinking..."):
            try:
                reply_text = get_ai_response(full_prompt, st.session_state.user_name)
            except Exception as e:
                reply_text = f"Sorry, something went wrong: {e}"
            st.markdown(reply_text)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": reply_text})
