"""
PERSONAL STUDY ASSISTANT — WITH YOUR OWN NOTES (RAG version)
------------------------------------------------------------------
This version searches YOUR stats notes before answering, instead of
just using general knowledge. This is what makes it more useful than
plain ChatGPT for your specific coursework.

IMPORTANT: Before running this for the first time, run:
   python build_knowledge_base.py
(This only needs to be done once, or whenever you add new files.)

SETUP:
   pip install google-genai chromadb pypdf

RUN:
   python study_assistant.py
"""

import os
import chromadb
from google import genai
from google.genai import types

# ---- SETUP ----
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

DB_FOLDER = "knowledge_base"
chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
collection = chroma_client.get_or_create_collection(name="stats_notes")

# ---- PERSONALITY ----
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

# Create a chat session (keeps conversation memory automatically)
chat = client.chats.create(
    model="gemini-flash-latest",
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
    ),
)


def search_notes(question, n_results=3):
    """
    Searches the knowledge base for chunks most relevant to the question.
    Returns the combined text of the top matches, plus which files they came from.
    """
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
    )

    if not results["documents"] or not results["documents"][0]:
        return None, []

    chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]

    combined_text = "\n\n---\n\n".join(chunks)
    unique_sources = list(set(sources))

    return combined_text, unique_sources


def get_bot_response(user_message):
    # 1. Search the notes for relevant context
    context, sources = search_notes(user_message)

    # 2. Build the actual message we send to the AI —
    #    we attach the relevant notes BEFORE the user's question
    if context:
        full_prompt = (
            f"Relevant excerpts from Taseer's Stats notes:\n\n{context}\n\n"
            f"---\n\nTaseer's question: {user_message}"
        )
    else:
        full_prompt = user_message

    # 3. Send it to the model
    response = chat.send_message(full_prompt)

    return response.text, sources


def main():
    print("=" * 55)
    print("  📚 Stats Study Assistant ready! Type 'quit' to exit.")
    print("  Ask anything about your stats coursework.")
    print("=" * 55)

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["quit", "exit"]:
            print("\nBot: Good session! Keep going 💪")
            break

        try:
            reply, sources = get_bot_response(user_input)
            print(f"\nBot: {reply}")
            if sources:
                print(f"\n   📄 (Based on: {', '.join(sources)})")
        except Exception as e:
            print(f"\n[Error] {e}")


if __name__ == "__main__":
    main()
