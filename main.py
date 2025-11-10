# main.py
from fastapi import FastAPI, Query
from utils import load_messages
from retriever import build_index, detect_user_name, retrieve_relevant_messages
from llm import generate_answer
from typing import Optional

app = FastAPI(
    title="Member Question Answering API",
    description="A simple RAG-based API that answers questions about members using message data.",
    version="1.0.0"
)

# ──────────────────────────────
# Global message cache
# ──────────────────────────────
messages = []
user_names = []

@app.on_event("startup")
def startup_event():
    """
    Load data and build embeddings when the server starts.
    """
    global messages, user_names
    print("🚀 Loading messages...")
    messages = load_messages()  # from cache if exists
    user_names = list({m["user_name"] for m in messages})
    build_index(messages)
    print("✅ Ready! API is live.")


# ──────────────────────────────
# /ask endpoint
# ──────────────────────────────
@app.get("/ask")
def ask(question: str = Query(..., description="Natural-language question to answer")):
    """
    Main endpoint: receives a question and returns an answer.
    """
    # Step 1: detect which user (if any)
    user_name = detect_user_name(question, user_names)

    # Step 2: retrieve top relevant messages
    context = retrieve_relevant_messages(question, top_k=5, user_name=user_name)

    # Step 3: generate answer with LLM
    answer = generate_answer(question, context)

    return {
        "question": question,
        "detected_user": user_name,
        "answer": answer,
        "context_used": [c["text"] for c in context]  # optional for debugging
    }


@app.get("/")
def root():
    return {"message": "Welcome to the Member Q&A API! Use /ask?question=Your+Question"}