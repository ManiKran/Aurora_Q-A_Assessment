from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils import load_messages
from retriever import build_index, detect_user_name, retrieve_relevant_messages
from llm import generate_answer
import os
from datetime import datetime

app = FastAPI(
    title="Aurora Member Q&A API",
    description="A high-accuracy RAG API that answers questions about members using their message history.",
    version="2.1.0"
)

# ──────────────────────────────
# Global cache
# ──────────────────────────────
messages = []
user_names = []

# ──────────────────────────────
# CORS setup
# ──────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Replace '*' with your frontend domain when deployed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────
# Startup event — smart caching & embedding setup
# ──────────────────────────────
@app.on_event("startup")
def startup_event():
    """
    Load cached messages and build embeddings (only if missing or outdated).
    """
    global messages, user_names

    print("🚀 Starting Aurora Q&A backend...")

    try:
        messages = load_messages()
        user_names = list({m["user_name"] for m in messages})
        print(f"📋 Loaded {len(messages)} messages from {len(user_names)} members.")

        chroma_path = "chroma_store"
        has_index = os.path.exists(chroma_path) and any(os.scandir(chroma_path))

        if not has_index:
            print("🧠 No existing embeddings found — building index...")
            build_index(messages)
        else:
            print("💾 Found existing Chroma index — skipping rebuild.")

        print("✅ Aurora Q&A API ready at /ask")

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

# ──────────────────────────────
# /ask endpoint
# ──────────────────────────────
@app.get("/ask")
def ask(question: str = Query(..., description="Natural-language question to answer")):
    """
    Receives a question and returns an LLM-generated, context-grounded answer.
    """
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        print(f"🧩 Received question: {question}")

        # Step 1: Detect which member is being referenced
        user_name = detect_user_name(question, user_names)

        # Step 2: Retrieve relevant messages (semantic + hybrid logic)
        context = retrieve_relevant_messages(question, top_k=5, user_name=user_name)

        if not context:
            return {
                "question": question,
                "detected_user": user_name,
                "answer": "I don’t have enough information to answer that.",
                "context_used": [],
            }

        # Step 3: Generate answer via LLM
        answer = generate_answer(question, context)

        # Format timestamps for response
        formatted_context = [
            {
                "user_name": c.get("user_name"),
                "text": c.get("text"),
                "timestamp": (
                    c["timestamp"].isoformat()
                    if c.get("timestamp")
                    and not isinstance(c["timestamp"], str)
                    else c.get("timestamp")
                ),
            }
            for c in context
        ]

        print(f"✅ Answer generated successfully for '{question}'.")

        return {
            "question": question,
            "detected_user": user_name,
            "answer": answer,
            "context_used": formatted_context,
        }

    except Exception as e:
        print(f"❌ Error in /ask: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing request.")

# ──────────────────────────────
# /health endpoint
# ──────────────────────────────
@app.get("/health")
def health():
    """Simple health check route for uptime monitoring."""
    return {
        "status": "ok",
        "messages_loaded": len(messages),
        "users": len(user_names),
        "timestamp": datetime.utcnow().isoformat(),
    }

# ──────────────────────────────
# Root
# ──────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Welcome to Aurora Member Q&A API!",
        "usage": "Try /ask?question=Your+Question",
        "version": "2.1.0",
    }