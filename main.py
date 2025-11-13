from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils import load_messages
from retriever import build_index, detect_user_name, retrieve_relevant_messages
from llm import generate_answer
import os
from datetime import datetime
import time  # 🕒 for performance timing

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
# Startup event
# ──────────────────────────────
@app.on_event("startup")
def startup_event():
    """Load cached messages and build embeddings (only if missing or outdated)."""
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
# /ask endpoint (with timing)
# ──────────────────────────────
@app.get("/ask")
def ask(question: str = Query(..., description="Natural-language question to answer")):
    """Receives a question and returns an LLM-generated, context-grounded answer."""
    start_total = time.perf_counter()

    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        print(f"\n🧩 Received question: {question}")
        t0 = time.perf_counter()

        # Step 1: Detect which member is being referenced
        user_name = detect_user_name(question, user_names)
        t1 = time.perf_counter()
        print(f"⏱️ User detection took {t1 - t0:.3f}s")

        # Step 2: Retrieve relevant messages (semantic + hybrid logic)
        context = retrieve_relevant_messages(question, top_k=5, user_name=user_name)
        t2 = time.perf_counter()
        print(f"⏱️ Context retrieval took {t2 - t1:.3f}s")

        if not context:
            print("⚠️ No context found — skipping LLM.")
            total_time = time.perf_counter() - start_total
            print(f"⏱️ Total processing time: {total_time:.3f}s\n")
            return {
                "question": question,
                "detected_user": user_name,
                "answer": "I don’t have enough information to answer that.",
                "context_used": [],
                "processing_time_sec": round(total_time, 3)
            }

        # Step 3: Generate answer via LLM
        answer = generate_answer(question, context)
        t3 = time.perf_counter()
        print(f"⏱️ LLM generation took {t3 - t2:.3f}s")

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

        total_time = time.perf_counter() - start_total
        print(f"✅ Answer generated successfully in {total_time:.3f}s for '{question}'\n")

        return {
            "question": question,
            "detected_user": user_name,
            "answer": answer,
            "context_used": formatted_context,
            "processing_time_sec": round(total_time, 3)
        }

    except Exception as e:
        total_time = time.perf_counter() - start_total
        print(f"❌ Error in /ask after {total_time:.3f}s: {e}")
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