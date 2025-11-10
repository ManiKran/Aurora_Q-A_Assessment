# retriever.py
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rapidfuzz import process, fuzz
import re
import unicodedata

# ──────────────────────────────
# Initialize components
# ──────────────────────────────
model = SentenceTransformer("all-MiniLM-L12-v2")

# create persistent Chroma DB folder (so it’s saved between runs)
CHROMA_PATH = "chroma_store"
os.makedirs(CHROMA_PATH, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="member_messages",
    metadata={"hnsw:space": "cosine"},
)

# ──────────────────────────────
# Build the vector index
# ──────────────────────────────
def build_index(messages):
    """
    Create embeddings and store them in Chroma.
    """
    print("🔄 Building embedding index...")
    texts = [m["message"] for m in messages]
    ids = [m["id"] for m in messages]
    metas = [{"user_name": m["user_name"], "user_id": m["user_id"]} for m in messages]

    # avoid duplicates if running again
    if collection.count() > 0:
        print("🧹 Clearing old embeddings from the collection...")
        # delete all documents regardless of filter
        try:
            collection.delete(where={"user_name": {"$ne": None}})
        except Exception as e:
            print("⚠️ Warning while clearing collection:", e)

    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metas)
    print(f"✅ Indexed {len(texts)} messages.")


def normalize_text(text: str):
    """Normalize text for matching: remove accents, lowercase, and replace curly quotes."""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = re.sub(r"[^a-zA-Z' ]+", " ", text)  # keep only letters, spaces, and apostrophes
    return text.lower().strip()

# ──────────────────────────────
# Fuzzy user name detection
# ──────────────────────────────
def detect_user_name(question, all_user_names):
    """
    Detects which user is mentioned in the question.
    Uses fuzzy matching that handles single first names too.
    """
    normalized_q = normalize_text(question)
    normalized_users = {normalize_text(n): n for n in all_user_names}

    best_match = None
    best_score = 0

    # check both full names and each first name separately
    for norm_name, original in normalized_users.items():
        parts = norm_name.split()
        for part in parts + [norm_name]:
            score = fuzz.partial_ratio(part, normalized_q)
            if score > best_score:
                best_score = score
                best_match = original

    if best_match and best_score >= 55:
        print(f"🧭 Detected user: {best_match} (score: {best_score})")
        return best_match

    print("⚠️ No user detected.")
    return None


# ──────────────────────────────
# Retrieve relevant messages
# ──────────────────────────────
def retrieve_relevant_messages(question, top_k=5, user_name=None):
    """
    Retrieve top-k relevant messages based on the question.
    Optionally filter by user name.
    """
    query_text = question
    if user_name:
        query_text = f"{user_name}: {question}"
    q_emb = model.encode(query_text).tolist()
    query_args = {"query_embeddings": [q_emb], "n_results": top_k}

    if user_name:
        query_args["where"] = {"user_name": user_name}

    results = collection.query(**query_args)

    if not results["documents"]:
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    return [{"text": d, "user_name": m["user_name"], "user_id": m["user_id"]} for d, m in zip(docs, metas)]