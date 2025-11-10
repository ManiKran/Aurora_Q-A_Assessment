import os
import re
import unicodedata
import numpy as np
from tqdm import tqdm
from datetime import datetime, timezone
from dateutil import parser as date_parser
from sentence_transformers import SentenceTransformer
import chromadb
from rapidfuzz import fuzz

# ──────────────────────────────
# Configuration
# ──────────────────────────────
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 🔥 High-accuracy model
CHROMA_PATH = "chroma_store"
os.makedirs(CHROMA_PATH, exist_ok=True)

print(f"🧠 Loading embedding model: {EMBED_MODEL}")
model = SentenceTransformer(EMBED_MODEL)

# Initialize persistent Chroma client
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="member_messages",
    metadata={"hnsw:space": "cosine"},
)

# ──────────────────────────────
# Utility Functions
# ──────────────────────────────
def normalize_text(text: str) -> str:
    """Normalize text for consistent fuzzy and lexical matching."""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = re.sub(r"[^a-zA-Z0-9+ ']+", " ", text)
    return text.lower().strip()


def parse_timestamp(ts):
    """Safely parse timestamps into timezone-aware datetime."""
    if not ts:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        ts = str(ts).replace("Z", "+00:00")
        return date_parser.parse(ts)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)

# ──────────────────────────────
# Build or Rebuild Vector Index
# ──────────────────────────────
def build_index(messages, batch_size=150):
    """
    Builds the embedding index for all messages.
    This runs once or whenever the messages or model are updated.
    """
    print("🔄 Building embedding index...")
    if collection.count() > 0:
        print("🧹 Clearing existing embeddings…")
        try:
            collection.delete(where={"user_name": {"$ne": None}})
        except Exception as e:
            print(f"⚠️ Could not clear collection: {e}")

    texts = [m["message"] for m in messages]
    ids = [m["id"] for m in messages]
    metas = [
        {
            "user_name": m["user_name"],
            "user_id": m["user_id"],
            "timestamp": m.get("timestamp"),
        }
        for m in messages
    ]

    total = len(texts)
    for i in tqdm(range(0, total, batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metas = metas[i:i + batch_size]
        embeddings = model.encode(
            batch_texts,
            show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()
        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_metas,
        )

    print(f"✅ Indexed {total} messages successfully using {EMBED_MODEL}.")

# ──────────────────────────────
# User Detection
# ──────────────────────────────
def detect_user_name(question, all_user_names):
    """Detect which member name is referenced in the question."""
    norm_q = normalize_text(question)
    best_match, best_score = None, 0

    for u in all_user_names:
        u_norm = normalize_text(u)
        parts = [u_norm] + u_norm.split()
        for part in parts:
            score = fuzz.partial_ratio(part, norm_q)
            if score > best_score:
                best_score, best_match = score, u

    if best_score >= 55:
        print(f"🧭 Detected user: {best_match} (score {best_score})")
        return best_match
    print("⚠️ No user detected.")
    return None

# ──────────────────────────────
# Retrieval Logic
# ──────────────────────────────
def retrieve_relevant_messages(question, top_k=10, user_name=None):
    """
    High-accuracy semantic retrieval pipeline.
      1️⃣ Restrict search to user (with fallback to global).
      2️⃣ Retrieve top semantically similar messages.
      3️⃣ Expand via centroid similarity for context.
      4️⃣ Rank by recency + relevance.
      5️⃣ Sort newest-first for context clarity.
    """
    q_text = f"{user_name}: {question}" if user_name else question
    q_emb = model.encode(q_text, normalize_embeddings=True).tolist()

    # 1️⃣ Primary search — only within detected user's messages
    if user_name:
        print(f"🎯 Searching within {user_name}'s messages…")
        query = {
            "query_embeddings": [q_emb],
            "n_results": top_k * 3,
            "where": {"user_name": user_name}
        }
        results = collection.query(**query)
    else:
        results = collection.query(query_embeddings=[q_emb], n_results=top_k * 3)

    # 2️⃣ Fallback to global search if none found
    if not results.get("documents"):
        print("⚠️ No user-specific matches — falling back to global search.")
        results = collection.query(query_embeddings=[q_emb], n_results=top_k * 3)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    scores = results.get("distances", [[]])[0]

    # 3️⃣ Centroid expansion (topical coherence)
    if len(docs) > 1:
        seed_embs = model.encode(
            docs[: min(len(docs), 12)],
            show_progress_bar=False,
            normalize_embeddings=True
        )
        centroid = np.mean(seed_embs, axis=0).tolist()
        expand_results = collection.query(
            query_embeddings=[centroid],
            n_results=top_k,
            where={"user_name": user_name} if user_name else None
        )
        docs += expand_results["documents"][0]
        metas += expand_results["metadatas"][0]
        scores += expand_results.get("distances", [[]])[0]

    # 4️⃣ Combine + score by recency and semantic relevance
    now = datetime.now(timezone.utc)
    combined = []
    for d, m, s in zip(docs, metas, scores):
        ts = parse_timestamp(m.get("timestamp"))
        combined.append({
            "text": d,
            "user_name": m.get("user_name"),
            "user_id": m.get("user_id"),
            "timestamp": ts,
            "score": s,
        })

    # 5️⃣ Deduplicate by message text
    seen = set()
    unique = []
    for c in combined:
        if c["text"] not in seen:
            unique.append(c)
            seen.add(c["text"])

    # 6️⃣ Sort newest-first, then by semantic proximity
    unique.sort(key=lambda x: (-x["timestamp"].timestamp(), x["score"]))

    # 7️⃣ Keep only this user's messages
    if user_name:
        unique = [u for u in unique if u["user_name"] == user_name]

    # 8️⃣ Return top results
    final = unique[:20]

    print(f"📑 Retrieved {len(final)} contextual messages for {user_name or 'unknown user'} (newest first).")
    for msg in final[:5]:
        ts = msg["timestamp"].isoformat() if msg["timestamp"] != datetime.min else "N/A"
        print(f"  - [{ts}] {msg['user_name']}: {msg['text'][:90]}")

    return final