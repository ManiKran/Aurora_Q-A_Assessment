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
import time

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
    """
    Detect which member name is referenced in the question.
    Hybrid approach:
      1️⃣ Literal substring check (most reliable)
      2️⃣ Fuzzy fallback (for partial or misspelled matches)
    """
    norm_q = normalize_text(question)
    best_match, best_score = None, 0

    # 1️⃣ Literal check first
    for u in all_user_names:
        if u.lower() in question.lower():
            print(f"🧭 Detected user literally: {u}")
            return u

        # check for partial match like "thiago" in "thiago's"
        u_parts = normalize_text(u).split()
        for part in u_parts:
            if re.search(rf"\b{part}\b", norm_q):
                print(f"🧭 Detected user (partial literal): {u}")
                return u

    # 2️⃣ Fuzzy fallback (only if literal match fails)
    for u in all_user_names:
        u_norm = normalize_text(u)
        score = fuzz.partial_ratio(u_norm, norm_q)
        if score > best_score:
            best_score, best_match = score, u

    if best_score >= 70:
        print(f"🧭 Fuzzy-detected user: {best_match} (score {best_score})")
        return best_match

    print("⚠️ No user detected.")
    return None

# ──────────────────────────────
# Retrieval Logic
# ──────────────────────────────
def retrieve_relevant_messages(question, top_k=5, user_name=None):
    """
    High-accuracy semantic retrieval pipeline (with detailed timing).
      1️⃣ Encode the query.
      2️⃣ Restrict search to user (with fallback to global).
      3️⃣ Expand via centroid similarity for context.
      4️⃣ Rank by recency + relevance.
      5️⃣ Sort newest-first for clarity.
    """

    print(f"\n🔍 Starting retrieval for question: '{question}'")
    start_total = time.perf_counter()

    # 1️⃣ Encode question
    t0 = time.perf_counter()
    q_text = f"{user_name}: {question}" if user_name else question
    q_emb = model.encode(q_text, normalize_embeddings=True).tolist()
    t1 = time.perf_counter()
    print(f"⏱️ [Step 1] Query embedding took {t1 - t0:.3f}s")

    # 2️⃣ Primary search — within detected user's messages
    if user_name:
        print(f"🎯 Searching within {user_name}'s messages…")
        query = {
            "query_embeddings": [q_emb],
            "n_results": top_k * 3,
            "where": {"user_name": user_name},
        }
        t2 = time.perf_counter()
        results = collection.query(**query)
        t3 = time.perf_counter()
        print(f"⏱️ [Step 2] User-specific Chroma query took {t3 - t2:.3f}s")
    else:
        t2 = time.perf_counter()
        results = collection.query(query_embeddings=[q_emb], n_results=top_k * 3)
        t3 = time.perf_counter()
        print(f"⏱️ [Step 2] Global Chroma query took {t3 - t2:.3f}s")

    # 3️⃣ Fallback to global if no user match
    if not results.get("documents"):
        print("⚠️ No user-specific matches — falling back to global search.")
        t4 = time.perf_counter()
        results = collection.query(query_embeddings=[q_emb], n_results=top_k * 3)
        t5 = time.perf_counter()
        print(f"⏱️ [Step 3] Fallback global query took {t5 - t4:.3f}s")
    else:
        t5 = t3

    # Extract results
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    scores = results.get("distances", [[]])[0]
    print(f"📦 Retrieved {len(docs)} initial results.")

    # 4️⃣ Centroid expansion for topical context
    t6 = time.perf_counter()
    if len(docs) > 1:
        seed_embs = model.encode(
            docs[: min(len(docs), 12)],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        centroid = np.mean(seed_embs, axis=0).tolist()
        expand_results = collection.query(
            query_embeddings=[centroid],
            n_results=top_k,
            where={"user_name": user_name} if user_name else None,
        )
        docs += expand_results["documents"][0]
        metas += expand_results["metadatas"][0]
        scores += expand_results.get("distances", [[]])[0]
        t7 = time.perf_counter()
        print(f"⏱️ [Step 4] Centroid expansion query took {t7 - t6:.3f}s")
    else:
        t7 = t6

    # 5️⃣ Combine + score
    t8 = time.perf_counter()
    now = datetime.now(timezone.utc)
    combined = []
    for d, m, s in zip(docs, metas, scores):
        ts = parse_timestamp(m.get("timestamp"))
        combined.append(
            {
                "text": d,
                "user_name": m.get("user_name"),
                "user_id": m.get("user_id"),
                "timestamp": ts,
                "score": s,
            }
        )
    t9 = time.perf_counter()
    print(f"⏱️ [Step 5] Combine + scoring took {t9 - t8:.3f}s")

    # 6️⃣ Deduplicate
    t10 = time.perf_counter()
    seen = set()
    unique = []
    for c in combined:
        if c["text"] not in seen:
            unique.append(c)
            seen.add(c["text"])
    t11 = time.perf_counter()
    print(f"⏱️ [Step 6] Deduplication took {t11 - t10:.3f}s")

    # 7️⃣ Sort newest-first
    t12 = time.perf_counter()
    unique.sort(key=lambda x: (-x["timestamp"].timestamp(), x["score"]))
    t13 = time.perf_counter()
    print(f"⏱️ [Step 7] Sorting took {t13 - t12:.3f}s")

    # 8️⃣ Filter user-specific
    if user_name:
        unique = [u for u in unique if u["user_name"] == user_name]

    final = unique[:10]

    # 🕒 Total
    total = time.perf_counter() - start_total
    print(f"✅ [Total Retrieval Time] {total:.3f}s for user {user_name or 'unknown'} ({len(final)} messages).")

    # Print sample
    for msg in final[:5]:
        ts = msg["timestamp"].isoformat() if msg["timestamp"] != datetime.min else "N/A"
        print(f"  - [{ts}] {msg['user_name']}: {msg['text'][:90]}")

    return final