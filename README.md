# 🧠 Aurora Member Q&A — RAG Architecture & Design Overview

While developing the **Aurora Member Q&A System**, I designed a modular pipeline based on a **Retrieval-Augmented Generation (RAG)** architecture.  
The system answers natural-language questions about members by analyzing their historical messages — with a strong focus on **accuracy, transparency, and deployability**.

---

## 🚀 Overview of the RAG Pipeline

The backend consists of three main stages: **Retrieval**, **Augmentation**, and **Generation** — orchestrated through a FastAPI service.

### **1️⃣ Retrieval**
When a user asks a question, I first identify the referenced member using a **hybrid literal + fuzzy name-matching** algorithm.  
Once identified, I perform **semantic retrieval** from a **ChromaDB** vector store that holds pre-computed embeddings of all messages.

**Key retrieval logic:**
- Message embeddings are generated using **`sentence-transformers/all-mpnet-base-v2`**.  
- A persistent ChromaDB index ensures fast re-use between sessions.  
- Retrieval first searches within the detected member’s messages, then falls back to global search.  
- I apply **centroid expansion** to enrich context and ensure topical coherence.  

> 🧩 *Result:* A ranked list of the most semantically relevant and recent messages for that member.

---

### **2️⃣ Augmentation**
The retrieved messages are cleaned, sorted, and formatted into readable, time-stamped context snippets.  
This context is then appended to the question before sending it to the LLM.
	•	Builds a deterministic prompt enforcing factual reasoning hierarchy.
	•	Interfaces with the OpenAI API (via OPENAI_API_KEY).
	•	Returns structured, grounded answers while normalizing fallbacks for consistency.

---

### **3️⃣ Generation**
For generation, I use **OpenAI GPT (`gpt-4o-mini`)** with a structured reasoning hierarchy to ensure factual consistency.

**Response logic hierarchy:**
1. ✅ *If the answer is explicitly found*, return a concise factual answer.  
2. ⚙️ *If it can be inferred*, start with  
   > “I don’t have the exact information for this, but based on the available context…”  
3. 🚫 *If no context applies*, respond with  
   > “I don’t have any information about the question you asked.”

This approach prevents hallucinations and maintains concise, trustworthy responses.

---

## ⚙️ System Components

### 🧩 **FastAPI Application (`main.py`)**
- `/ask` endpoint handles incoming questions and orchestrates the RAG process.  
- On startup, loads messages via the public API (`utils.py`) and checks for a Chroma index.  
- If missing, it rebuilds embeddings automatically using `build_index()`.

### 🧩 **Retriever Module (`retriever.py`)**
- Generates embeddings with **SentenceTransformer**.  
- Uses **ChromaDB** with cosine similarity for semantic search.  
- Detects users using hybrid **literal + fuzzy matching** (RapidFuzz).  
- Implements multi-stage retrieval:
  - User-scoped search  
  - Centroid expansion  
  - Deduplication and recency sorting  

### 🧩 **LLM Module (`llm.py`)**
- Builds contextual prompts with time-stamped conversation snippets.  
- Uses **OpenAI GPT** for controlled reasoning.  
- Enforces strict generation rules to avoid hallucinations.  
- Handles fallback responses gracefully if API errors occur.

---

## 🧮 Alternative Approaches Considered

| Component | Alternative | Problem Encountered | Final Decision |
|------------|-------------|--------------------|----------------|
| **Knowledge Base** | SQLite / PostgreSQL full-text search | Poor semantic recall. | ✅ Switched to ChromaDB |
| **Embedding Model** | `MiniLM-L6-v2` (faster) | Missed nuanced context. | ✅ Chose `all-mpnet-base-v2` |
| **User Detection** | Pure fuzzy match | Incorrect user attribution. | ✅ Hybrid literal + fuzzy |
| **Retrieval Scope** | Global retrieval | Pulled irrelevant context. | ✅ User-scoped with fallback |
| **Generation** | Free-form LLM output | Hallucinated information. | ✅ Controlled rule-based output |
| **Index Handling** | Rebuild each run | High startup latency. | ✅ Cached persistent Chroma index |

---

## 🧩 Problems Faced & Fixes

| Issue | Root Cause | Solution |
|-------|-------------|----------|
| ❌ Incorrect user mapping | Fuzzy match confusion between similar names | Combined literal substring + fuzzy threshold |
| 🕒 Cold-start latency (~30 s) | SentenceTransformer model loading on CPU | Warm-up during FastAPI startup |
| ⚠️ Duplicate / blank API messages | Data noise from source | Normalized text & filtered whitespace |
| 🔌 502 Bad Gateway on Railway | Fixed port in frontend | Used `npx serve -s dist -l $PORT` |
| 🧱 Host blocking on Vite preview | Railway domain not allowed | Added `preview.allowedHosts` in `vite.config.js` |
| 💬 OpenAI network / quota errors | API interruptions | Implemented exception handling + fallbacks |

---

## 🔍 Key Takeaways

- The **RAG pipeline** significantly improved factual accuracy and transparency.  
- **Quality embeddings** matter more than small latency gains.  
- **Persistent Chroma storage** removed the need for re-indexing.  
- A **structured prompt hierarchy** ensured safe and interpretable generation.  
- Clear module separation simplified maintenance and debugging.  

---

## 🏁 Final Outcome

The final **Aurora Member Q&A System**:
- Provides accurate, context-aware answers grounded in real member data.  
- Operates fully on **CPU-based infrastructure** with persistent vector storage.  
- Deploys seamlessly on **Railway**, supporting scalable API queries.  
- Maintains **explainability** through transparent logs and contextual reasoning.  

> 🧠 *This architecture reflects a deliberate trade-off — prioritizing accuracy, interpretability, and reproducibility over raw speed, resulting in a robust and production-ready RAG system.*

---

## 🗺️ RAG Architecture Flow

```mermaid
graph LR
A[💬 User Question] --> B[🧭 User Detection<br>(Literal + Fuzzy Matching)]
B --> C[🔍 Semantic Retrieval<br>via ChromaDB]
C --> D[🧩 Context Augmentation<br>(Timestamped Messages)]
D --> E[🤖 LLM Generation<br>(GPT-4o-mini)]
E --> F[✅ Grounded Answer]

style A fill:#f9f9f9,stroke:#aaa,stroke-width:1px;
style F fill:#e2f7e2,stroke:#6c6,stroke-width:1px;
