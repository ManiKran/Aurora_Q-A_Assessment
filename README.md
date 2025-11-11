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
| **Embedding Model** | `MiniLM-L6-v2` (faster) | Missed nuanced context. | ✅ Chose `all-mpnet-base-v2` |
| **User Detection** | Pure fuzzy match | Incorrect user attribution. | ✅ Hybrid literal + fuzzy |
| **Retrieval Scope** | Global retrieval | Pulled irrelevant context. | ✅ User-scoped with fallback |
| **Generation** | Free-form LLM output | Hallucinated information. | ✅ Controlled rule-based output |
| **Index Handling** | Rebuild each run | High startup latency. | ✅ Cached persistent Chroma index |

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

## App Link: 
https://aurora-assessment-saimanikiran.up.railway.app/
---

## Output Examples:

1. Question: When is Layla planning to go to london
```bash
Answer: I don’t have the exact information for this, but based on the available context, Layla has mentioned needing a car service and a chauffeur in London, and she also requires a suite at Claridge's starting Monday. However, the specific date for her trip to London is not explicitly stated.
```

2. Question: What are Sophia's food preferences recently?
```bash
Answer: I don’t have the exact information for this, but based on the available context, Sophia's recent food preferences include a gluten-free menu for her stay in Paris, vegetarian meals at Italian restaurants, and a vegetarian choice at The Ritz. Additionally, she prefers a chef's tasting menu when available and has dietary restrictions regarding shellfish.
```

3. Question: What layla thinks about her venice trip?
```bash
Answer: Lyla thinks her Venice trip was sensational, as she expressed gratitude for the team's efforts regarding that trip.
```

4. Question: What is Amina's latest concern?
```bash
Answer: I don’t have the exact information for this, but based on the available context, Amina's latest concern appears to be regarding the billing statement for her recent yacht trip, as mentioned in her message dated October 10, 2025.
```

5. Question: How many cars does Thiago has?
```bash
Answer: Idon’t have any information about the question you asked.
```
---

## 📊 Data Quality & Anomaly Analysis

Before building embeddings and running retrieval, I performed a detailed integrity check on the member messages dataset fetched via the public API.

### ✅ Summary of Data Validation

| Check | Description | Result |
|--------|--------------|---------|
| **Total Messages** | Total records analyzed from cache/API | **3349** |
| **Missing Values** | Checked across all columns (`id`, `user_id`, `user_name`, `timestamp`, `message`) | **None found** |
| **Empty / Whitespace Messages** | Messages with blank or whitespace-only text | **0** |
| **Duplicate IDs** | Duplicate unique identifiers | **0** |
| **Duplicate Message Texts** | Repeated message content across users | **0** |
| **User Inconsistencies** | Members with multiple or conflicting name entries | **0** |
| **Invalid Timestamps** | Non-parsable or malformed timestamps | **0** |
| **Future Timestamps** | Entries incorrectly dated after current UTC time | **0** |
| **Extremely Long Messages** | > 1000 characters | **0** |
| **Extremely Short Messages** | < 3 characters | **0** |

---

### 🧩 Observations

- The dataset is **clean and consistent**, with **no structural or semantic anomalies** detected.  
- User identifiers, names, and timestamps are all valid and synchronized.  
- No duplicate or missing records were found across the 3,349 entries.  
- This high data integrity ensures **accurate embeddings** and **reliable retrieval performance** in the RAG pipeline.  

> 🧠 *Conclusion:* The message dataset required **no corrective preprocessing** — its cleanliness allowed me to directly build embeddings using `sentence-transformers/all-mpnet-base-v2` and focus optimization efforts on retrieval and LLM reasoning instead.
