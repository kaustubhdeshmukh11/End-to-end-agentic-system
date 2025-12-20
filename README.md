# 🧠 Autonomous Deep-Research Agent

A stateful, autonomous AI agent that recursively performs web research, downloads academic papers, and maintains long-term memory using a persistent graph architecture.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Orchestration-orange)
![Groq](https://img.shields.io/badge/Groq-Llama_3_70B-purple)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

## 🚀 Key Highlights

* **Autonomous Execution:** Engineered a multi-step agent that recursively executes web research (Tavily), auto-downloads Arxiv papers, and updates its own RAG vector store in real-time without human intervention.
* **Persistent State Management:** Architected a stateful, multi-user system using **LangGraph** and **SQLite**, implementing persistent memory checkpoints to support long-running, concurrent research sessions.
* **Hybrid "God Stack" ($0 Cost):**
    * **Inference:** Llama 3.3 70B (via Groq) for sub-second tool calling.
    * **Embeddings:** Google Gemini (`text-embedding-004`) for high-performance retrieval.
    * **Knowledge:** FAISS (Vector) + SQLite (Relational).

---

## 🧩 System Architecture

The system follows a **First Principles** design, decoupling perception, cognition, and memory:

1.  **Cognition (The Graph):**
    * A cyclic state machine (`Start` → `Agent` ↔ `Tools` → `End`) enables self-correction.
    * If a search result is insufficient, the agent loops back to refine its query or switch tools.
    
2.  **Memory (The Persistence):**
    * **Long-Term:** `SqliteSaver` checkpoints every interaction, allowing users to pause and resume sessions.
    * **Short-Term:** `_THREAD_RETRIEVERS` maps specific user threads to their downloaded PDF indices in memory.

3.  **Perception (The Interface):**
    * A stateless **Streamlit** UI that mirrors the backend state using `st.session_state` and streams tokens in real-time.

---

## 🛠️ Tech Stack

* **Orchestration:** LangGraph, LangChain
* **Database:** SQLite (State Checkpoints), FAISS (Vector Store)
* **Models:** Groq (Llama-3.3-70b-versatile), Google Generative AI (Embeddings)
* **Tools:** Tavily (Web Search), Arxiv API, PyPDFLoader

---

