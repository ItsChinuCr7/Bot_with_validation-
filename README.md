# 🤖 Bot with Validation — AI Document Intelligence Assistant

An advanced RAG system with self-healing answers, web research, and permanent mistake memory.

---

## ✨ Features

- **Hybrid Search** — BM25 keyword + FAISS vector search combined
- **Document Tree Navigation** — Vectorless reasoning-first retrieval
- **Web Research** — Scrapes DuckDuckGo + Gemini via Selenium (no API key needed)
- **Self-Healing** — Answers validated by Gemini, mistakes saved permanently
- **Conversation Memory** — Remembers last 3 turns
- **Answer Mode Selector** — Auto / Research / General / RAG
- **Permanent Learning** — Past mistakes stored in JSON, injected into future prompts

---

## 🏗️ Architecture

```
Query
  ├── Auto-routing (chat / calc / research / general / RAG)
  ├── Document Tree → structure-aware retrieval
  ├── Hybrid Search → BM25 + Vector + MMR re-ranking
  ├── Web Research  → DuckDuckGo + Gemini synthesis
  └── Self-Healing  → Gemini validates → corrects → saves lesson
```

---

## 📁 Project Structure

```
ai_doc_system/
├── app.py                    ← Main Streamlit app
├── requirements.txt
├── agents/
│   └── agent_router.py       ← Smart query routing
├── config/
│   └── settings.py
├── evaluation/
│   ├── answer_validator.py
│   └── self_healer.py        ← Gemini validator + permanent memory
├── ingestion/
│   ├── document_tree.py      ← Hierarchical document tree builder
│   └── pdf_loader.py
├── models/
│   ├── embeddings.py
│   └── llm.py
├── reasoning/
│   └── rag_pipeline.py
├── research/
│   └── web_researcher.py     ← Selenium scraper (DuckDuckGo + Gemini)
└── retrieval/
    ├── hybrid_search.py
    ├── smart_retriever.py
    ├── tree_retriever.py
    └── vector_store.py
```

---

## 🚀 Setup

### 1. Install dependencies
```powershell
pip install streamlit langchain langchain-community langchain-ollama langchain-huggingface langchain-text-splitters faiss-cpu sentence-transformers rank_bm25 pymupdf selenium webdriver-manager beautifulsoup4 pyperclip
```

### 2. Install Ollama models
```powershell
ollama pull phi3
ollama pull llama3:8b-instruct-q4_0
```

### 3. Start Brave with debug port (for Gemini features)
```powershell
& "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" --remote-debugging-port=9222
```
Log into Gemini in that Brave window, then minimize it.

### 4. Run the app
```powershell
cd D:\ai_doc_system
streamlit run app.py
```

---

## 🧠 How Self-Healing Works

```
1. phi3 generates answer
2. Answer sent to Gemini via your logged-in Brave
3. Gemini validates: CORRECT / PARTIALLY_CORRECT / WRONG
4. If wrong → Gemini corrected answer shown (green bubble)
5. Mistake + lesson saved to self_healing_memory.json permanently
6. Next similar question → lesson injected into prompt automatically
```

---

## 📋 Requirements

- Windows 10/11
- Python 3.10+
- Ollama running locally
- Brave Browser
- 8GB RAM minimum

---

## 👤 Author

**ItsChinuCr7**
