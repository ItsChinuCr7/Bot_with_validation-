"""
Legacy project generator retained for reference only.
It is disabled because its embedded app template bypasses centralized guardrails.
"""

FILES = {

"config/__init__.py": "",

"config/settings.py": '''
LLM_MODEL = "phi3"
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_INDEX_PATH = "faiss_index"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 8
FINAL_CONTEXT_DOCS = 4
MAX_CHARS_PER_DOC = 700
MAX_REASONING_DEPTH = 2
''',

"models/__init__.py": "",

"models/llm.py": '''
import streamlit as st
from langchain_ollama import ChatOllama
from config.settings import LLM_MODEL, OLLAMA_URL

@st.cache_resource
def load_llm():
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0.1,
        num_predict=512,
    )
''',

"models/embeddings.py": '''
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
''',

"ingestion/__init__.py": "",

"ingestion/pdf_loader.py": '''
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

def load_and_split_pdfs(uploaded_files):
    documents = []
    tmp_paths = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_paths.append(tmp.name)
            loader = PyPDFLoader(tmp.name)
            pages = loader.load()
            for page in pages:
                page.metadata["source_file"] = file.name
            documents.extend(pages)
    for path in tmp_paths:
        try:
            os.unlink(path)
        except Exception:
            pass
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\\n\\n", "\\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
''',

"retrieval/__init__.py": "",

"retrieval/vector_store.py": '''
from langchain_community.vectorstores import FAISS
from config.settings import VECTOR_INDEX_PATH

def create_vector_store(docs, embeddings):
    vector_db = FAISS.from_documents(docs, embeddings)
    vector_db.save_local(VECTOR_INDEX_PATH)
    return vector_db

def load_vector_store(embeddings):
    return FAISS.load_local(
        VECTOR_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
''',

"retrieval/retriever.py": '''
from config.settings import TOP_K_RETRIEVAL

def get_mmr_retriever(vector_db):
    return vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K_RETRIEVAL,
            "fetch_k": TOP_K_RETRIEVAL * 2,
            "lambda_mult": 0.6,
        },
    )
''',

"retrieval/hybrid_search.py": '''
from rank_bm25 import BM25Okapi

class HybridSearcher:
    def __init__(self, docs, vector_db, alpha=0.5):
        self.docs = docs
        self.vector_db = vector_db
        self.alpha = alpha
        tokenized = [doc.page_content.lower().split() for doc in docs]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, top_k=8):
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_norm = bm25_scores / bm25_max

        vector_results = self.vector_db.similarity_search_with_score(query, k=top_k * 2)
        vector_score_map = {}
        for doc, score in vector_results:
            for i, d in enumerate(self.docs):
                if d.page_content == doc.page_content:
                    vector_score_map[i] = 1.0 / (1.0 + score)
                    break

        combined = []
        for i, doc in enumerate(self.docs):
            v_score = vector_score_map.get(i, 0.0)
            b_score = float(bm25_norm[i])
            hybrid_score = self.alpha * v_score + (1 - self.alpha) * b_score
            combined.append((hybrid_score, doc))

        combined.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in combined[:top_k]]
''',

"retrieval/reranker.py": '''
import streamlit as st
from sentence_transformers import CrossEncoder
from config.settings import FINAL_CONTEXT_DOCS

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

def rerank_docs(query, docs, top_k=FINAL_CONTEXT_DOCS):
    reranker = load_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]
''',

"reasoning/__init__.py": "",

"reasoning/query_rewriter.py": '''
REWRITE_PROMPT = """Rewrite this question into a keyword-dense search query.
Rules: be specific, remove filler words, max 15 words, return ONLY the query.

Question: {query}
Rewritten query:"""

def rewrite_query(llm, query):
    try:
        result = llm.invoke(REWRITE_PROMPT.format(query=query)).content.strip()
        if not result or len(result) > 200:
            return query
        return result
    except Exception:
        return query
''',

"reasoning/context_builder.py": '''
from config.settings import MAX_CHARS_PER_DOC, FINAL_CONTEXT_DOCS

CONTEXT_CHAR_BUDGET = 2500

def build_context(docs):
    context_parts = []
    total_chars = 0
    seen = set()

    for doc in docs[:FINAL_CONTEXT_DOCS]:
        content = doc.page_content.strip()
        fingerprint = content[:100]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        content = content[:MAX_CHARS_PER_DOC]
        if total_chars + len(content) > CONTEXT_CHAR_BUDGET:
            remaining = CONTEXT_CHAR_BUDGET - total_chars
            if remaining < 100:
                break
            content = content[:remaining]
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[Source: {source} | Page: {page}]\\n{content}")
        total_chars += len(content)

    return "\\n\\n---\\n\\n".join(context_parts)
''',

"reasoning/rag_pipeline.py": '''
from retrieval.reranker import rerank_docs
from reasoning.context_builder import build_context
from reasoning.query_rewriter import rewrite_query
from config.settings import MAX_REASONING_DEPTH

RAG_PROMPT = """You are a precise document assistant.
Answer using ONLY the context below.
If not found, say: "I could not find this in the documents."
Be concise. Cite source file when possible.

Question: {question}

Context:
{context}

Answer:"""

def run_rag_pipeline(llm, retriever, query, depth=0):
    search_query = rewrite_query(llm, query) if depth == 0 else query
    docs = retriever.invoke(search_query)
    if not docs:
        return "No relevant documents found. Please upload and process PDFs first."
    ranked_docs = rerank_docs(search_query, docs)
    context = build_context(ranked_docs)
    prompt = RAG_PROMPT.format(question=query, context=context)
    answer = llm.invoke(prompt).content.strip()

    if depth < MAX_REASONING_DEPTH:
        uncertainty = ["i could not find", "not in the context", "need more information"]
        for signal in uncertainty:
            if signal in answer.lower():
                fallback = f"Write a different 5-word search query for: {query}\\n\\nQuery:"
                new_query = llm.invoke(fallback).content.strip()
                return run_rag_pipeline(llm, retriever, new_query, depth + 1)
    return answer
''',

"agents/__init__.py": "",

"agents/agent_router.py": '''
ROUTING_RULES = {
    "web": ["latest", "news", "today", "current", "2024", "2025", "recent"],
    "calc": ["calculate", "compute", "math", "solve", "equation"],
    "chat": ["hello", "hi ", "how are you", "who are you", "what can you do"],
}

def route_query(query):
    q = query.lower()
    for tool, keywords in ROUTING_RULES.items():
        for kw in keywords:
            if kw in q:
                return tool
    return "rag"

def handle_web_query(query):
    return "Web search not connected yet. Add Tavily API key to enable it."

def handle_calc_query(llm, query):
    return llm.invoke(f"Solve step by step:\\n{query}").content

def handle_chat_query(llm, query):
    return llm.invoke(f"Answer conversationally:\\n{query}").content
''',

"evaluation/__init__.py": "",

"evaluation/answer_validator.py": '''
UNCERTAINTY_PHRASES = [
    "i don\'t know", "i do not know", "i\'m not sure",
    "i cannot find", "i could not find", "not in the context",
]

def validate_answer(answer):
    if not answer or len(answer.strip()) < 15:
        return {"valid": False, "confidence": "low", "flag": "Answer too short."}
    for phrase in UNCERTAINTY_PHRASES:
        if phrase in answer.lower():
            return {"valid": True, "confidence": "low", "flag": f"LLM uncertain: \'{phrase}\'"}
    confidence = "high" if len(answer) > 200 else "medium" if len(answer) > 80 else "low"
    return {"valid": True, "confidence": confidence, "flag": None}

def format_confidence_badge(confidence):
    return {"high": "🟢 High confidence", "medium": "🟡 Medium confidence", "low": "🔴 Low confidence"}.get(confidence, "")
''',

"app.py": '''
import streamlit as st
import os
import time

from models.llm import load_llm
from models.embeddings import load_embeddings
from ingestion.pdf_loader import load_and_split_pdfs
from retrieval.vector_store import create_vector_store, load_vector_store
from retrieval.retriever import get_mmr_retriever
from retrieval.hybrid_search import HybridSearcher
from reasoning.rag_pipeline import run_rag_pipeline
from agents.agent_router import route_query, handle_web_query, handle_calc_query, handle_chat_query
from evaluation.answer_validator import validate_answer, format_confidence_badge
from config.settings import VECTOR_INDEX_PATH

st.set_page_config(page_title="AI Document Assistant", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.user-bubble { background-color:#2b313e; padding:12px 16px; border-radius:12px; margin-bottom:10px; text-align:right; color:white; }
.bot-bubble { background-color:#1a1a2e; padding:12px 16px; border-radius:12px; margin-bottom:15px; color:#00ffcc; border-left:3px solid #00ffcc; }
.meta-line { font-size:12px; color:#888; margin-bottom:20px; }
</style>
""", unsafe_allow_html=True)

llm = load_llm()
embeddings = load_embeddings()

with st.sidebar:
    st.title("⚙️ Document Settings")
    st.divider()
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    use_hybrid = st.checkbox("Enable Hybrid Search (BM25 + Vector)", value=True)

    if st.button("📂 Process Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            with st.spinner("Indexing..."):
                docs = load_and_split_pdfs(uploaded_files)
                vector_db = create_vector_store(docs, embeddings)
                st.session_state["vector_db"] = vector_db
                st.session_state["docs"] = docs
                if use_hybrid:
                    st.session_state["hybrid_searcher"] = HybridSearcher(docs, vector_db, alpha=0.5)
            st.success(f"✅ {len(docs)} chunks indexed from {len(uploaded_files)} file(s).")

    st.divider()
    st.caption("Query Rewriting → MMR/Hybrid → Re-rank → RAG")

if "vector_db" not in st.session_state and os.path.exists(VECTOR_INDEX_PATH):
    try:
        st.session_state["vector_db"] = load_vector_store(embeddings)
    except Exception:
        pass

st.title("🤖 AI Document Intelligence Assistant")

if "vector_db" not in st.session_state:
    st.info("👈 Upload and process your PDFs from the sidebar to get started.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        st.markdown(f\'<div class="user-bubble">{entry["message"]}</div>\', unsafe_allow_html=True)
    else:
        st.markdown(f\'<div class="bot-bubble">{entry["message"]}</div>\', unsafe_allow_html=True)
        if entry.get("meta"):
            st.markdown(f\'<div class="meta-line">{entry["meta"]}</div>\', unsafe_allow_html=True)

user_input = st.chat_input("Ask something about your documents...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "message": user_input})

    with st.spinner("Thinking..."):
        start = time.time()
        route = route_query(user_input)

        if route == "web":
            answer = handle_web_query(user_input)
        elif route == "calc":
            answer = handle_calc_query(llm, user_input)
        elif route == "chat":
            answer = handle_chat_query(llm, user_input)
        else:
            if "vector_db" not in st.session_state:
                answer = "Please upload and process PDFs from the sidebar first."
            else:
                if use_hybrid and "hybrid_searcher" in st.session_state:
                    class HybridRetriever:
                        def __init__(self, searcher):
                            self.searcher = searcher
                        def invoke(self, query):
                            from config.settings import TOP_K_RETRIEVAL
                            return self.searcher.search(query, top_k=TOP_K_RETRIEVAL)
                    retriever = HybridRetriever(st.session_state["hybrid_searcher"])
                else:
                    retriever = get_mmr_retriever(st.session_state["vector_db"])
                answer = run_rag_pipeline(llm, retriever, user_input)

        elapsed = round(time.time() - start, 2)
        validation = validate_answer(answer)
        badge = format_confidence_badge(validation["confidence"])
        meta_parts = [f"⏱ {elapsed}s", f"🔀 Route: {route}", badge]
        if validation.get("flag"):
            meta_parts.append(f"⚠️ {validation[\'flag\']}")
        meta_str = " &nbsp;|&nbsp; ".join(meta_parts)

    st.session_state.chat_history.append({"role": "bot", "message": answer, "meta": meta_str})
    st.rerun()
''',

"requirements.txt": """streamlit
langchain
langchain-community
langchain-ollama
langchain-huggingface
langchain-text-splitters
faiss-cpu
sentence-transformers
rank_bm25
pypdf
""",
}

def create_project():
    raise RuntimeError(
        "This legacy project generator is disabled because its embedded app "
        "bypasses centralized guardrails. Run the maintained app.py instead."
    )

if __name__ == "__main__":
    create_project()