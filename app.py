import streamlit as st
import os
import time
import logging
from config.logging_config import setup_logging
setup_logging("INFO")
logger = logging.getLogger(__name__)

from models.llm import load_llm, load_llm_heavy
from models.model_factory import get_available_models, get_model_info
from models.embeddings import load_embeddings
from ingestion.pdf_loader import load_and_split_pdfs
from retrieval.vector_store import create_vector_store, load_vector_store
from retrieval.retriever import get_mmr_retriever
from retrieval.hybrid_search import HybridSearcher
from reasoning.rag_pipeline import run_rag_pipeline, run_general_with_memory, format_history
from inference.inference_engine import run_inference
from data.cache.cache_manager import get_cache_stats, clear_cache
from agents.agent_router import (
    route_query, route_with_llm,
    handle_web_query, handle_chat_query,
)
from research.web_researcher import (
    decide_sources, parse_user_source_override,
    run_research, extract_research_query,
)
from evaluation.answer_validator import validate_answer, format_confidence_badge
from config.settings import VECTOR_INDEX_PATH

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Global ── */
body { background-color: #0e0e16; }

/* ── Chat bubbles ── */
.user-bubble {
    background: linear-gradient(135deg, #1e2a3a, #243040);
    padding: 12px 18px; border-radius: 18px 18px 4px 18px;
    margin: 6px 0 6px 60px; color: #e0e8f0;
    border: 1px solid #2a3a4a; font-size: 15px;
    text-align: right;
}
.bot-bubble {
    background: linear-gradient(135deg, #0d1a2a, #102030);
    padding: 14px 18px; border-radius: 4px 18px 18px 18px;
    margin: 6px 60px 6px 0; color: #7de8cc;
    border-left: 3px solid #00ccaa; font-size: 15px;
}
.healed-bubble {
    background: linear-gradient(135deg, #0a1f12, #0d2a18);
    padding: 14px 18px; border-radius: 4px 18px 18px 18px;
    margin: 6px 60px 6px 0; color: #44ff99;
    border-left: 3px solid #00ee66; font-size: 15px;
}

/* ── Source cards ── */
.source-card {
    background: #080e18; border: 1px solid #1a2a3a;
    border-radius: 10px; padding: 12px 16px; margin: 8px 0;
    font-size: 13px; color: #7799cc;
}
.source-card b { color: #99bbee; }
.source-card a { color: #5599ff; text-decoration: none; }
.source-card a:hover { text-decoration: underline; }

/* ── Gemini feedback ── */
.gemini-card {
    background: #0f0f00; border: 1px solid #3a3a00;
    border-radius: 10px; padding: 12px 16px; margin: 8px 0;
    font-size: 13px; color: #cccc55; white-space: pre-wrap;
}

/* ── Meta line ── */
.meta-line {
    font-size: 11px; color: #556677;
    margin: 2px 0 10px 4px; letter-spacing: 0.3px;
}

/* ── Stat box ── */
.stat-box {
    background: #0a0a14; border: 1px solid #222233;
    border-radius: 8px; padding: 10px 14px;
    font-size: 12px; color: #8899aa; line-height: 1.8;
}

/* ── Route badge ── */
.badge {
    display: inline-block; padding: 2px 8px;
    border-radius: 10px; font-size: 11px; font-weight: 600;
    margin-left: 6px; vertical-align: middle;
}
.badge-research { background:#1a2a00; color:#88cc00; border:1px solid #446600; }
.badge-rag      { background:#001a2a; color:#0088cc; border:1px solid #004466; }
.badge-general  { background:#1a001a; color:#aa44aa; border:1px solid #660066; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #0a0a14; }
section[data-testid="stSidebar"] .stRadio label { font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ── Load models (cached) ──────────────────────────────────
@st.cache_resource
def _load_models(model_name: str = None):
    return load_llm(model_name), load_llm_heavy(), load_embeddings()

# Model selector — persists across reruns
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = None

_sel = st.session_state.get("selected_model")
llm, llm_heavy, embeddings = _load_models(_sel)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    # ── Model Selector ─────────────────────────────────────
    st.markdown("**🤖 Active Model**")
    available_models = get_available_models()
    from config.settings import LLM_MODEL
    default_idx = available_models.index(LLM_MODEL) if LLM_MODEL in available_models else 0
    chosen_model = st.selectbox(
        "Select model",
        available_models,
        index=default_idx,
        label_visibility="collapsed",
    )
    if chosen_model != st.session_state.get("selected_model"):
        st.session_state["selected_model"] = chosen_model
        _load_models.clear()
        st.rerun()

    # Show model info
    info = get_model_info(chosen_model)
    if info["parameters"] != "unknown":
        st.caption(f"Params: {info['parameters']} · {info['quantization']} · {info['family']}")
    st.divider()

    uploaded_files = st.file_uploader(
        "📎 Upload PDFs", type="pdf", accept_multiple_files=True,
        help="Upload one or more PDF documents to query"
    )
    use_hybrid = st.checkbox("🔀 Hybrid Search (BM25 + Vector)", value=True,
        help="Combines keyword search with semantic search for better results")

    if st.button("⚡ Process Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            with st.status("Processing documents...", expanded=True) as status:

                # ── Step 1: Read PDFs ──────────────────────────
                st.write("📖 **Step 1/4** — Reading PDF files...")
                prog = st.progress(0, text="Loading pages...")
                import time as _time
                _time.sleep(0.2)
                prog.progress(10, text="Parsing PDF structure...")

                chunks, doc_trees = load_and_split_pdfs(uploaded_files)

                prog.progress(30, text=f"Extracted {len(chunks)} chunks")
                st.write(f"✅ **{len(chunks)} chunks** extracted from {len(uploaded_files)} file(s)")

                # ── Step 2: Document Tree ──────────────────────
                st.write("🌳 **Step 2/4** — Building document tree...")
                prog.progress(45, text="Mapping document structure...")
                total_nodes = sum(len(t.all_nodes) for t in doc_trees.values())
                st.write(f"✅ **{total_nodes} sections** indexed in tree")

                # ── Step 3: Vector Index ───────────────────────
                st.write("🔢 **Step 3/4** — Building FAISS vector index...")
                from config.settings import EMBEDDING_MODEL
                st.caption(f"  Embedding {len(chunks)} chunks with {EMBEDDING_MODEL} — processing vectors...")
                prog.progress(50, text=f"Embedding chunks (0/{len(chunks)})...")

                # Batch progress simulation during embedding
                batch = max(1, len(chunks) // 10)
                for i in range(10):
                    prog.progress(50 + i*4, text=f"Embedding chunks (~{(i+1)*batch}/{len(chunks)})...")
                    _time.sleep(0.05)

                vector_db = create_vector_store(chunks, embeddings)
                prog.progress(90, text="Vector index built!")
                st.write(f"✅ **FAISS index ready** — {len(chunks)} vectors stored")

                st.session_state["vector_db"] = vector_db
                st.session_state["docs"]       = chunks
                st.session_state["doc_trees"]  = doc_trees

                # ── Step 4: BM25 ───────────────────────────────
                if use_hybrid:
                    st.write("📊 **Step 4/4** — Building BM25 keyword index...")
                    prog.progress(93, text="Building BM25 index...")
                    st.session_state["hybrid_searcher"] = HybridSearcher(chunks, vector_db, alpha=0.5)
                    st.write("✅ **BM25 index ready**")
                else:
                    st.write("⏭️ **Step 4/4** — BM25 skipped (Hybrid Search off)")

                prog.progress(100, text="All done!")
                st.balloons()
                status.update(
                    label=f"✅ Done! {len(chunks)} chunks · {total_nodes} sections · ready to query",
                    state="complete"
                )

    st.divider()

    memory_on = st.checkbox("🧠 Conversation Memory", value=True,
        help="Remember previous messages in this session")

    st.divider()
    st.markdown("**🎯 Answer Mode**")
    answer_mode = st.radio(
        label="answer_mode_radio",
        options=["🤖 Auto (smart detect)", "💡 General (LLM Knowledge)", "📄 RAG (documents)"],
        index=0,
        label_visibility="collapsed",
    )
    mode_map = {
        "🤖 Auto (smart detect)": None,
        "💡 General (LLM Knowledge)": "general",
        "📄 RAG (documents)":     "rag",
    }
    forced_mode = mode_map[answer_mode]

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()



    st.divider()
    st.markdown("**📋 Logs**")
    log_path = os.path.join(os.path.dirname(__file__), "logs", "app.log")
    if os.path.exists(log_path):
        with st.expander("View recent logs"):
            with open(log_path, "r", encoding="utf-8") as _lf:
                lines = _lf.readlines()
            last50 = "".join(lines[-50:])
            st.code(last50, language="text")
        if st.button("🗑️ Clear Logs", use_container_width=True):
            open(log_path, "w").close()
            st.success("Logs cleared!")
    else:
        st.caption("No logs yet")

# ── Auto-load saved vector index ─────────────────────────
if "vector_db" not in st.session_state and os.path.exists(VECTOR_INDEX_PATH):
    try:
        st.session_state["vector_db"] = load_vector_store(embeddings)
    except Exception:
        pass

# ── Header ────────────────────────────────────────────────
st.markdown("# 🤖 AI Document Intelligence Assistant")

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    if "vector_db" not in st.session_state:
        st.info("📭 No documents loaded — upload PDFs or ask general questions!")
    else:
        tree_count = sum(len(t.all_nodes) for t in st.session_state.get("doc_trees", {}).values())
        st.success(f"📚 Documents loaded — {tree_count} tree sections indexed")
with col2:
    if memory_on:
        st.success("🧠 Memory ON")
    else:
        st.warning("Memory OFF")
with col3:
    pass

# ── Chat history ──────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        st.markdown(f'<div class="user-bubble">{entry["message"]}</div>', unsafe_allow_html=True)
    else:
        bubble = "bot-bubble"
        st.markdown(f'<div class="{bubble}">{entry["message"]}</div>', unsafe_allow_html=True)

        if entry.get("source_extracts"):
            with st.expander(f"📄 View {len(entry['source_extracts'])} scraped source(s)"):
                for r in entry["source_extracts"]:
                    src   = r.get("source", "?").upper()
                    url   = r.get("url", "")
                    body  = r.get("content", "")[:600]
                    dots  = "..." if len(r.get("content", "")) > 600 else ""
                    st.markdown(f"""<div class="source-card">
<b>🔗 {src}</b> — <a href="{url}" target="_blank">{url[:75]}</a><br><br>
{body}{dots}
</div>""", unsafe_allow_html=True)

        if entry.get("meta"):
            st.markdown(f'<div class="meta-line">{entry["meta"]}</div>', unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────
user_input = st.chat_input("Ask anything — upload docs to query them, or ask general questions...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    history_for_llm  = st.session_state.chat_history[:-1] if memory_on else []

    start = time.time()

    with st.status("Thinking...", expanded=True) as status:

        logger.info(f"Query: {user_input[:80]!r}")
        result = run_inference(
            query          = user_input,
            llm            = llm,
            llm_heavy      = llm_heavy,
            session_state  = st.session_state,
            memory_on      = memory_on,
            forced_mode    = forced_mode,
            use_hybrid     = use_hybrid,
            progress_callback = st.write,
        )

        logger.info(f"Route: {result['route']} | Answer length: {len(result['answer'])} chars")
        answer           = result["answer"]
        route            = result["route"]
        retrieval_method = result["retrieval_method"]
        chosen_sources   = result["chosen_sources"]
        source_extracts  = result["source_extracts"]
        context_used     = result["context_used"]

        elapsed    = round(time.time() - start, 2)
        validation = validate_answer(answer)
        badge      = format_confidence_badge(validation["confidence"])

        status.update(
            label=f"Done in {elapsed}s — {route.upper()}",
            state="complete",
            expanded=False,
        )

    # Meta line
    parts = [f"⏱ {elapsed}s", f"Route: {route.upper()}"]
    if retrieval_method:
        parts.append(f"Retrieval: {retrieval_method}")
    if chosen_sources:
        parts.append(f"Sources: {' · '.join(s.upper() for s in chosen_sources)}")
    parts.append(badge)

    st.session_state.chat_history.append({
        "role":            "bot",
        "message":         answer,
        "source_extracts": source_extracts,
        "meta":            " &nbsp;|&nbsp; ".join(parts),
    })
    st.rerun()