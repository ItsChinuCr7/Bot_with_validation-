import streamlit as st
import os
import time

from models.llm import load_llm, load_llm_heavy
from models.embeddings import load_embeddings
from ingestion.pdf_loader import load_and_split_pdfs
from retrieval.vector_store import create_vector_store, load_vector_store
from retrieval.retriever import get_mmr_retriever
from retrieval.hybrid_search import HybridSearcher
from reasoning.rag_pipeline import run_rag_pipeline, run_general_with_memory, format_history
from agents.agent_router import (
    route_query, route_with_llm,
    handle_web_query, handle_calc_query, handle_chat_query,
)
from research.web_researcher import (
    decide_sources, parse_user_source_override,
    run_research, extract_research_query,
)
from evaluation.answer_validator import validate_answer, format_confidence_badge
from evaluation.self_healer import HealingMemory, self_heal
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
.badge-gemini   { background:#1a1a00; color:#ccaa00; border:1px solid #665500; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #0a0a14; }
section[data-testid="stSidebar"] .stRadio label { font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ── Load models (cached) ──────────────────────────────────
@st.cache_resource
def _load_models():
    return load_llm(), load_llm_heavy(), load_embeddings()

llm, llm_heavy, embeddings = _load_models()

if "healing_memory" not in st.session_state:
    st.session_state["healing_memory"] = HealingMemory()
healing_memory = st.session_state["healing_memory"]

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
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
                st.write("📖 Reading PDF files...")
                chunks, doc_trees = load_and_split_pdfs(uploaded_files)
                st.write(f"✅ Extracted **{len(chunks)}** chunks")
                total_nodes = sum(len(t.all_nodes) for t in doc_trees.values())
                st.write(f"🌳 Document tree: **{total_nodes}** sections")
                st.write("🔢 Building vector index...")
                vector_db = create_vector_store(chunks, embeddings)
                st.session_state["vector_db"] = vector_db
                st.session_state["docs"]       = chunks
                st.session_state["doc_trees"]  = doc_trees
                if use_hybrid:
                    st.write("📊 Building BM25 index...")
                    st.session_state["hybrid_searcher"] = HybridSearcher(chunks, vector_db, alpha=0.5)
                status.update(
                    label=f"✅ Done! {len(chunks)} chunks · {total_nodes} tree nodes",
                    state="complete"
                )

    st.divider()

    memory_on = st.checkbox("🧠 Conversation Memory", value=True,
        help="Remember previous messages in this session")

    st.divider()
    st.markdown("**🎯 Answer Mode**")
    answer_mode = st.radio(
        label="answer_mode_radio",
        options=["🤖 Auto (smart detect)", "🔍 Research (web)", "💡 General (phi3)", "📄 RAG (documents)"],
        index=0,
        label_visibility="collapsed",
    )
    mode_map = {
        "🤖 Auto (smart detect)": None,
        "🔍 Research (web)":      "research",
        "💡 General (phi3)":      "general",
        "📄 RAG (documents)":     "rag",
    }
    forced_mode = mode_map[answer_mode]

    if forced_mode == "research":
        st.caption("🌐 All queries will search DuckDuckGo + Wikipedia + Bing + Gemini")

    st.divider()
    healing_on = st.checkbox(
        "🔬 Self-Healing (Gemini Validator)", value=False,
        help="Validates answers using Gemini in your Brave browser. Adds ~30-60s per query.\nBrave must be open with --remote-debugging-port=9222"
    )

    if healing_on:
        st.markdown("**Heal which routes:**")
        heal_rag      = st.checkbox("📄 RAG answers",          value=True)
        heal_general  = st.checkbox("💡 General knowledge",    value=False)
        heal_research = st.checkbox("🔍 Research answers",     value=False)
        heal_routes   = (
            (["rag"]       if heal_rag      else []) +
            (["general"]   if heal_general  else []) +
            (["research"]  if heal_research else [])
        )
    else:
        heal_routes = []

    # Stats
    stats = healing_memory.get_stats()
    if stats["total"] > 0:
        st.divider()
        st.markdown("**📊 Learning Stats**")
        accuracy_color = "#44ff88" if stats["accuracy"] >= 80 else ("#ffaa00" if stats["accuracy"] >= 60 else "#ff5555")
        st.markdown(f"""<div class="stat-box">
Validated: <b>{stats['total']}</b> &nbsp;|&nbsp;
Correct: <b style="color:#44ff88">{stats['correct']}</b> &nbsp;|&nbsp;
Wrong: <b style="color:#ff5555">{stats['wrong']}</b><br>
Partial: <b style="color:#ffaa44">{stats['partial']}</b> &nbsp;|&nbsp;
Accuracy: <b style="color:{accuracy_color}">{stats['accuracy']}%</b>
</div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("**💡 Research examples**")
    st.caption("`research black holes`")
    st.caption("`research on gemini quantum physics`")
    st.caption("`research on all climate change`")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

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
        st.info("📭 No documents loaded — I can answer general questions and research the web!")
    else:
        tree_count = sum(len(t.all_nodes) for t in st.session_state.get("doc_trees", {}).values())
        st.success(f"📚 Documents loaded — {tree_count} tree sections indexed")
with col2:
    if memory_on:
        st.success("🧠 Memory ON")
    else:
        st.warning("Memory OFF")
with col3:
    if healing_on:
        st.success("🔬 Healing ON")

# ── Chat history ──────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        st.markdown(f'<div class="user-bubble">{entry["message"]}</div>', unsafe_allow_html=True)
    else:
        bubble = "healed-bubble" if entry.get("was_corrected") else "bot-bubble"
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

        if entry.get("gemini_feedback"):
            with st.expander("🔬 Gemini Validation Details"):
                st.markdown(f'<div class="gemini-card">{entry["gemini_feedback"]}</div>', unsafe_allow_html=True)

        if entry.get("meta"):
            st.markdown(f'<div class="meta-line">{entry["meta"]}</div>', unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────
user_input = st.chat_input("Ask anything — docs / general knowledge / research black holes ...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    history_for_llm  = st.session_state.chat_history[:-1] if memory_on else []

    start            = time.time()
    has_docs         = "vector_db" in st.session_state
    source_extracts  = []
    chosen_sources   = []
    answer           = ""
    retrieval_method = ""
    context_used     = ""
    gemini_feedback  = ""
    was_corrected    = False

    past_mistake = healing_memory.find_similar_mistake(user_input)
    lessons      = healing_memory.get_lessons()

    with st.status("Thinking...", expanded=True) as status:

        if past_mistake and past_mistake.get("verdict") in ["WRONG", "PARTIALLY_CORRECT"]:
            st.write(f"⚠️ Similar past mistake found — applying correction...")

        # ── Route ───────────────────────────────────────
        st.write("🔀 Detecting query type...")
        route = route_query(user_input, has_docs=has_docs, forced_mode=forced_mode)
        if route == "needs_classification":
            st.write("🧠 Classifying with LLM...")
            route = route_with_llm(llm, user_input, has_docs=has_docs)
        st.write(f"✅ Route → **{route.upper()}**")

        import re as _re

        # ── Research ─────────────────────────────────────
        if route == "research":
            if _re.search(r"\bresearch\b", user_input, _re.IGNORECASE):
                clean_query, source_override = parse_user_source_override(user_input)
                if not source_override:
                    clean_query = extract_research_query(user_input)
            else:
                clean_query    = user_input
                source_override = None

            st.write(f"🧠 Picking sources for: *{clean_query}*")
            chosen_sources = decide_sources(llm, clean_query, user_override=source_override)
            st.write(f"📡 Sources: **{', '.join(s.upper() for s in chosen_sources)}**")

            for src in chosen_sources:
                st.write(f"🌐 Scraping **{src}**...")

            result = run_research(clean_query, chosen_sources)

            if result["error"]:
                answer = f"⚠️ Research failed: {result['error']}"
            else:
                source_extracts = result.get("results", [])
                context_used    = result.get("context", "")
                st.write(f"✅ Scraped **{len(source_extracts)}** source(s)")
                for r in source_extracts:
                    st.write(f"  → {r['source'].upper()}: {len(r['content'])} chars")

                if not context_used.strip():
                    answer = "No content could be extracted from the sources."
                else:
                    # Pass scraped content through LLM for clean formatted answer
                    st.write("✍️ Synthesizing with phi3...")
                    synthesis_prompt = f"""You are a research assistant. Based on the web research below, answer the question clearly and thoroughly.

FORMAT YOUR ANSWER AS:
## [Topic Title]

**Overview**
2-3 sentence summary of who/what this is.

**Key Facts**
- Fact 1
- Fact 2
- Fact 3
(include dates, numbers, important details)

**Details**
2-3 paragraphs expanding on the most important aspects.

**Summary**
One final paragraph wrapping up the key takeaways.

---
QUESTION: {clean_query}

RESEARCH DATA:
{context_used[:6000]}

Write a well-structured, informative answer based on the research above. Do not copy text verbatim — synthesize and explain clearly."""

                    try:
                        answer = llm.invoke(synthesis_prompt).content
                    except Exception as e:
                        # Fallback: use heavy model
                        try:
                            answer = llm_heavy.invoke(synthesis_prompt).content
                        except Exception:
                            answer = context_used[:3000]

        # ── Web (legacy) ──────────────────────────────────
        elif route == "web":
            answer = handle_web_query(user_input)

        # ── Calc ─────────────────────────────────────────
        elif route == "calc":
            st.write("🔢 Solving...")
            history_str = format_history(history_for_llm, 2) if memory_on and history_for_llm else ""
            prompt = "Solve step by step:\n"
            if history_str:
                prompt += f"\nHistory:\n{history_str}"
            prompt += f"\nQuestion: {user_input}\nAnswer:"
            answer = llm.invoke(prompt).content

        # ── Chat ─────────────────────────────────────────
        elif route == "chat":
            answer = handle_chat_query(llm, user_input)

        # ── General ──────────────────────────────────────
        elif route == "general":
            st.write("💡 Using phi3 knowledge...")
            q = f"{user_input}\n\n[SYSTEM NOTE — {lessons}]" if lessons else user_input
            answer = run_general_with_memory(llm, q, history_for_llm)

        # ── RAG ──────────────────────────────────────────
        else:
            if use_hybrid and "hybrid_searcher" in st.session_state:
                class HybridRetriever:
                    def __init__(self, s): self.searcher = s
                    def invoke(self, q):
                        from config.settings import TOP_K_RETRIEVAL
                        return self.searcher.search(q, top_k=TOP_K_RETRIEVAL)
                retriever = HybridRetriever(st.session_state["hybrid_searcher"])
                st.write("🔀 Using Hybrid Search (BM25 + Vector)")
            else:
                retriever = get_mmr_retriever(st.session_state["vector_db"])
                st.write("🔢 Using MMR Vector Search")

            doc_trees = st.session_state.get("doc_trees", None)
            st.write("🔍 Retrieving context...")
            st.write("✍️ Generating answer...")

            result = run_rag_pipeline(
                llm, retriever, user_input,
                chat_history=history_for_llm,
                document_trees=doc_trees,
            )
            answer, retrieval_method = result if isinstance(result, tuple) else (result, "vector")
            st.write(f"✅ Retrieval: {retrieval_method}")

        # ── Self-Healing ──────────────────────────────────
        if healing_on and route in heal_routes and len(answer) > 20:
            st.write("🔬 Sending to Gemini for validation...")
            try:
                heal_result     = self_heal(
                    question        = user_input,
                    original_answer = answer,
                    memory          = healing_memory,
                    context         = context_used[:500],
                )
                answer          = heal_result["final_answer"]
                was_corrected   = heal_result["was_corrected"]
                gemini_feedback = heal_result["gemini_said"]
                verdict         = heal_result["verdict"]
                accuracy        = heal_result["accuracy_score"]
                label           = "✅ CORRECTED" if was_corrected else f"✅ Validated"
                st.write(f"{label}: {verdict} ({accuracy}/10)")
            except Exception as e:
                st.write(f"⚠️ Self-healing error: {e}")

        elapsed    = round(time.time() - start, 2)
        validation = validate_answer(answer)
        badge      = format_confidence_badge(validation["confidence"])
        healed_tag = " ✨ HEALED" if was_corrected else ""

        status.update(
            label=f"Done in {elapsed}s — {route.upper()}{healed_tag}",
            state="complete",
            expanded=False,
        )

    # Meta line
    parts = [f"⏱ {elapsed}s", f"Route: {route.upper()}"]
    if retrieval_method:
        parts.append(f"Retrieval: {retrieval_method}")
    if chosen_sources:
        parts.append(f"Sources: {' · '.join(s.upper() for s in chosen_sources)}")
    if was_corrected:
        parts.append("✨ CORRECTED BY GEMINI")
    parts.append(badge)

    st.session_state.chat_history.append({
        "role":            "bot",
        "message":         answer,
        "source_extracts": source_extracts,
        "meta":            " &nbsp;|&nbsp; ".join(parts),
        "was_corrected":   was_corrected,
        "gemini_feedback": gemini_feedback,
    })
    st.rerun()