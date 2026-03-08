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
    route_query, route_with_llm, extract_research_query,
    handle_web_query, handle_calc_query, handle_chat_query,
)
from research.web_researcher import (
    decide_sources, parse_user_source_override,
    run_research, extract_research_query,
)
from evaluation.answer_validator import validate_answer, format_confidence_badge
from evaluation.self_healer import HealingMemory, self_heal
from config.settings import VECTOR_INDEX_PATH

st.set_page_config(page_title="AI Document Assistant", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.user-bubble {
    background-color:#2b313e; padding:12px 16px;
    border-radius:12px; margin-bottom:10px;
    text-align:right; color:white;
}
.bot-bubble {
    background-color:#1a1a2e; padding:12px 16px;
    border-radius:12px; margin-bottom:15px;
    color:#00ffcc; border-left:3px solid #00ffcc;
}
.healed-bubble {
    background-color:#0d2b1a; padding:12px 16px;
    border-radius:12px; margin-bottom:15px;
    color:#00ff88; border-left:3px solid #00ff88;
}
.source-extract {
    background:#0a0a1a; border:1px solid #1a1a3a;
    border-radius:8px; padding:10px 14px;
    font-size:12px; color:#8888cc; margin-bottom:8px;
}
.gemini-feedback {
    background:#1a1a0d; border:1px solid #444400;
    border-radius:8px; padding:10px 14px;
    font-size:12px; color:#cccc00; margin-bottom:10px;
}
.meta-line { font-size:12px; color:#888; margin-bottom:8px; }
.stat-box {
    background:#111; border:1px solid #333;
    border-radius:8px; padding:8px 12px;
    font-size:12px; color:#aaa;
}
</style>
""", unsafe_allow_html=True)

llm        = load_llm()
llm_heavy  = load_llm_heavy()
embeddings = load_embeddings()

if "healing_memory" not in st.session_state:
    st.session_state["healing_memory"] = HealingMemory()
healing_memory = st.session_state["healing_memory"]

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")
    st.divider()

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    use_hybrid     = st.checkbox("Hybrid Search (BM25 + Vector)", value=True)

    if st.button("Process Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            with st.status("Processing documents...", expanded=True) as status:
                st.write("Reading PDF files...")
                chunks, doc_trees = load_and_split_pdfs(uploaded_files)
                st.write(f"Extracted {len(chunks)} chunks")
                total_nodes = sum(len(t.all_nodes) for t in doc_trees.values())
                st.write(f"Document tree: {total_nodes} sections")
                st.write("Building vector index...")
                vector_db = create_vector_store(chunks, embeddings)
                st.session_state["vector_db"] = vector_db
                st.session_state["docs"]      = chunks
                st.session_state["doc_trees"] = doc_trees
                if use_hybrid:
                    st.write("Building BM25 index...")
                    st.session_state["hybrid_searcher"] = HybridSearcher(chunks, vector_db, alpha=0.5)
                status.update(label=f"Done! {len(chunks)} chunks + {total_nodes} nodes", state="complete")

    st.divider()
    memory_on  = st.checkbox("Conversation Memory", value=True)

    st.divider()
    st.markdown("**Answer Mode:**")
    answer_mode = st.radio(
        label="answer_mode",
        options=["Auto (smart detect)", "Research (web)", "General (phi3)", "RAG (documents)"],
        index=0,
        label_visibility="collapsed",
    )
    # Map to internal route name
    mode_map = {
        "Auto (smart detect)":  None,
        "Research (web)":       "research",
        "General (phi3)":       "general",
        "RAG (documents)":      "rag",
    }
    forced_mode = mode_map[answer_mode]

    if forced_mode == "research":
        st.caption("Every question will search DuckDuckGo + Gemini")
    healing_on = st.checkbox("Self-Healing (Gemini Validator)", value=False,
        help="Validates answers via Gemini. Adds 30-60s per query.")

    # Fixed route selection using radio buttons instead of multiselect
    st.markdown("**Heal which answers:**")
    heal_rag      = st.checkbox("RAG (document answers)", value=True)
    heal_general  = st.checkbox("General knowledge", value=False)
    heal_research = st.checkbox("Research answers", value=False)

    # Build heal_routes list from checkboxes
    heal_routes = []
    if heal_rag:      heal_routes.append("rag")
    if heal_general:  heal_routes.append("general")
    if heal_research: heal_routes.append("research")

    # Stats
    stats = healing_memory.get_stats()
    if stats["total"] > 0:
        st.divider()
        st.markdown("**Learning Stats:**")
        st.markdown(f"""<div class="stat-box">
Validated: {stats['total']} | Correct: {stats['correct']}<br>
Wrong: {stats['wrong']} | Partial: {stats['partial']}<br>
Accuracy: {stats['accuracy']}%
</div>""", unsafe_allow_html=True)

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown("**Research (DuckDuckGo + Gemini):**")
    st.caption("research black holes")
    st.caption("research on gemini quantum physics")

# ── Auto-load ─────────────────────────────────────────────
if "vector_db" not in st.session_state and os.path.exists(VECTOR_INDEX_PATH):
    try:
        st.session_state["vector_db"] = load_vector_store(embeddings)
    except Exception:
        pass

# ── Title ─────────────────────────────────────────────────
st.title("AI Document Intelligence Assistant")

col1, col2 = st.columns([3, 1])
with col1:
    if "vector_db" not in st.session_state:
        st.info("No docs loaded — I can answer general questions and research the web!")
    else:
        tree_count = sum(len(t.all_nodes) for t in st.session_state.get("doc_trees", {}).values())
        st.success(f"Documents loaded — {tree_count} tree sections | Ask about docs, general knowledge, or research.")
with col2:
    if memory_on:
        st.success("Memory ON")
    else:
        st.warning("Memory OFF")

# ── Chat History Display ──────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        st.markdown(f'<div class="user-bubble">{entry["message"]}</div>', unsafe_allow_html=True)
    else:
        bubble = "healed-bubble" if entry.get("was_corrected") else "bot-bubble"
        st.markdown(f'<div class="{bubble}">{entry["message"]}</div>', unsafe_allow_html=True)

        # Show extracted source data (collapsible)
        if entry.get("source_extracts"):
            with st.expander(f"📄 View extracted data from {len(entry['source_extracts'])} source(s)"):
                for extract in entry["source_extracts"]:
                    st.markdown(f"""<div class="source-extract">
<b>🔗 {extract['source'].upper()}</b> — <a href="{extract['url']}" target="_blank" style="color:#5599ff">{extract['url'][:70]}</a><br><br>
{extract['content'][:600]}{'...' if len(extract['content']) > 600 else ''}
</div>""", unsafe_allow_html=True)

        # Show Gemini validation feedback
        if entry.get("gemini_feedback"):
            with st.expander("🔬 Gemini Validation Details"):
                st.markdown(f'<div class="gemini-feedback">{entry["gemini_feedback"]}</div>',
                    unsafe_allow_html=True)

        if entry.get("meta"):
            st.markdown(f'<div class="meta-line">{entry["meta"]}</div>', unsafe_allow_html=True)

# ── Chat Input ────────────────────────────────────────────
user_input = st.chat_input("Ask anything — docs / general / research black holes...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    history_for_llm = st.session_state.chat_history[:-1] if memory_on else []

    start            = time.time()
    has_docs         = "vector_db" in st.session_state
    source_extracts  = []
    chosen_sources   = []
    answer           = ""
    retrieval_method = ""
    context_used     = ""
    gemini_feedback  = ""
    was_corrected    = False

    # Check past mistakes before answering
    past_mistake = healing_memory.find_similar_mistake(user_input)
    lessons      = healing_memory.get_lessons()

    with st.status("Processing...", expanded=True) as status:

        if past_mistake and past_mistake.get("verdict") in ["WRONG", "PARTIALLY_CORRECT"]:
            st.write(f"⚠️ Similar past mistake found — applying learned correction...")

        # Route
        st.write("Detecting query type...")
        route = route_query(user_input, has_docs=has_docs, forced_mode=forced_mode)
        if route == "needs_classification":
            st.write("Classifying query...")
            route = route_with_llm(llm, user_input, has_docs=has_docs)
        st.write(f"Route: {route.upper()}")

        # ── Research ────────────────────────────────────────
        if route == "research":
            # Only strip "research" keyword if user actually typed it
            import re
            if re.search(r"\bresearch\b", user_input, re.IGNORECASE):
                clean_query, source_override = parse_user_source_override(user_input)
                if not source_override:
                    clean_query = extract_research_query(user_input)
            else:
                # Mode was forced from sidebar — use full query as-is
                clean_query    = user_input
                source_override = None

            st.write(f"Deciding sources for: {clean_query}")
            chosen_sources = decide_sources(llm, clean_query, user_override=source_override)
            st.write(f"Sources selected: {', '.join(chosen_sources).upper()}")

            for src in chosen_sources:
                st.write(f"Scraping {src}...")

            result = run_research(clean_query, chosen_sources)

            if result["error"]:
                answer = f"Research failed: {result['error']}"
            elif not result["context"]:
                answer = "Could not extract content. Try a more specific topic."
            else:
                source_extracts = result["results"]  # structured per source
                context_used    = result["context"]
                st.write(f"Scraped {len(source_extracts)} source(s) — data extracted")

                # Show preview of what was extracted
                for r in source_extracts:
                    st.write(f"  ✅ {r['source'].upper()}: {len(r['content'])} chars extracted")

                st.write("Sending all extracted data to Gemini for final synthesis...")
                answer = result["final_answer"] or result["context"][:4000]

        # ── Web ─────────────────────────────────────────────
        elif route == "web":
            answer = handle_web_query(user_input)

        # ── Calc ────────────────────────────────────────────
        elif route == "calc":
            st.write("Solving...")
            history_str = format_history(history_for_llm, 2) if memory_on and history_for_llm else ""
            prompt = "Solve step by step"
            if history_str:
                prompt += f"\n\nHistory:\n{history_str}"
            prompt += f"\n\nQuestion: {user_input}\n\nAnswer:"
            answer = llm.invoke(prompt).content

        # ── Chat ────────────────────────────────────────────
        elif route == "chat":
            answer = handle_chat_query(llm, user_input)

        # ── General ─────────────────────────────────────────
        elif route == "general":
            st.write("Searching phi3 knowledge...")
            q = f"{user_input}\n\n[SYSTEM NOTE — {lessons}]" if lessons else user_input
            answer = run_general_with_memory(llm, q, history_for_llm)

        # ── RAG ─────────────────────────────────────────────
        else:
            st.write("Rewriting query...")
            if use_hybrid and "hybrid_searcher" in st.session_state:
                class HybridRetriever:
                    def __init__(self, searcher):
                        self.searcher = searcher
                    def invoke(self, query):
                        from config.settings import TOP_K_RETRIEVAL
                        return self.searcher.search(query, top_k=TOP_K_RETRIEVAL)
                retriever = HybridRetriever(st.session_state["hybrid_searcher"])
                st.write("Using Hybrid Search (BM25 + Vector)")
            else:
                retriever = get_mmr_retriever(st.session_state["vector_db"])
                st.write("Using MMR Vector Search")

            doc_trees = st.session_state.get("doc_trees", None)
            st.write("Retrieving content...")
            st.write("Generating answer...")

            result = run_rag_pipeline(
                llm, retriever, user_input,
                chat_history=history_for_llm,
                document_trees=doc_trees,
            )
            answer, retrieval_method = result if isinstance(result, tuple) else (result, "vector")
            st.write(f"Retrieval: {retrieval_method}")

        # ── Self-Healing ─────────────────────────────────────
        should_heal = healing_on and route in heal_routes and len(answer) > 20

        if should_heal:
            st.write(f"🔬 Attaching to Brave → sending to Gemini for validation...")
            try:
                heal_result = self_heal(
                    question=user_input,
                    original_answer=answer,
                    memory=healing_memory,
                    context=context_used[:500],
                )
                answer          = heal_result["final_answer"]
                was_corrected   = heal_result["was_corrected"]
                gemini_feedback = heal_result["gemini_said"]
                verdict         = heal_result["verdict"]
                accuracy        = heal_result["accuracy_score"]

                if was_corrected:
                    st.write(f"✅ Answer CORRECTED ({verdict}, {accuracy}/10)")
                else:
                    st.write(f"✅ Validated: {verdict} ({accuracy}/10)")

            except Exception as e:
                st.write(f"⚠️ Self-healing error: {e}")

        elapsed    = round(time.time() - start, 2)
        validation = validate_answer(answer)
        badge      = format_confidence_badge(validation["confidence"])

        status.update(
            label=f"Done in {elapsed}s — {route.upper()}" + (" ✨HEALED" if was_corrected else ""),
            state="complete",
            expanded=False,
        )

    meta_parts = [f"{elapsed}s", f"Route: {route}"]
    if retrieval_method:
        meta_parts.append(f"Retrieval: {retrieval_method}")
    if chosen_sources:
        meta_parts.append(f"Sources: {', '.join(chosen_sources)}")
    if was_corrected:
        meta_parts.append("CORRECTED BY GEMINI")
    meta_parts.append(badge)
    meta_str = " | ".join(meta_parts)

    st.session_state.chat_history.append({
        "role":           "bot",
        "message":        answer,
        "source_extracts": source_extracts,
        "meta":           meta_str,
        "was_corrected":  was_corrected,
        "gemini_feedback": gemini_feedback,
    })
    st.rerun()