# ============================================================
# inference/inference_engine.py
# Central inference orchestrator.
# All query handling logic lives here — app.py just calls
# run_inference() and gets back a clean result dict.
# ============================================================

import re
import logging
from data.cache.cache_manager import get_cached, set_cache

logger = logging.getLogger(__name__)
from prompts.templates import (
    RESEARCH_SYNTHESIS_PROMPT,
    CALC_PROMPT,
)
from agents.agent_router import (
    route_query,
    route_with_llm,
    handle_web_query,
    handle_chat_query,
)
from reasoning.rag_pipeline import (
    run_rag_pipeline,
    run_general_with_memory,
    format_history,
)
from research.web_researcher import (
    decide_sources,
    parse_user_source_override,
    run_research,
    extract_research_query,
)
from retrieval.retriever import get_mmr_retriever
from config.settings import TOP_K_RETRIEVAL


# ── Hybrid Retriever wrapper ───────────────────────────────
class HybridRetriever:
    def __init__(self, searcher):
        self.searcher = searcher

    def invoke(self, q):
        return self.searcher.search(q, top_k=TOP_K_RETRIEVAL)


# ── Main entry point ───────────────────────────────────────
def run_inference(
    query: str,
    llm,
    llm_heavy,
    session_state: dict,
    memory_on: bool = True,
    forced_mode: str = None,
    use_hybrid: bool = True,
    progress_callback=None,   # optional fn(msg: str) for UI updates
) -> dict:
    """
    Central inference function. Handles all routes and returns a unified result dict.

    Returns:
        {
            "answer":           str,
            "route":            str,
            "retrieval_method": str,
            "chosen_sources":   list,
            "source_extracts":  list,
            "context_used":     str,
            "was_corrected":    bool,
            "gemini_feedback":  str,
        }
    """

    def log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    has_docs       = "vector_db" in session_state
    history_for_llm = session_state.get("chat_history", [])[:-1] if memory_on else []

    # Init result
    result = {
        "answer":           "",
        "route":            "",
        "retrieval_method": "",
        "chosen_sources":   [],
        "source_extracts":  [],
        "context_used":     "",

    }

    # ── Route ─────────────────────────────────────────────
    log("🔀 Detecting query type...")
    route = route_query(query, has_docs=has_docs, forced_mode=forced_mode)
    if route == "needs_classification":
        log("🧠 Classifying with LLM...")
        route = route_with_llm(llm, query, has_docs=has_docs)
    log(f"✅ Route → **{route.upper()}**")
    result["route"] = route

    # ── Cache check ────────────────────────────────────────
    cached = get_cached(query, route)
    if cached:
        log("⚡ Cache hit — returning instant answer")
        cached["route"] = route
        return cached

    # ── Research ───────────────────────────────────────────
    if route == "research":
        if re.search(r"\bresearch\b", query, re.IGNORECASE):
            clean_query, source_override = parse_user_source_override(query)
            if not source_override:
                clean_query = extract_research_query(query)
        else:
            clean_query     = query
            source_override = None

        log(f"🧠 Picking sources for: *{clean_query}*")
        chosen_sources = decide_sources(llm, clean_query, user_override=source_override)
        log(f"📡 Sources: **{', '.join(s.upper() for s in chosen_sources)}**")
        result["chosen_sources"] = chosen_sources

        for src in chosen_sources:
            log(f"🌐 Scraping **{src}**...")

        research_result = run_research(clean_query, chosen_sources)

        if research_result["error"]:
            result["answer"] = f"⚠️ Research failed: {research_result['error']}"
        else:
            source_extracts = research_result.get("results", [])
            context_used    = research_result.get("context", "")
            result["source_extracts"] = source_extracts
            result["context_used"]    = context_used

            log(f"✅ Scraped **{len(source_extracts)}** source(s)")
            for r in source_extracts:
                log(f"  → {r['source'].upper()}: {len(r['content'])} chars")

            if not context_used.strip():
                result["answer"] = "No content could be extracted from the sources."
            else:
                log("✍️ Synthesizing with LLM...")
                synthesis_prompt = RESEARCH_SYNTHESIS_PROMPT.format(
                    question=clean_query,
                    context=context_used[:6000],
                )
                try:
                    result["answer"] = llm.invoke(synthesis_prompt).content
                except Exception:
                    try:
                        result["answer"] = llm_heavy.invoke(synthesis_prompt).content
                    except Exception:
                        result["answer"] = context_used[:3000]

    # ── Web (legacy) ───────────────────────────────────────
    elif route == "web":
        result["answer"] = handle_web_query(query)

    # ── Calc ───────────────────────────────────────────────
    elif route == "calc":
        log("🔢 Solving...")
        history_str = format_history(history_for_llm, 2) if memory_on and history_for_llm else ""
        hist_block  = f"History:\n{history_str}\n\n" if history_str else ""
        prompt      = CALC_PROMPT.format(history=hist_block, question=query)
        result["answer"] = llm.invoke(prompt).content

    # ── Chat ───────────────────────────────────────────────
    elif route == "chat":
        result["answer"] = handle_chat_query(llm, query)

    # ── General ────────────────────────────────────────────
    elif route == "general":
        log("💡 Using LLM knowledge...")
        q = query
        result["answer"] = run_general_with_memory(llm, q, history_for_llm)

    # ── RAG ────────────────────────────────────────────────
    else:
        if use_hybrid and "hybrid_searcher" in session_state:
            retriever = HybridRetriever(session_state["hybrid_searcher"])
            log("🔀 Using Hybrid Search (BM25 + Vector)")
        else:
            retriever = get_mmr_retriever(session_state["vector_db"])
            log("🔢 Using MMR Vector Search")

        doc_trees = session_state.get("doc_trees", None)
        log("🔍 Retrieving context...")
        log("✍️ Generating answer...")

        rag_result = run_rag_pipeline(
            llm, retriever, query,
            chat_history=history_for_llm,
            document_trees=doc_trees,
        )
        answer, retrieval_method = rag_result if isinstance(rag_result, tuple) else (rag_result, "vector")
        result["answer"]           = answer
        result["retrieval_method"] = retrieval_method
        log(f"✅ Retrieval: {retrieval_method}")

    # ── Save to cache ──────────────────────────────────────
    set_cache(query, route, result)

    return result