# ============================================================
# mcp_server.py
# MCP Server for AI Document Intelligence Assistant
# Exposes your RAG, General LLM, and Web Research as tools
# that Claude Desktop can call directly.
#
# Run: python mcp_server.py
# ============================================================

import sys
import os

# Make sure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from config.logging_config import setup_logging

setup_logging("INFO")

# ── Init MCP server ────────────────────────────────────────
mcp = FastMCP("AI Document Intelligence Assistant")

# ── Lazy-load heavy components once ───────────────────────
_llm         = None
_llm_heavy   = None
_embeddings  = None
_vector_db   = None
_doc_trees   = None
_chunks      = None


def _get_llm():
    global _llm
    if _llm is None:
        from models.llm import load_llm
        _llm = load_llm()
    return _llm


def _get_llm_heavy():
    global _llm_heavy
    if _llm_heavy is None:
        from models.llm import load_llm_heavy
        _llm_heavy = load_llm_heavy()
    return _llm_heavy


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        from models.embeddings import load_embeddings
        _embeddings = load_embeddings()
    return _embeddings


def _get_vector_db():
    global _vector_db, _doc_trees, _chunks
    if _vector_db is None:
        from retrieval.vector_store import load_vector_store
        from ingestion.document_tree import DocumentTree
        try:
            _vector_db = load_vector_store(_get_embeddings())
        except Exception:
            _vector_db = None
    return _vector_db


# ── Tool 1: Query Documents (RAG) ──────────────────────────
@mcp.tool()
def query_documents(question: str) -> str:
    """
    Query your uploaded PDF documents using RAG (Retrieval-Augmented Generation).
    Use this when the question is about specific documents, reports, contracts,
    research papers, or any file that has been uploaded to the system.

    Args:
        question: The question to ask about the documents.

    Returns:
        Answer grounded in the document content, with source references.
    """
    vector_db = _get_vector_db()
    if vector_db is None:
        return (
            "No documents are currently loaded in the system. "
            "Please upload PDFs via the Streamlit UI first, then restart the MCP server."
        )

    try:
        from reasoning.rag_pipeline import run_rag_pipeline
        from retrieval.retriever import get_mmr_retriever

        retriever = get_mmr_retriever(vector_db)
        result    = run_rag_pipeline(
            _get_llm(),
            retriever,
            question,
            chat_history=[],
            document_trees=_doc_trees,
        )
        answer, method = result if isinstance(result, tuple) else (result, "vector")
        return f"{answer}\n\n---\n*Retrieved via: {method}*"

    except Exception as e:
        return f"RAG query failed: {e}"


# ── Tool 2: Ask General Question ───────────────────────────
@mcp.tool()
def ask_general(question: str) -> str:
    """
    Ask a general knowledge question answered by the local phi3/gemma LLM.
    Use this for science, history, coding, explanations, definitions,
    or any question that doesn't require uploaded documents.

    Args:
        question: Any general knowledge question.

    Returns:
        A detailed answer from the local LLM.
    """
    try:
        from reasoning.rag_pipeline import run_general_with_memory
        return run_general_with_memory(_get_llm(), question, history=[])
    except Exception as e:
        return f"General query failed: {e}"


# ── Tool 3: Web Research ────────────────────────────────────
@mcp.tool()
def web_research(topic: str, sources: str = "auto") -> str:
    """
    Research any topic by searching the web in real time.
    Scrapes DuckDuckGo, Wikipedia, and Bing, then synthesizes
    the results into a clean structured answer via the local LLM.

    Args:
        topic:   The topic or question to research.
        sources: Which sources to use. Options: "auto", "wikipedia",
                 "duckduckgo", "bing", "all". Default is "auto" (LLM picks best sources).

    Returns:
        A synthesized research answer with ## headers and bullet points.
    """
    try:
        from research.web_researcher import run_research, decide_sources
        from prompts.templates import RESEARCH_SYNTHESIS_PROMPT

        # Determine sources
        valid = ["duckduckgo", "wikipedia", "bing"]
        if sources == "auto":
            chosen = decide_sources(_get_llm(), topic, user_override=None)
        elif sources == "all":
            chosen = valid
        elif sources in valid:
            chosen = [sources]
        else:
            chosen = decide_sources(_get_llm(), topic, user_override=None)

        result = run_research(topic, chosen)

        if result["error"]:
            return f"Research failed: {result['error']}"

        context = result.get("context", "")
        if not context.strip():
            return "No content could be extracted from the sources."

        # Synthesize via LLM
        prompt = RESEARCH_SYNTHESIS_PROMPT.format(
            question=topic,
            context=context[:6000],
        )
        answer = _get_llm().invoke(prompt).content
        sources_used = [r["source"].upper() for r in result.get("results", [])]
        return f"{answer}\n\n---\n*Sources: {', '.join(sources_used)}*"

    except Exception as e:
        return f"Web research failed: {e}"


# ── Tool 4: List Loaded Documents ──────────────────────────
@mcp.tool()
def list_documents() -> str:
    """
    List all PDF documents currently loaded in the system.
    Use this to check what documents are available before querying them.

    Returns:
        List of loaded document names and chunk counts.
    """
    vector_db = _get_vector_db()
    if vector_db is None:
        return "No documents loaded. Upload PDFs via the Streamlit UI first."

    try:
        # Get doc metadata from FAISS index
        if hasattr(vector_db, "docstore") and hasattr(vector_db.docstore, "_dict"):
            docs = vector_db.docstore._dict.values()
            from collections import Counter
            sources = Counter(
                d.metadata.get("source_file", "unknown")
                for d in docs
            )
            lines = [f"📄 {name}: {count} chunks" for name, count in sources.items()]
            total = sum(sources.values())
            return (
                f"Loaded documents ({total} total chunks):\n\n" +
                "\n".join(lines)
            )
        return "Documents are loaded but metadata is unavailable."
    except Exception as e:
        return f"Could not list documents: {e}"


# ── Run server ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  AI Document Intelligence MCP Server")
    print("  Tools: query_documents, ask_general,")
    print("         web_research, list_documents")
    print("=" * 50)
    mcp.run()
