# ============================================================
# reasoning/rag_pipeline.py
# RAG pipeline with smart retrieval (vector + tree hybrid)
# ============================================================

from retrieval.reranker import rerank_docs
from reasoning.context_builder import build_context
from reasoning.query_rewriter import rewrite_query
from retrieval.smart_retriever import smart_retrieve
from config.settings import MAX_REASONING_DEPTH
import logging

logger = logging.getLogger(__name__)

from prompts.templates import (
    RAG_PROMPT,
    RAG_PROMPT_NO_HISTORY,
    GENERAL_PROMPT_WITH_HISTORY,
    QUERY_REWRITE_WITH_HISTORY_PROMPT,
    FALLBACK_SEARCH_PROMPT,
)

MEMORY_TURNS = 3


def format_history(chat_history: list, max_turns: int = MEMORY_TURNS) -> str:
    if not chat_history:
        return "None"
    recent = chat_history[-(max_turns * 2):]
    lines  = []
    for entry in recent:
        role = "User" if entry["role"] == "user" else "Assistant"
        msg  = entry["message"]
        if role == "Assistant" and len(msg) > 400:
            msg = msg[:400] + "..."
        lines.append(f"{role}: {msg}")
    return "\n".join(lines)


def run_rag_pipeline(
    llm,
    retriever,
    query: str,
    chat_history: list = None,
    document_trees: dict = None,
    depth: int = 0,
) -> tuple:
    """
    Full RAG pipeline with smart hybrid retrieval.

    Returns:
        (answer, retrieval_method_used)
    """

    # Step 1: Rewrite query
    if depth == 0 and chat_history:
        history_str = format_history(chat_history, max_turns=1)
        rewrite_prompt = QUERY_REWRITE_WITH_HISTORY_PROMPT.format(
            history=history_str, question=query
        )
        try:
            search_query = llm.invoke(rewrite_prompt).content.strip()
            if not search_query or len(search_query) > 200:
                search_query = query
        except Exception:
            search_query = query
    elif depth == 0:
        search_query = rewrite_query(llm, query)
    else:
        search_query = query

    # Step 2: Smart retrieval — tree or vector or both
    # Merge all trees into one combined tree for multi-doc support
    combined_tree = None
    if document_trees:
        from ingestion.document_tree import DocumentTree
        combined_tree = DocumentTree("combined")
        for tree in document_trees.values():
            combined_tree.all_nodes.extend(tree.all_nodes)
            combined_tree.root.children.extend(tree.root.children)

    docs, method = smart_retrieve(
        llm=llm,
        query=search_query,
        vector_retriever=retriever,
        document_tree=combined_tree,
    )

    if not docs:
        return "No relevant documents found. Please upload and process PDFs first.", "none"

    # Step 3: Re-rank (skip if tree already did precise navigation)
    if "tree" not in method:
        docs = rerank_docs(search_query, docs)

    # Step 4: Build context
    context = build_context(docs)

    # Step 5: Generate answer
    history_str = format_history(chat_history) if chat_history else ""
    if history_str and history_str != "None":
        prompt = RAG_PROMPT.format(
            history=history_str, context=context, question=query
        )
    else:
        prompt = RAG_PROMPT_NO_HISTORY.format(context=context, question=query)

    answer = llm.invoke(prompt).content.strip()

    # Step 6: Recursive retry if uncertain
    if depth < MAX_REASONING_DEPTH:
        for signal in ["i could not find", "not in the context"]:
            if signal in answer.lower():
                fallback = FALLBACK_SEARCH_PROMPT.format(question=query)
                new_query = llm.invoke(fallback).content.strip()
                return run_rag_pipeline(
                    llm, retriever, new_query, chat_history, document_trees, depth + 1
                )

    return answer, method


def run_general_with_memory(llm, query: str, chat_history: list = None) -> str:
    history_str = format_history(chat_history) if chat_history else ""
    if history_str and history_str != "None":
        prompt = GENERAL_PROMPT_WITH_HISTORY.format(
            history=history_str, question=query
        )
    else:
        prompt = f"Answer clearly from your own knowledge:\n\n{query}"
    return llm.invoke(prompt).content.strip()