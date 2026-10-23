# ============================================================
# reasoning/rag_pipeline.py
# RAG pipeline with smart retrieval (vector + tree hybrid)
# ============================================================

import hashlib

from retrieval.reranker import MIN_RELEVANCE_SCORE, score_docs
from reasoning.context_builder import build_context
from reasoning.query_rewriter import rewrite_query
from retrieval.smart_retriever import smart_retrieve
from config.settings import MAX_REASONING_DEPTH
import logging
from guardrails.input_validator import (
    SAFE_BLOCKED_RESPONSE,
    GuardrailCheckError,
    InputValidationError,
    OutputValidationError,
    document_diagnostic,
    enforce_output,
    screen_documents,
    screen_untrusted_text,
    validate_input,
)

logger = logging.getLogger(__name__)

from prompts.templates import (
    GENERAL_PROMPT_WITH_HISTORY,
    QUERY_REWRITE_WITH_HISTORY_PROMPT,
    FALLBACK_SEARCH_PROMPT,
)
from prompts.registry import ACTIVE_RAG_PROMPT, get_prompt_version

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
    return_metadata: bool = False,
    guardrail_event=None,
    diagnostic_callback=None,
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
    try:
        search_query = validate_input(search_query).text
    except InputValidationError as exc:
        if guardrail_event:
            guardrail_event("query_rewrite", "blocked", exc.reason_code)
        raise
    except Exception as exc:
        raise GuardrailCheckError("Rewritten query validation failed.") from exc

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
        guardrail_event=guardrail_event,
    )
    retrieved_count = len(docs)
    retrieval_diagnostics = []
    try:
        docs, rejected_documents = screen_documents(
            docs, diagnostics=retrieval_diagnostics
        )
    except GuardrailCheckError:
        if guardrail_event:
            guardrail_event("document_screening", "error", "check_failed")
        _report_retrieval_diagnostics(
            diagnostic_callback, search_query, retrieval_diagnostics
        )
        raise
    if guardrail_event:
        guardrail_event(
            "document_screening",
            "filtered" if rejected_documents else "passed",
            "prompt_injection" if rejected_documents else "none",
            count=rejected_documents,
        )

    if not docs:
        _report_retrieval_diagnostics(
            diagnostic_callback, search_query, retrieval_diagnostics
        )
        message = (
            "Retrieved content did not pass safety checks. "
            "Please review the source documents or ask another question."
            if rejected_documents else
            "No relevant documents found. Please upload and process PDFs first."
        )
        result = (message, "none")
        return (
            (*result, retrieved_count, 0, "", [])
            if return_metadata else result
        )

    # Step 3: Score all screened chunks and discard weak matches.
    try:
        ranked_docs = score_docs(search_query, docs)
    except GuardrailCheckError:
        if guardrail_event:
            guardrail_event("retrieval_relevance", "error", "check_failed")
        _report_retrieval_diagnostics(
            diagnostic_callback, search_query, retrieval_diagnostics
        )
        raise
    relevance_scores = {}
    relevant_docs = []
    for score, doc in ranked_docs:
        accepted = score >= MIN_RELEVANCE_SCORE
        metadata = getattr(doc, "metadata", {})
        diagnostic = document_diagnostic(
            doc,
            score=round(score, 5),
            rejection_reason=None if accepted else "below_relevance_threshold",
        )
        retrieval_diagnostics.append(diagnostic)
        if accepted:
            relevant_docs.append(doc)
            relevance_scores[metadata.get("chunk_id", id(doc))] = score
    docs = relevant_docs
    _report_retrieval_diagnostics(
        diagnostic_callback, search_query, retrieval_diagnostics
    )
    if guardrail_event:
        guardrail_event(
            "retrieval_relevance",
            "filtered" if len(docs) < retrieved_count - rejected_documents else "passed",
            (
                "below_relevance_threshold"
                if len(docs) < retrieved_count - rejected_documents else "none"
            ),
            count=retrieved_count - rejected_documents - len(docs),
        )
    if not docs:
        message = "No sufficiently relevant documents were found for this question."
        result = (message, "none")
        return (*result, retrieved_count, 0, "", []) if return_metadata else result

    # Step 4: Build context
    context, source_refs = build_context(
        docs,
        include_sources=True,
        relevance_scores=relevance_scores,
    )

    # Step 5: Generate answer
    history_str = format_history(chat_history) if chat_history else ""
    prompt_config = get_prompt_version(ACTIVE_RAG_PROMPT)
    if history_str and history_str != "None":
        prompt = prompt_config["template"].format(
            history=history_str, context=context, question=query
        )
    else:
        prompt = prompt_config["template_no_history"].format(
            context=context, question=query
        )

    answer = llm.invoke(prompt).content.strip()
    try:
        enforce_output(answer)
    except OutputValidationError as exc:
        if guardrail_event:
            guardrail_event("output_validation", "blocked", exc.reason_code)
        result = (SAFE_BLOCKED_RESPONSE, "blocked")
        return (
            *result, retrieved_count, 0, "", []
        ) if return_metadata else result

    # Step 6: Recursive retry if uncertain
    if depth < MAX_REASONING_DEPTH:
        for signal in ["i could not find", "not in the context"]:
            if signal in answer.lower():
                fallback = FALLBACK_SEARCH_PROMPT.format(question=query)
                new_query = llm.invoke(fallback).content.strip()
                try:
                    new_query = validate_input(new_query).text
                except InputValidationError as exc:
                    if guardrail_event:
                        guardrail_event("retry_query", "blocked", exc.reason_code)
                    return (
                        (SAFE_BLOCKED_RESPONSE, "blocked", retrieved_count, 0, "", [])
                        if return_metadata else (SAFE_BLOCKED_RESPONSE, "blocked")
                    )
                except Exception as exc:
                    raise GuardrailCheckError("Retry query validation failed.") from exc
                return run_rag_pipeline(
                    llm, retriever, new_query, chat_history, document_trees, depth + 1,
                    return_metadata=return_metadata,
                    guardrail_event=guardrail_event,
                    diagnostic_callback=diagnostic_callback,
                )

    result = (answer, method)
    return (
        *result, retrieved_count, len(source_refs), context, source_refs
    ) if return_metadata else result


def _report_retrieval_diagnostics(callback, query: str, chunks: list[dict]) -> None:
    if callback is None:
        return
    callback({
        "query_fingerprint": hashlib.sha256(
            query.encode("utf-8")
        ).hexdigest()[:16],
        "relevance_threshold": MIN_RELEVANCE_SCORE,
        "chunks": chunks,
    })


def run_general_with_memory(llm, query: str, chat_history: list = None) -> str:
    history_str = format_history(chat_history) if chat_history else ""
    if history_str and history_str != "None":
        prompt = GENERAL_PROMPT_WITH_HISTORY.format(
            history=history_str, question=query
        )
    else:
        prompt = f"Answer clearly from your own knowledge:\n\n{query}"
    return llm.invoke(prompt).content.strip()