# ============================================================
# retrieval/smart_retriever.py
#
# HYBRID RETRIEVER — combines:
#   1. Vector/Hybrid search  (semantic similarity)
#   2. Tree navigation       (structure-aware reasoning)
#
# The LLM decides which method to use based on query type:
#
#   "What does Section 3.2 say?"    → Tree navigation
#   "What is the payment deadline?" → Tree navigation
#   "Explain the methodology"       → Vector search
#   "What are the key findings?"    → Both merged
# ============================================================

from retrieval.tree_retriever import tree_retrieve
from retrieval.reranker import rerank_docs


QUERY_TYPE_PROMPT = """Classify this document question into ONE of these retrieval strategies:

"tree"   - Question refers to a specific section, heading, chapter, clause, or numbered item
           Examples: "What does section 3 say?", "What are the terms in clause 4?",
                     "Summarize chapter 2", "What is in the introduction?"

"vector" - Question asks about a concept, topic, or meaning across the whole document
           Examples: "What is the main methodology?", "Explain the key findings",
                     "What does the document say about risk?"

"both"   - Complex question that needs both structural and semantic search
           Examples: "Compare section 2 and 3", "What is the overall conclusion?"

Question: {query}

Answer with ONLY one word — "tree", "vector", or "both":"""


def classify_query_type(llm, query: str) -> str:
    """Ask LLM to classify query into retrieval strategy."""
    try:
        result = llm.invoke(
            QUERY_TYPE_PROMPT.format(query=query)
        ).content.strip().lower()

        if "tree" in result:
            return "tree"
        elif "vector" in result:
            return "vector"
        else:
            return "both"
    except Exception:
        return "vector"  # safe default


def smart_retrieve(
    llm,
    query: str,
    vector_retriever,
    document_tree=None,
    top_k: int = 5,
) -> tuple:
    """
    Hybrid retrieval — uses tree, vector, or both based on query type.

    Returns:
        (docs, retrieval_method_used)
    """

    # If no tree available, fall back to vector only
    if document_tree is None or not document_tree.all_nodes:
        docs = vector_retriever.invoke(query)
        return docs, "vector"

    # Classify query
    method = classify_query_type(llm, query)

    if method == "tree":
        docs = tree_retrieve(llm, document_tree, query, top_k=top_k)
        if not docs:  # fallback if tree fails
            docs = vector_retriever.invoke(query)
            method = "vector (tree fallback)"
        return docs, method

    elif method == "vector":
        docs = vector_retriever.invoke(query)
        return docs, method

    else:  # "both"
        tree_docs   = tree_retrieve(llm, document_tree, query, top_k=3)
        vector_docs = vector_retriever.invoke(query)

        # Merge — deduplicate by content fingerprint
        seen     = set()
        combined = []
        for doc in tree_docs + vector_docs:
            fp = doc.page_content[:80]
            if fp not in seen:
                seen.add(fp)
                combined.append(doc)

        # Re-rank the merged set
        if combined:
            combined = rerank_docs(query, combined, top_k=top_k)

        return combined, "hybrid (tree + vector)"