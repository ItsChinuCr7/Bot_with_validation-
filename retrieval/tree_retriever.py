# ============================================================
# retrieval/tree_retriever.py
#
# Vectorless Reasoning-First Retrieval
#
# Instead of similarity search, the LLM:
# 1. Reads the document outline (table of contents)
# 2. Reasons about which sections are most relevant
# 3. Navigates directly to those sections
# 4. Extracts grounded content
#
# This is structure-aware retrieval — no embeddings needed.
# ============================================================

from ingestion.document_tree import DocumentTree, TreeNode
from guardrails.input_validator import screen_untrusted_text


# LLM prompt to navigate the document tree
TREE_NAVIGATION_PROMPT = """You are a document navigation expert.

Given a document outline and a question, identify which sections most likely contain the answer.

Document Outline:
{outline}

Question: {question}

Instructions:
- Look at section titles and levels carefully
- Pick the 3 most relevant section titles that likely contain the answer
- Consider parent sections too (they may contain summary info)
- Return ONLY a Python list of exact section titles, nothing else
- Example format: ["Section Title 1", "Section Title 2", "Section Title 3"]

Most relevant sections:"""


def tree_retrieve(
    llm,
    document_tree: DocumentTree,
    query: str,
    top_k: int = 3,
    guardrail_event=None,
) -> list:
    """
    Use LLM reasoning to navigate document tree and retrieve relevant sections.

    Args:
        llm:           Language model for reasoning
        document_tree: Built DocumentTree object
        query:         User question
        top_k:         Max sections to retrieve

    Returns:
        List of LangChain-compatible Document objects
    """
    from langchain_core.documents import Document

    if not document_tree.all_nodes:
        return []

    # Titles and section bodies are document-sourced and are screened before
    # the navigation model sees them.
    safe_nodes = []
    rejected_nodes = 0
    for node in document_tree.all_nodes:
        safe_title, title_rejected = screen_untrusted_text(node.title)
        safe_content, content_rejected = screen_untrusted_text(node.content)
        if title_rejected or content_rejected:
            rejected_nodes += 1
            continue
        safe_nodes.append((node, safe_title, safe_content))
    if guardrail_event:
        guardrail_event(
            "document_tree_screening",
            "filtered" if rejected_nodes else "passed",
            "prompt_injection" if rejected_nodes else "none",
            count=rejected_nodes,
        )
    if not safe_nodes:
        return []

    # Build the outline only from the screened node titles.
    outline = "\n".join(
        f"[L{node.level}] {safe_title} (p.{node.page})"
        for node, safe_title, _ in safe_nodes
    )

    # Truncate outline if too long for LLM context
    if len(outline) > 3000:
        outline = outline[:3000] + "\n... (truncated)"

    # Step 2: Ask LLM to navigate to relevant sections
    prompt   = TREE_NAVIGATION_PROMPT.format(outline=outline, question=query)
    response = llm.invoke(prompt).content.strip()

    # Step 3: Parse LLM response to get section titles
    selected_titles = _parse_title_list(response)

    # Step 4: Find matching nodes in tree
    matched_nodes = []

    if selected_titles:
        for title in selected_titles:
            # Exact match first
            for node, safe_title, _ in safe_nodes:
                if safe_title.strip().lower() == title.strip().lower():
                    matched_nodes.append(node)
                    break

        # Fuzzy fallback — partial match
        if not matched_nodes:
            keywords = [t.split()[0] for t in selected_titles if t.split()]
            matched_nodes = [
                node for node, safe_title, _ in safe_nodes
                if any(keyword.lower() in safe_title.lower() for keyword in keywords)
            ]

    # Fallback — return first few nodes if LLM navigation failed
    if not matched_nodes:
        matched_nodes = [node for node, _, _ in safe_nodes[:top_k]]

    # Step 5: Build Document objects from matched nodes
    results = []
    safe_by_id = {id(node): (title, content) for node, title, content in safe_nodes}
    for node in matched_nodes[:top_k]:
        title, node_content = safe_by_id[id(node)]
        content_parts = [f"[{title}]\n{node_content}"]
        for child in node.children:
            if id(child) in safe_by_id:
                child_title, child_content = safe_by_id[id(child)]
                content_parts.append(f"[{child_title}]\n{child_content}")
        content = "\n\n".join(content_parts)
        doc = Document(
            page_content=content[:1500],  # cap per section
            metadata={
                "source_file": node.source,
                "page":        node.page,
                "section":     node.title,
                "path":        node.full_path(),
                "retrieval":   "tree",
            }
        )
        results.append(doc)

    return results


def _parse_title_list(response: str) -> list:
    """Parse LLM response to extract list of section titles."""
    import ast
    import re

    # Try direct Python list parse
    try:
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            return ast.literal_eval(match.group())
    except Exception:
        pass

    # Fallback — extract quoted strings
    titles = re.findall(r'"([^"]+)"', response)
    if titles:
        return titles

    # Last resort — split by newlines
    lines = [l.strip().strip('"-*•').strip() for l in response.split('\n') if l.strip()]
    return [l for l in lines if 3 < len(l) < 100][:3]