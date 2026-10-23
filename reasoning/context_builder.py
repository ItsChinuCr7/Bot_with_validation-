import hashlib

from config.settings import MAX_CHARS_PER_DOC, FINAL_CONTEXT_DOCS

CONTEXT_CHAR_BUDGET = 2500

def build_context(
    docs,
    include_sources: bool = False,
    relevance_scores: dict | None = None,
):
    context_parts = []
    source_refs = []
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
        context_parts.append(f"[Source: {source} | Page: {page}]\n{content}")
        source_refs.append({
            "source_file": source,
            "page": page,
            "chunk_id": doc.metadata.get("chunk_id"),
            "chunk_hash": doc.metadata.get("chunk_hash"),
            "context_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "ingested_at": doc.metadata.get("ingested_at"),
            "embedding_model": doc.metadata.get("embedding_model"),
            "modified_at": doc.metadata.get("modified_at"),
            "section": doc.metadata.get("section"),
            "path": doc.metadata.get("path"),
            "relevance_score": (
                relevance_scores.get(
                    doc.metadata.get("chunk_id"),
                    relevance_scores.get(id(doc)),
                )
                if relevance_scores is not None else None
            ),
        })
        total_chars += len(content)

    context = "\n\n---\n\n".join(context_parts)
    return (context, source_refs) if include_sources else context
