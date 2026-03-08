from config.settings import MAX_CHARS_PER_DOC, FINAL_CONTEXT_DOCS

CONTEXT_CHAR_BUDGET = 2500

def build_context(docs):
    context_parts = []
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
        total_chars += len(content)

    return "\n\n---\n\n".join(context_parts)
