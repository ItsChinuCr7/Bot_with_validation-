REWRITE_PROMPT = """Rewrite this question into a keyword-dense search query.
Rules: be specific, remove filler words, max 15 words, return ONLY the query.

Question: {query}
Rewritten query:"""

def rewrite_query(llm, query):
    try:
        result = llm.invoke(REWRITE_PROMPT.format(query=query)).content.strip()
        if not result or len(result) > 200:
            return query
        return result
    except Exception:
        return query
