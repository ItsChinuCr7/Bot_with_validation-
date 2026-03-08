# ============================================================
# agents/agent_router.py
# ============================================================

ROUTING_RULES = {
    "web":  ["latest news", "current news", "search the web", "search online"],
    "calc": ["calculate", "compute", "solve", "what is the value of"],
    "chat": ["hello", "hi there", "how are you", "who are you", "what can you do"],
}

CLASSIFY_PROMPT = """You are a query classifier.

Decide if this question needs uploaded documents or general knowledge.

Rules:
- If it asks about a specific uploaded file, report, contract, paper → answer "rag"
- If it is a general knowledge question (science, history, coding, facts) → answer "general"
- If unsure → answer "general"

Question: {query}

Answer with ONLY one word, either "rag" or "general":"""


def extract_research_query(query: str) -> str:
    """Strip the word 'research' from the query if present."""
    import re
    clean = re.sub(r"\bresearch\b", "", query, flags=re.IGNORECASE).strip()
    clean = re.sub(r"\s+", " ", clean).strip(" ,:-")
    return clean if clean else query


def route_query(query: str, has_docs: bool = False, forced_mode: str = None) -> str:
    """
    Route query to correct handler.

    forced_mode: if set from sidebar selection, use that mode directly.
                 Options: "research", "rag", "general", None (auto-detect)
    """
    q = query.lower()

    # If user selected a mode in sidebar — respect it (skip keyword detection)
    if forced_mode and forced_mode in ["research", "rag", "general"]:
        return forced_mode

    # Always detect calc regardless of mode
    for kw in ROUTING_RULES["calc"]:
        if kw in q:
            return "calc"

    # Always detect pure chat/greetings
    for kw in ROUTING_RULES["chat"]:
        if kw in q:
            return "chat"

    # Research keyword still works as shortcut even without forced_mode
    if "research" in q:
        return "research"

    for tool, keywords in ROUTING_RULES.items():
        for kw in keywords:
            if kw in q:
                return tool

    if not has_docs:
        return "general"

    return "needs_classification"


def route_with_llm(llm, query: str, has_docs: bool) -> str:
    if not has_docs:
        return "general"
    try:
        result = llm.invoke(CLASSIFY_PROMPT.format(query=query)).content.strip().lower()
        return "rag" if "rag" in result else "general"
    except Exception:
        return "rag"


def handle_web_query(query: str) -> str:
    return "Web search not connected. Use 'research <topic>' to search the web via Selenium instead!"


def handle_calc_query(llm, query: str) -> str:
    return llm.invoke(f"Solve step by step:\n\n{query}").content


def handle_chat_query(llm, query: str) -> str:
    return llm.invoke(f"You are a friendly AI assistant. Answer conversationally:\n\n{query}").content


def handle_general_query(llm, query: str) -> str:
    prompt = f"""You are a knowledgeable AI assistant.
Answer this question clearly and accurately from your own knowledge.
If you are not sure, say so honestly.

Question: {query}

Answer:"""
    return llm.invoke(prompt).content