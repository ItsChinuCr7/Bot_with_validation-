# ============================================================
# agents/agent_router.py
# ============================================================

ROUTING_RULES = {
    "web":  ["latest news", "current news", "search the web", "search online"],
    "calc": ["calculate", "compute", "solve", "what is the value of"],
    "chat": ["hello", "hi there", "how are you", "who are you", "what can you do"],
}

from prompts.templates import CLASSIFY_PROMPT, GENERAL_PROMPT, CHAT_PROMPT, CALC_PROMPT


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


def route_with_llm(llm, query: str, has_docs: bool, history: list | None = None) -> str:
    if not has_docs:
        return "general"
    try:
        classification_query = query
        if history:
            turns = "\n".join(
                f"{entry['role']}: {entry['message']}"
                for entry in history[-4:]
            )
            classification_query = (
                f"Recent conversation:\n{turns}\n\n"
                f"Current question: {query}"
            )
        result = llm.invoke(
            CLASSIFY_PROMPT.format(query=classification_query)
        ).content.strip().lower()
        return "rag" if "rag" in result else "general"
    except Exception:
        return "rag"


def handle_web_query(query: str) -> str:
    return "Web search not connected. Use 'research <topic>' to search the web via Selenium instead!"


def handle_calc_query(llm, query: str) -> str:
    return llm.invoke(f"Solve step by step:\n\n{query}").content


def handle_chat_query(llm, query: str) -> str:
    prompt = CHAT_PROMPT.format(question=query)
    return llm.invoke(prompt).content


def handle_general_query(llm, query: str) -> str:
    prompt = GENERAL_PROMPT.format(question=query)
    return llm.invoke(prompt).content