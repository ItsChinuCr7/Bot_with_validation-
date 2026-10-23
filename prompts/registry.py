from prompts.templates import RAG_PROMPT, RAG_PROMPT_NO_HISTORY

PROMPT_REGISTRY = {
    "rag_v1": {
        "version": "1.0",
        "created": "2026-10-21",
        "template": RAG_PROMPT,
        "template_no_history": RAG_PROMPT_NO_HISTORY,
    },
}

ACTIVE_RAG_PROMPT = "rag_v1"


def get_prompt_version(name: str = ACTIVE_RAG_PROMPT) -> dict:
    try:
        return PROMPT_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown RAG prompt version: {name}") from exc
