import re

from config.settings import HIGH_RELEVANCE_SCORE, MIN_RELEVANCE_SCORE

UNCERTAINTY_PHRASES = [
    "i don't know", "i do not know", "i'm not sure",
    "i cannot find", "i could not find", "not in the context",
]


_STOPWORDS = {
    "about", "after", "also", "and", "are", "because", "been", "being",
    "describes",
    "could", "does", "from", "have", "into", "more", "most", "other",
    "should", "some", "such", "than", "that", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "under", "very",
    "what", "when", "where", "which", "while", "with", "would",
}


def _grounding_coverage(answer: str, context: str) -> float:
    answer_terms = {
        term for term in re.findall(r"\b[a-zA-Z]{4,}\b", answer.lower())
        if term not in _STOPWORDS
    }
    if not answer_terms:
        return 0.0
    context_terms = set(re.findall(r"\b[a-zA-Z]{4,}\b", context.lower()))
    return len(answer_terms & context_terms) / len(answer_terms)


def validate_answer(
    answer: str,
    route: str = "general",
    context: str = "",
    source_chunks: list[dict] | None = None,
    guardrail_flags: list[str] | None = None,
):
    if not answer or len(answer.strip()) < 15:
        return {
            "valid": False,
            "confidence": "low",
            "flag": "Answer too short.",
            "grounding_score": 0.0,
            "retrieval_score": None,
        }
    for phrase in UNCERTAINTY_PHRASES:
        if phrase in answer.lower():
            return {
                "valid": True,
                "confidence": "low",
                "flag": f"LLM uncertain: '{phrase}'",
                "grounding_score": _grounding_coverage(answer, context),
                "retrieval_score": None,
            }

    chunks = source_chunks or []
    scores = [
        float(chunk["relevance_score"])
        for chunk in chunks
        if isinstance(chunk, dict)
        and isinstance(chunk.get("relevance_score"), (int, float))
    ]
    retrieval_score = max(scores) if scores else None
    grounding_score = _grounding_coverage(answer, context)
    flags = set(guardrail_flags or [])
    valid = bool(
        route == "rag"
        and context.strip()
        and chunks
        and not flags
        and retrieval_score is not None
        and retrieval_score >= MIN_RELEVANCE_SCORE
        and grounding_score >= 0.20
    )

    confidence = "low"
    if (
        valid
        and route == "rag"
        and retrieval_score is not None
        and retrieval_score >= HIGH_RELEVANCE_SCORE
        and grounding_score >= 0.35
    ):
        confidence = "high"
    elif (
        valid
        and route == "rag"
        and retrieval_score is not None
        and retrieval_score >= MIN_RELEVANCE_SCORE
        and grounding_score >= 0.15
    ):
        confidence = "medium"

    flag = None if confidence != "low" else (
        "Answer confidence is low: evidence or validation was insufficient."
    )
    return {
        "valid": valid,
        "confidence": confidence,
        "flag": flag,
        "grounding_score": round(grounding_score, 3),
        "retrieval_score": (
            round(retrieval_score, 3) if retrieval_score is not None else None
        ),
    }

def format_confidence_badge(confidence):
    return {
        "high": "🟢 High confidence",
        "medium": "🟡 Medium confidence",
        "low": "🔴 Low confidence",
    }.get(confidence, "🔴 Low confidence")
