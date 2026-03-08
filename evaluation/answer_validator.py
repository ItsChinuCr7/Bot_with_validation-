UNCERTAINTY_PHRASES = [
    "i don't know", "i do not know", "i'm not sure",
    "i cannot find", "i could not find", "not in the context",
]

def validate_answer(answer):
    if not answer or len(answer.strip()) < 15:
        return {"valid": False, "confidence": "low", "flag": "Answer too short."}
    for phrase in UNCERTAINTY_PHRASES:
        if phrase in answer.lower():
            return {"valid": True, "confidence": "low", "flag": f"LLM uncertain: '{phrase}'"}
    confidence = "high" if len(answer) > 200 else "medium" if len(answer) > 80 else "low"
    return {"valid": True, "confidence": confidence, "flag": None}

def format_confidence_badge(confidence):
    return {"high": "🟢 High confidence", "medium": "🟡 Medium confidence", "low": "🔴 Low confidence"}.get(confidence, "")
