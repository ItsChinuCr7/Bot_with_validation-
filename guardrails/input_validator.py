import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, ValidationError

MAX_QUERY_LENGTH = 4000
MAX_UNTRUSTED_TEXT_LENGTH = 100000
SAFE_BLOCKED_RESPONSE = (
    "I can't provide that response because it did not pass safety checks. "
    "Please rephrase your request."
)
INSUFFICIENT_EVIDENCE_RESPONSE = (
    "I could not find enough relevant information in the indexed documents "
    "to answer that question."
)

_INJECTION_PATTERNS = (
    re.compile(
        r"(?:^|[\n.!?]\s*)(?:please\s+)?"
        r"(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?"
        r"(?:the\s+)?(?:previous|prior|above|earlier|system|developer|safety|"
        r"all)\s+(?:instructions?|rules?|guidelines?|messages?)\b"
        r"|"
        r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?"
        r"(?:previous|prior|above|earlier)\s+instructions?\s+"
        r"(?:and|then)\s+(?:reveal|show|print|output|expose)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:reveal|print|show|repeat|quote|output|disclose)\s+(?:the\s+)?"
        r"(?:hidden\s+|initial\s+)?(?:system|developer|secret|internal)\s+"
        r"(?:prompt|instructions?|message|content)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:act|behave|respond|operate)\s+as\s+(?:an?\s+)?"
        r"(?:unrestricted|unfiltered|uncensored|jailbroken)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:developer|system)\s+mode\s+(?:enabled|on)|"
        r"\b(?:DAN|do anything now)\b|"
        r"\b(?:do not|don't|never)\s+follow\s+(?:the\s+)?"
        r"(?:system|developer|previous|safety)\s+(?:prompt|instructions?)\b",
        re.I,
    ),
    re.compile(r"<\s*/?\s*(?:system|developer|assistant)\s*>", re.I),
)
_PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\w)(?:\+?\d[\d().\-\s]{7,}\d)(?!\w)"),
    "person_name": re.compile(
        r"\b(?:my name is|i am|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"
    ),
}
_TOXIC_PATTERNS = (
    re.compile(r"\b(?:kill|hurt)\s+(?:yourself|them|him|her)\b", re.I),
    re.compile(r"\b(?:i will|i'm going to)\s+(?:attack|kill|hurt)\b", re.I),
)
_SEXUAL_REQUEST_PATTERNS = (
    re.compile(
        r"\b(?:i\s+)?(?:want|wanna|would like|plan|intend)\s+"
        r"(?:to\s+)?(?:fuck\s+(?!(?:up|off|around|with|this|that|it|my|the)\b)"
        r"\w+|have sex|sleep with)\b",
        re.I,
    ),
)
_UNSAFE_OUTPUT_PATTERNS = (
    re.compile(r"\b(?:you should|you must|go)\s+(?:kill|hurt)\s+yourself\b", re.I),
    re.compile(r"\b(?:i will|i'm going to|i am going to)\s+(?:attack|kill|hurt)\b", re.I),
    re.compile(
        r"\b(?:step[- ]by[- ]step|instructions?)\s+(?:to|for)\s+"
        r"(?:build|make|construct)\s+(?:a\s+)?(?:bomb|explosive|weapon)\b",
        re.I,
    ),
    re.compile(
        r"\bhow\s+to\s+(?:build|make|construct)\s+(?:a\s+)?"
        r"(?:bomb|explosive|weapon)\b",
        re.I,
    ),
)


class InputValidationError(ValueError):
    def __init__(self, message: str, reason_code: str = "input_rejected"):
        super().__init__(message)
        self.reason_code = reason_code


class GuardrailCheckError(RuntimeError):
    """Raised when a required guardrail could not complete reliably."""


class OutputValidationError(ValueError):
    def __init__(self, reason_code: str):
        super().__init__("Generated output did not pass safety checks.")
        self.reason_code = reason_code


class QueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)


class ValidatedInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    findings: tuple[str, ...]


class InferenceOutput(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    answer: str = Field(min_length=1)
    route: str = Field(min_length=1)
    retrieval_method: str = ""
    chosen_sources: list[str] = Field(default_factory=list)
    source_extracts: list[dict] = Field(default_factory=list)
    context_used: str = ""
    chunks_retrieved: int = 0
    chunks_used: int = 0
    cache_hit: bool = False
    prompt_version: str | None = None
    source_chunks: list[dict] = Field(default_factory=list)
    retrieval_diagnostics: dict = Field(default_factory=dict)
    guardrail_outcomes: list[dict] = Field(default_factory=list)
    safety_outcome: str = "unknown"


def validate_input(query: str) -> ValidatedInput:
    try:
        normalized_query = QueryInput(
            query=unicodedata.normalize("NFKC", query)
        ).query
    except ValidationError as exc:
        too_long = any(error["type"] == "string_too_long" for error in exc.errors())
        if too_long:
            raise InputValidationError(
                f"Query is too long; maximum length is {MAX_QUERY_LENGTH} characters.",
                "query_too_long",
            ) from exc
        raise InputValidationError(
            f"Query must be a non-empty string of at most {MAX_QUERY_LENGTH} characters.",
            "query_invalid",
        ) from exc
    except Exception as exc:
        raise GuardrailCheckError("Input validation could not be completed.") from exc

    try:
        if any(pattern.search(normalized_query) for pattern in _INJECTION_PATTERNS):
            raise InputValidationError(
                "The query was blocked by the prompt-injection guardrail.",
                "prompt_injection",
            )
        if any(pattern.search(normalized_query) for pattern in _TOXIC_PATTERNS):
            raise InputValidationError(
                "The query was blocked by the safety-language guardrail.",
                "unsafe_request",
            )
        if any(
            pattern.search(normalized_query)
            for pattern in _SEXUAL_REQUEST_PATTERNS
        ):
            raise InputValidationError(
                "The query was blocked by the sexual-content guardrail.",
                "sexual_content_request",
            )

        findings = []
        sanitized = normalized_query
        for entity_type, pattern in _PII_PATTERNS.items():
            if pattern.search(sanitized):
                findings.append(f"pii_{entity_type}")
                sanitized = pattern.sub(f"[REDACTED_{entity_type.upper()}]", sanitized)
        return ValidatedInput.model_validate(
            {"text": sanitized, "findings": tuple(findings)}
        )
    except (InputValidationError, GuardrailCheckError):
        raise
    except Exception as exc:
        raise GuardrailCheckError("Input screening could not be completed.") from exc


def sanitize_history(chat_history: list) -> tuple[list, int]:
    """Return prompt-safe history with PII redacted and injection entries removed."""
    try:
        safe_history = []
        removed = 0
        for entry in chat_history or []:
            if not isinstance(entry, dict) or not isinstance(entry.get("message"), str):
                raise GuardrailCheckError("Conversation history has an invalid shape.")
            role = entry.get("role")
            if role not in {"user", "assistant", "bot"}:
                continue
            message = unicodedata.normalize("NFKC", entry["message"])
            if any(pattern.search(message) for pattern in _INJECTION_PATTERNS):
                removed += 1
                continue
            for entity_type, pattern in _PII_PATTERNS.items():
                message = pattern.sub(f"[REDACTED_{entity_type.upper()}]", message)
            message = message.replace("<", "&lt;").replace(">", "&gt;")
            safe_history.append({"role": role, "message": message})
        return safe_history, removed
    except GuardrailCheckError:
        raise
    except Exception as exc:
        raise GuardrailCheckError("Conversation history screening failed.") from exc


def sanitize_stored_history(chat_history: list) -> tuple[list, int]:
    """Sanitize persisted chat messages while preserving safe UI metadata."""
    safe_entries = []
    removed = 0
    try:
        for entry in chat_history or []:
            safe_messages, removed_count = sanitize_history([entry])
            removed += removed_count
            if not safe_messages:
                continue
            safe_entry = dict(entry)
            safe_entry["message"] = safe_messages[0]["message"]
            if isinstance(safe_entry.get("source_extracts"), list):
                safe_extracts = []
                for item in safe_entry["source_extracts"]:
                    if not isinstance(item, dict):
                        continue
                    content, rejected = screen_untrusted_text(
                        str(item.get("content", ""))
                    )
                    if not rejected:
                        safe_item = dict(item)
                        safe_item["content"] = content
                        safe_extracts.append(safe_item)
                safe_entry["source_extracts"] = safe_extracts
            safe_entries.append(safe_entry)
        return safe_entries, removed
    except Exception as exc:
        raise GuardrailCheckError("Stored conversation screening failed.") from exc


def screen_untrusted_text(text: str) -> tuple[str, bool]:
    """Redact PII in untrusted text and reject instruction-like document content."""
    try:
        if not isinstance(text, str):
            raise TypeError("Untrusted text must be a string.")
        normalized = unicodedata.normalize("NFKC", text)
        if len(normalized) > MAX_UNTRUSTED_TEXT_LENGTH:
            raise ValueError("Untrusted text exceeds the screening limit.")
        if any(pattern.search(normalized) for pattern in _INJECTION_PATTERNS):
            return "", True
        sanitized = normalized
        for entity_type, pattern in _PII_PATTERNS.items():
            sanitized = pattern.sub(f"[REDACTED_{entity_type.upper()}]", sanitized)
        sanitized = sanitized.replace("<", "&lt;").replace(">", "&gt;")
        return sanitized, False
    except Exception as exc:
        raise GuardrailCheckError("Untrusted content screening failed.") from exc


def screen_documents(
    docs: list,
    diagnostics: list[dict] | None = None,
) -> tuple[list, int]:
    """Filter injected documents and return safe copies with PII redacted."""
    safe_docs = []
    rejected = 0
    for doc in docs:
        safe_text, rejected_document = screen_untrusted_text(
            getattr(doc, "page_content", None)
        )
        if rejected_document:
            rejected += 1
            if diagnostics is not None:
                diagnostics.append(document_diagnostic(
                    doc, score=None, rejection_reason="document_screening_rejected"
                ))
            continue
        try:
            metadata = dict(getattr(doc, "metadata", {}))
            for key in ("source_file", "section", "path"):
                if isinstance(metadata.get(key), str):
                    safe_value, rejected_metadata = screen_untrusted_text(metadata[key])
                    if rejected_metadata:
                        rejected_document = True
                        break
                    metadata[key] = safe_value
            if rejected_document:
                rejected += 1
                if diagnostics is not None:
                    diagnostics.append(document_diagnostic(
                        doc,
                        score=None,
                        rejection_reason="document_metadata_rejected",
                    ))
                continue
            if hasattr(doc, "model_copy"):
                safe_doc = doc.model_copy(
                    update={"page_content": safe_text, "metadata": metadata}
                )
            else:
                import copy

                safe_doc = copy.copy(doc)
                safe_doc.page_content = safe_text
                safe_doc.metadata = metadata
            safe_docs.append(safe_doc)
        except Exception as exc:
            raise GuardrailCheckError("Retrieved document could not be sanitized.") from exc
    return safe_docs, rejected


def document_diagnostic(
    doc,
    score: float | None,
    rejection_reason: str | None,
) -> dict:
    metadata = getattr(doc, "metadata", {})
    diagnostic = {
        "chunk_id": None,
        "source_file": None,
        "page": None,
        "score": score,
        "rejection_reason": rejection_reason,
    }
    if not isinstance(metadata, dict):
        return diagnostic
    chunk_id = metadata.get("chunk_id")
    if isinstance(chunk_id, str):
        safe_id, rejected = screen_untrusted_text(chunk_id)
        if not rejected:
            diagnostic["chunk_id"] = safe_id[:128]
    source_file = metadata.get("source_file")
    if isinstance(source_file, str):
        safe_source, rejected = screen_untrusted_text(source_file)
        if not rejected:
            diagnostic["source_file"] = safe_source[:256]
    page = metadata.get("page")
    if isinstance(page, int) and not isinstance(page, bool) and page >= 0:
        diagnostic["page"] = page
    return diagnostic


def validate_inference_output(result: dict) -> dict:
    """Validate and normalize the structured result returned by inference."""
    try:
        return InferenceOutput.model_validate(result).model_dump()
    except ValidationError as exc:
        raise GuardrailCheckError("Inference output validation failed.") from exc


def enforce_output(answer: str) -> None:
    """Reject unsafe, overlong, or PII-bearing generated content."""
    if not isinstance(answer, str):
        raise GuardrailCheckError("Output validation received an invalid response.")
    try:
        if len(answer) > 20000:
            raise OutputValidationError("output_length_exceeded")
        if any(pattern.search(answer) for pattern in _UNSAFE_OUTPUT_PATTERNS):
            raise OutputValidationError("unsafe_output")
        if any(pattern.search(answer) for pattern in _INJECTION_PATTERNS):
            raise OutputValidationError("prompt_injection_in_output")
        if any(pattern.search(answer) for pattern in _PII_PATTERNS.values()):
            raise OutputValidationError("pii_in_output")
    except OutputValidationError:
        raise
    except Exception as exc:
        raise GuardrailCheckError("Output validation could not be completed.") from exc


def validate_output(answer: str, route: str, context: str = "") -> list[str]:
    """Return grounding findings for the inference layer to enforce."""
    flags = []
    if route == "rag" and not context.strip():
        return ["insufficient_evidence"]
    if route == "rag" and answer.strip():
        stopwords = {
            "about", "after", "also", "and", "are", "based", "been", "being",
            "could", "describes", "does", "from", "have", "information", "into",
            "more", "most", "other", "should", "some", "such", "than", "that",
            "their", "them", "then", "there", "these", "they", "this", "those",
            "through", "under", "very", "what", "when", "where", "which", "while",
            "with", "would",
        }
        context_terms = {
            term for term in re.findall(r"\b[a-zA-Z0-9]{4,}\b", context.lower())
            if term not in stopwords
        }
        answer_terms = {
            term for term in re.findall(r"\b[a-zA-Z0-9]{4,}\b", answer.lower())
            if term not in stopwords
        }
        shared_terms = context_terms & answer_terms
        overlap = len(shared_terms) / max(1, len(answer_terms))
        if (
            (overlap < 0.20 or (len(answer_terms) >= 3 and len(shared_terms) < 2))
            and "could not find" not in answer.lower()
        ):
            flags.append("grounding_review_required")
    return flags
