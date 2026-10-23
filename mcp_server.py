import logging
import os
from functools import lru_cache
from pathlib import Path
import sys

from mcp.server.fastmcp import FastMCP

from config.settings import EMBEDDING_MODEL, LLM_MODEL, LLM_MODEL_HEAVY, VECTOR_INDEX_PATH
from guardrails.input_validator import (
    SAFE_BLOCKED_RESPONSE,
    GuardrailCheckError,
    InputValidationError,
    OutputValidationError,
    validate_input,
)
from inference.inference_engine import run_inference
from middleware.rate_limiter import RateLimitExceeded
from observability.logfire_tracing import record_guardrail_decision
from langchain_huggingface import HuggingFaceEmbeddings
from models.model_factory import create_llm
from retrieval.vector_store import load_vector_store

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)
mcp = FastMCP("AI Document Intelligence Assistant")


@lru_cache(maxsize=1)
def _models():
    return (
        create_llm(model_name=LLM_MODEL, mode="light"),
        create_llm(model_name=LLM_MODEL_HEAVY, mode="heavy"),
    )


@lru_cache(maxsize=1)
def _vector_store():
    if not Path(VECTOR_INDEX_PATH).exists():
        raise RuntimeError(
            f"No FAISS index at {VECTOR_INDEX_PATH}; process documents in the app first."
        )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return load_vector_store(embeddings)


def _session_state(include_documents: bool = False) -> dict:
    state = {"_session_id": f"mcp-{os.getpid()}"}
    if include_documents:
        state["vector_db"] = _vector_store()
    return state


def _run_query(query: str, mode: str) -> str:
    try:
        validated = validate_input(query)
    except InputValidationError as exc:
        record_guardrail_decision(
            mode, "input_validation", "blocked", exc.reason_code
        )
        return SAFE_BLOCKED_RESPONSE
    except GuardrailCheckError:
        record_guardrail_decision(
            mode, "input_validation", "blocked", "check_failed"
        )
        return SAFE_BLOCKED_RESPONSE
    if validated.findings:
        record_guardrail_decision(
            mode, "input_validation", "redacted", "pii_redacted",
            len(validated.findings),
        )

    try:
        if mode == "rag":
            state = _session_state(include_documents=True)
        else:
            state = _session_state()
        llm, llm_heavy = _models()
        result = run_inference(
            query=validated.text,
            llm=llm,
            llm_heavy=llm_heavy,
            session_state=state,
            memory_on=False,
            forced_mode=mode,
            use_hybrid=False,
        )
    except (InputValidationError, GuardrailCheckError, OutputValidationError, RateLimitExceeded):
        return SAFE_BLOCKED_RESPONSE
    except Exception as exc:
        logger.error("MCP request failed (%s)", type(exc).__name__)
        return "The request could not be completed safely. Please try again."
    answer = result["answer"]
    if mode == "rag" and result.get("source_chunks"):
        answer += "\n\nSource chunks:\n" + "\n".join(
            f"- {source['source_file']} page {source['page']} "
            f"(context SHA-256 {source['context_sha256']}"
            f"{', chunk ' + source['chunk_id'] if source['chunk_id'] else ''})"
            for source in result["source_chunks"]
        )
    return answer


@mcp.tool()
def ask_documents(query: str) -> str:
    """Answer a question using the locally indexed PDF documents."""
    return _run_query(query, "rag")


@mcp.tool()
def ask_general(query: str) -> str:
    """Answer a general-knowledge question using the configured local model."""
    return _run_query(query, "general")


@mcp.tool()
def research_web(query: str) -> str:
    """Research a topic using the app's configured web research sources."""
    return _run_query(query, "research")


@mcp.tool()
def list_documents() -> list[dict]:
    """List unique source files and page counts present in the FAISS index."""
    documents = _vector_store().docstore._dict.values()
    sources: dict[str, set[int]] = {}
    for document in documents:
        source = document.metadata.get("source_file", "unknown")
        page = document.metadata.get("page")
        sources.setdefault(source, set())
        if isinstance(page, int):
            sources[source].add(page)
    return [
        {"source_file": source, "pages_indexed": len(pages)}
        for source, pages in sorted(sources.items())
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
