# ============================================================
# inference/inference_engine.py
# Central inference orchestrator.
# All query handling logic lives here — app.py just calls
# run_inference() and gets back a clean result dict.
# ============================================================

import re
import hashlib
import json
import logging
import time
import uuid
from data.cache.cache_manager import get_cached, set_cache
from guardrails.input_validator import (
    INSUFFICIENT_EVIDENCE_RESPONSE,
    SAFE_BLOCKED_RESPONSE,
    GuardrailCheckError,
    InputValidationError,
    OutputValidationError,
    enforce_output,
    sanitize_history,
    validate_input,
    validate_inference_output,
    validate_output,
)
from middleware.rate_limiter import (
    RateLimitExceeded,
    audit_query,
    rate_limiter,
)
from observability.metrics import TrackedLLM, record_inference
from observability.experiment_tracking import log_experiment
from observability.logfire_tracing import inference_trace
from prompts.registry import ACTIVE_RAG_PROMPT, PROMPT_REGISTRY
from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL

logger = logging.getLogger(__name__)
from prompts.templates import (
    RESEARCH_SYNTHESIS_PROMPT,
    CALC_PROMPT,
)
from agents.agent_router import (
    route_query,
    route_with_llm,
    handle_web_query,
    handle_chat_query,
)
from reasoning.rag_pipeline import (
    run_rag_pipeline,
    run_general_with_memory,
    format_history,
)
from research.web_researcher import (
    decide_sources,
    parse_user_source_override,
    run_research,
    extract_research_query,
)
from retrieval.retriever import get_mmr_retriever
from config.settings import TOP_K_RETRIEVAL


# ── Hybrid Retriever wrapper ───────────────────────────────
class HybridRetriever:
    def __init__(self, searcher):
        self.searcher = searcher

    def invoke(self, q):
        return self.searcher.search(q, top_k=TOP_K_RETRIEVAL)


# ── Main entry point ───────────────────────────────────────
def _run_inference(
    query: str,
    llm,
    llm_heavy,
    session_state: dict,
    memory_on: bool = True,
    forced_mode: str = None,
    use_hybrid: bool = True,
    progress_callback=None,   # optional fn(msg: str) for UI updates
    guardrail_event=None,
) -> dict:
    """
    Central inference function. Handles all routes and returns a unified result dict.

    Returns:
        {
            "answer":           str,
            "route":            str,
            "retrieval_method": str,
            "chosen_sources":   list,
            "source_extracts":  list,
            "context_used":     str,
            "was_corrected":    bool,
            "gemini_feedback":  str,
        }
    """

    def log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    has_docs = "vector_db" in session_state
    history_for_llm, removed_history = sanitize_history(
        session_state.get("chat_history", []) if memory_on else []
    )
    if removed_history and guardrail_event:
        guardrail_event(
            "history_screening", "filtered", "prompt_injection",
            count=removed_history,
        )

    # Init result
    result = {
        "answer":           "",
        "route":            "",
        "retrieval_method": "",
        "chosen_sources":   [],
        "source_extracts":  [],
        "context_used":     "",
        "chunks_retrieved": 0,
        "chunks_used":      0,
        "cache_hit":        False,
        "prompt_version":   None,
        "source_chunks":    [],
        "retrieval_diagnostics": {},
        "guardrail_outcomes": [],
        "safety_outcome": "unknown",
    }

    # ── Route ─────────────────────────────────────────────
    log("🔀 Detecting query type...")
    route = route_query(query, has_docs=has_docs, forced_mode=forced_mode)
    if route == "needs_classification":
        previous_route = _last_assistant_route(
            session_state.get("chat_history", [])
        )
        if (
            memory_on
            and has_docs
            and previous_route == "rag"
            and _is_follow_up_query(query)
        ):
            route = "rag"
        else:
            log("🧠 Classifying with LLM...")
            route = route_with_llm(
                llm,
                query,
                has_docs=has_docs,
                history=(
                    history_for_llm
                    if memory_on and previous_route == "rag" else None
                ),
            )
    log(f"✅ Route → **{route.upper()}**")
    result["route"] = route
    if route == "rag":
        result["prompt_version"] = PROMPT_REGISTRY[ACTIVE_RAG_PROMPT]["version"]
        if not has_docs:
            raise ValueError("RAG mode requires processed documents.")

    # ── Cache check ────────────────────────────────────────
    cache_query, cache_route = _cache_identity(
        query, route, session_state, history_for_llm
    )
    cached = (
        get_cached(cache_query, cache_route)
        if cache_query is not None and cache_route is not None else None
    )
    if cached:
        log("⚡ Cache hit — returning instant answer")
        cached["route"] = route
        cached["cache_hit"] = True
        return _enforce_result_output(cached, guardrail_event)

    # ── Research ───────────────────────────────────────────
    if route == "research":
        if re.search(r"\bresearch\b", query, re.IGNORECASE):
            clean_query, source_override = parse_user_source_override(query)
            if not source_override:
                clean_query = extract_research_query(query)
        else:
            clean_query     = query
            source_override = None

        log("🧠 Picking research sources...")
        chosen_sources = decide_sources(llm, clean_query, user_override=source_override)
        log(f"📡 Sources: **{', '.join(s.upper() for s in chosen_sources)}**")
        result["chosen_sources"] = chosen_sources

        for src in chosen_sources:
            log(f"🌐 Scraping **{src}**...")

        research_result = run_research(clean_query, chosen_sources)

        if research_result["error"]:
            result["answer"] = (
                "Research could not be completed safely. Please try again later."
            )
        else:
            source_extracts = research_result.get("results", [])
            context_used    = research_result.get("context", "")
            safe_extracts, rejected_extracts = sanitize_research_extracts(
                source_extracts
            )
            result["source_extracts"] = safe_extracts
            result["context_used"]    = context_used

            log(f"✅ Scraped **{len(source_extracts)}** source(s)")
            if guardrail_event:
                guardrail_event(
                    "research_source_screening",
                    "filtered" if rejected_extracts else "passed",
                    "prompt_injection" if rejected_extracts else "none",
                    count=rejected_extracts,
                )
            for r in source_extracts:
                log(f"  → {r['source'].upper()}: {len(r['content'])} chars")

            if not context_used.strip():
                result["answer"] = "No content could be extracted from the sources."
            else:
                safe_context, rejected_context = screen_research_context(context_used)
                if guardrail_event:
                    guardrail_event(
                        "research_context_screening",
                        "blocked" if rejected_context else "passed",
                        "prompt_injection" if rejected_context else "none",
                        count=1 if rejected_context else 0,
                    )
                if rejected_context:
                    result["answer"] = SAFE_BLOCKED_RESPONSE
                    result["context_used"] = ""
                    result["source_extracts"] = []
                else:
                    context_used = safe_context
                    result["context_used"] = safe_context
                log("✍️ Synthesizing with LLM...")
                if not rejected_context:
                    synthesis_prompt = RESEARCH_SYNTHESIS_PROMPT.format(
                        question=clean_query,
                        context=context_used[:6000],
                    )
                    try:
                        result["answer"] = llm.invoke(synthesis_prompt).content
                    except Exception:
                        try:
                            result["answer"] = llm_heavy.invoke(synthesis_prompt).content
                        except Exception:
                            result["answer"] = (
                                "Research could not be completed safely. "
                                "Please try again later."
                            )

    # ── Web (legacy) ───────────────────────────────────────
    elif route == "web":
        result["answer"] = handle_web_query(query)

    # ── Calc ───────────────────────────────────────────────
    elif route == "calc":
        log("🔢 Solving...")
        history_str = format_history(history_for_llm, 2) if memory_on and history_for_llm else ""
        hist_block  = f"History:\n{history_str}\n\n" if history_str else ""
        prompt      = CALC_PROMPT.format(history=hist_block, question=query)
        result["answer"] = llm.invoke(prompt).content

    # ── Chat ───────────────────────────────────────────────
    elif route == "chat":
        result["answer"] = handle_chat_query(llm, query)

    # ── General ────────────────────────────────────────────
    elif route == "general":
        log("💡 Using LLM knowledge...")
        q = query
        result["answer"] = run_general_with_memory(llm, q, history_for_llm)

    # ── RAG ────────────────────────────────────────────────
    else:
        if use_hybrid and "hybrid_searcher" in session_state:
            retriever = HybridRetriever(session_state["hybrid_searcher"])
            log("🔀 Using Hybrid Search (BM25 + Vector)")
        else:
            retriever = get_mmr_retriever(session_state["vector_db"])
            log("🔢 Using MMR Vector Search")

        doc_trees = session_state.get("doc_trees", None)
        log("🔍 Retrieving context...")
        log("✍️ Generating answer...")

        rag_result = run_rag_pipeline(
            llm, retriever, query,
            chat_history=history_for_llm,
            document_trees=doc_trees,
            return_metadata=True,
            guardrail_event=guardrail_event,
            diagnostic_callback=lambda diagnostic: result.update(
                retrieval_diagnostics=diagnostic
            ),
        )
        (
            answer,
            retrieval_method,
            chunks_retrieved,
            chunks_used,
            context_used,
            source_chunks,
        ) = rag_result
        result["answer"]           = answer
        result["retrieval_method"] = retrieval_method
        result["chunks_retrieved"] = chunks_retrieved
        result["chunks_used"]      = chunks_used
        result["context_used"]     = context_used
        result["source_chunks"]   = source_chunks
        log(f"✅ Retrieval: {retrieval_method}")

    result = _enforce_result_output(result, guardrail_event)

    # ── Save only safety-checked responses to cache ────────
    if cache_query is not None and cache_route is not None:
        set_cache(cache_query, cache_route, result)

    return result


def _is_follow_up_query(query: str) -> bool:
    return bool(re.match(
        r"^\s*(?:and\b|also\b|what about\b|how about\b|why\b|how\b|"
        r"what else\b|tell me more\b|explain (?:that|this|it)\b|"
        r"can you elaborate\b|does that\b|what do you mean\b|which one\b)",
        query,
        re.IGNORECASE,
    ))


def _last_assistant_route(chat_history: list) -> str | None:
    for entry in reversed(chat_history or []):
        if not isinstance(entry, dict) or entry.get("role") not in {"bot", "assistant"}:
            continue
        route = entry.get("route")
        if isinstance(route, str):
            return route.lower()
        meta = entry.get("meta")
        if isinstance(meta, str):
            match = re.search(r"\bRoute:\s*(RAG|GENERAL|RESEARCH|CHAT|CALC)\b", meta, re.I)
            if match:
                return match.group(1).lower()
        return None
    return None


def _cache_identity(query: str, route: str, session_state: dict, history: list):
    if route != "rag":
        return query, route
    vector_db = session_state.get("vector_db")
    if vector_db is None:
        return None, None
    documents = session_state.get("docs")
    if not isinstance(documents, list):
        docstore = getattr(vector_db, "docstore", None)
        documents = list(getattr(docstore, "_dict", {}).values())
    fingerprints = []
    for doc in documents:
        metadata = getattr(doc, "metadata", {}) or {}
        fingerprint = metadata.get("chunk_hash")
        if not isinstance(fingerprint, str):
            fingerprint = hashlib.sha256(
                str(getattr(doc, "page_content", "")).encode("utf-8")
            ).hexdigest()
        fingerprints.append(fingerprint)
    if not fingerprints:
        return None, None
    corpus_id = hashlib.sha256(
        "\n".join(sorted(fingerprints)).encode("utf-8")
    ).hexdigest()
    cache_route = (
        f"rag-v2:{PROMPT_REGISTRY[ACTIVE_RAG_PROMPT]['version']}:{corpus_id}"
    )
    cache_query = query
    if history:
        history_hash = hashlib.sha256(
            json.dumps(history, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        cache_query = f"{query}\n[history:{history_hash}]"
    return cache_query, cache_route


def _enforce_result_output(result: dict, guardrail_event=None) -> dict:
    try:
        enforce_output(result["answer"])
    except OutputValidationError as exc:
        if guardrail_event:
            guardrail_event("output_validation", "blocked", exc.reason_code)
        result.update({
            "answer": SAFE_BLOCKED_RESPONSE,
            "context_used": "",
            "source_extracts": [],
            "source_chunks": [],
            "guardrail_flags": [exc.reason_code],
        })
    else:
        if guardrail_event:
            guardrail_event("output_validation", "passed", "none")
    return result


def screen_research_context(context: str) -> tuple[str, bool]:
    from guardrails.input_validator import screen_untrusted_text

    return screen_untrusted_text(context)


def sanitize_research_extracts(extracts: list) -> tuple[list, int]:
    from guardrails.input_validator import screen_untrusted_text

    sanitized = []
    rejected_count = 0
    for item in extracts:
        if not isinstance(item, dict):
            raise GuardrailCheckError("Research source screening failed.")
        content, rejected = screen_untrusted_text(str(item.get("content", "")))
        if rejected:
            rejected_count += 1
            continue
        safe_item = dict(item)
        safe_item["content"] = content
        for key in ("source", "url", "title"):
            if isinstance(safe_item.get(key), str):
                safe_value, rejected_metadata = screen_untrusted_text(
                    safe_item[key]
                )
                if rejected_metadata:
                    rejected_count += 1
                    break
                safe_item[key] = safe_value
        else:
            sanitized.append(safe_item)
    return sanitized, rejected_count


def run_inference(
    query: str,
    llm,
    llm_heavy,
    session_state: dict,
    memory_on: bool = True,
    forced_mode: str = None,
    use_hybrid: bool = True,
    progress_callback=None,
) -> dict:
    requested_route = forced_mode or "auto"
    with inference_trace(requested_route) as trace:
        def guardrail_event(stage, decision, reason_code, count=0):
            if trace is not None:
                trace.record_guardrail(stage, decision, reason_code, count)

        return _run_inference_traced(
            query=query,
            llm=llm,
            llm_heavy=llm_heavy,
            session_state=session_state,
            memory_on=memory_on,
            forced_mode=forced_mode,
            use_hybrid=use_hybrid,
            progress_callback=progress_callback,
            trace=trace,
            guardrail_event=guardrail_event,
        )


def _run_inference_traced(
    query: str,
    llm,
    llm_heavy,
    session_state: dict,
    memory_on: bool = True,
    forced_mode: str = None,
    use_hybrid: bool = True,
    progress_callback=None,
    trace=None,
    guardrail_event=None,
) -> dict:
    """Validate, rate-limit, trace, and audit one inference request."""
    started = time.perf_counter()
    session_id = session_state.setdefault("_session_id", str(uuid.uuid4()))
    traced_llm = TrackedLLM(llm)
    traced_heavy = TrackedLLM(llm_heavy)
    metrics = {
        "route": "blocked",
        "model": None,
        "models_used": [],
        "latency_ms": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "token_count_source": "provider",
        "cache_hit": False,
        "retrieval_method": None,
        "chunks_retrieved": 0,
        "chunks_used": 0,
        "prompt_version": None,
        "error": None,
    }
    outcome = "blocked"
    validated = None
    guardrail_failure_reason = None
    record_guardrail_event = guardrail_event
    guardrail_outcomes = []

    def guardrail_event(stage, decision, reason_code, count=0):
        nonlocal guardrail_failure_reason
        if decision in {"blocked", "error"} and guardrail_failure_reason is None:
            guardrail_failure_reason = reason_code
        guardrail_outcomes.append({
            "stage": stage,
            "decision": decision,
            "reason_code": reason_code,
            "count": count,
        })
        if record_guardrail_event:
            record_guardrail_event(stage, decision, reason_code, count)

    try:
        validated = validate_input(query)
        if guardrail_event:
            guardrail_event(
                "input_validation",
                "redacted" if validated.findings else "passed",
                "pii_redacted" if validated.findings else "none",
                count=len(validated.findings),
            )
        rate_limiter.check(session_id)
        result = _run_inference(
            query=validated.text,
            llm=traced_llm,
            llm_heavy=traced_heavy,
            session_state=session_state,
            memory_on=memory_on,
            forced_mode=forced_mode,
            use_hybrid=use_hybrid,
            progress_callback=progress_callback,
            guardrail_event=guardrail_event,
        )
        try:
            result = validate_inference_output(result)
        except GuardrailCheckError:
            if guardrail_event:
                guardrail_event(
                    "output_validation", "blocked", "invalid_output_shape"
                )
            result = {
                "answer": SAFE_BLOCKED_RESPONSE,
                "route": "blocked",
                "retrieval_method": "",
                "chosen_sources": [],
                "source_extracts": [],
                "context_used": "",
                "chunks_retrieved": 0,
                "chunks_used": 0,
                "cache_hit": False,
                "prompt_version": None,
                "source_chunks": [],
                "guardrail_flags": ["invalid_output_shape"],
            }
        try:
            enforce_output(result["answer"])
        except OutputValidationError as exc:
            if guardrail_event:
                guardrail_event("output_validation", "blocked", exc.reason_code)
            result.update({
                "answer": SAFE_BLOCKED_RESPONSE,
                "context_used": "",
                "source_extracts": [],
                "source_chunks": [],
                "guardrail_flags": [exc.reason_code],
            })
        result["guardrail_flags"] = list(result.get("guardrail_flags", []))
        result["guardrail_flags"].extend(validated.findings)
        grounding_flags = validate_output(
            result["answer"], result["route"], result.get("context_used", "")
        )
        result["guardrail_flags"].extend(grounding_flags)
        blocked_output_flags = {
            "unsafe_output",
            "abusive_language",
            "prompt_injection_in_output",
            "pii_in_output",
            "output_length_exceeded",
        }
        if result["route"] == "rag" and not (
            blocked_output_flags & set(result["guardrail_flags"])
        ) and any(
            flag in grounding_flags
            for flag in ("insufficient_evidence", "grounding_review_required")
        ):
            if guardrail_event:
                guardrail_event(
                    "answer_validation", "abstained", "insufficient_evidence"
                )
            result.update({
                "answer": INSUFFICIENT_EVIDENCE_RESPONSE,
                "context_used": "",
                "source_chunks": [],
                "chunks_used": 0,
                "retrieval_method": "none",
            })
            enforce_output(result["answer"])
        metrics.update({
            "route": result["route"],
            "cache_hit": result.get("cache_hit", False),
            "retrieval_method": result.get("retrieval_method") or None,
            "chunks_retrieved": result.get("chunks_retrieved", 0),
            "chunks_used": result.get("chunks_used", 0),
            "prompt_version": result.get("prompt_version"),
        })
        result["safe_query"] = validated.text
        outcome = (
            "blocked"
            if result["route"] == "blocked"
            or result.get("retrieval_method") == "blocked"
            or guardrail_failure_reason is not None
            or any(
                flag in result["guardrail_flags"]
                for flag in (
                    "unsafe_output",
                    "prompt_injection_in_output",
                    "output_length_exceeded",
                    "pii_in_output",
                    "abusive_language",
                )
            )
            else "success"
        )
        result["guardrail_outcomes"] = guardrail_outcomes
        result["safety_outcome"] = "blocked" if outcome == "blocked" else "passed"
        return result
    except (InputValidationError, RateLimitExceeded) as exc:
        metrics["error"] = type(exc).__name__
        if isinstance(exc, InputValidationError) and guardrail_event:
            guardrail_event("input_validation", "blocked", exc.reason_code)
        raise
    except GuardrailCheckError:
        outcome = "blocked"
        metrics["error"] = "GuardrailCheckError"
        if guardrail_event:
            guardrail_event("guardrail_check", "blocked", "check_failed")
        raise
    except OutputValidationError as exc:
        metrics["error"] = "OutputValidationError"
        if guardrail_event:
            guardrail_event("output_validation", "blocked", exc.reason_code)
        raise
    except Exception as exc:
        outcome = "error"
        metrics["error"] = type(exc).__name__
        logger.error("Inference failed (%s)", type(exc).__name__)
        raise
    finally:
        metrics["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        metrics["outcome"] = outcome
        metrics["tokens_in"] = traced_llm.input_tokens + traced_heavy.input_tokens
        metrics["tokens_out"] = traced_llm.output_tokens + traced_heavy.output_tokens
        metrics["models_used"] = [
            model.get_model_name()
            for model in (traced_llm, traced_heavy)
            if model.calls
        ]
        metrics["model"] = metrics["models_used"][0] if metrics["models_used"] else None
        if traced_llm.estimated_tokens or traced_heavy.estimated_tokens:
            metrics["token_count_source"] = "provider_or_character_estimate"
        if trace is not None:
            safety_decision = (
                ("error", "inference_error")
                if outcome == "error"
                else (
                    ("blocked", guardrail_failure_reason)
                    if guardrail_failure_reason or outcome == "blocked"
                    else ("passed", "none")
                )
            )
            trace.record_safety_outcome(*safety_decision)
            trace.record(metrics)
        audit_query(
            session_id,
            validated.text if validated is not None else None,
            metrics["route"],
            outcome,
        )
        record_inference(metrics)
        log_experiment(
            run_name=f"{metrics['route']}-{session_id[:8]}",
            params={
                "model": metrics["model"],
                "models_used": metrics["models_used"],
                "route": metrics["route"],
                "retrieval_method": metrics["retrieval_method"],
                "prompt_version": metrics["prompt_version"],
                "embedding_model": EMBEDDING_MODEL,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "cache_hit": metrics["cache_hit"],
            },
            metrics={
                "latency_ms": metrics["latency_ms"],
                "tokens_in": metrics["tokens_in"],
                "tokens_out": metrics["tokens_out"],
                "chunks_retrieved": metrics["chunks_retrieved"],
                "chunks_used": metrics["chunks_used"],
            },
        )