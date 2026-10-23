# Task Change Report

This report covers the AI/ML engineering feature implementation in the current worktree. Pre-existing edits to caches, FAISS binaries, logs, and tracked bytecode are excluded.

## 1. Files Added

| File | Purpose |
|---|---|
| [.dockerignore](./.dockerignore) | Excludes local indexes, PDFs, logs, credentials, and runtime state from Docker build context. |
| [.github/workflows/ml-quality.yml](./.github/workflows/ml-quality.yml) | Adds compile/unit checks and a configurable RAGAS faithfulness gate. |
| [.gitignore](./.gitignore) | Adds the correctly named Git ignore file, retaining the former `gitignore` rules and excluding evaluation/MLflow output. |
| [CHANGELOG_FOR_CLAUDE.md](./CHANGELOG_FOR_CLAUDE.md) | Earlier detailed handoff for this implementation. |
| [Dockerfile](./Dockerfile) | Defines a non-root Python 3.12 Streamlit container. |
| [evaluation/ab_tester.py](./evaluation/ab_tester.py) | Runs paired queries across two models using caller-provided answer and scoring functions. |
| [evaluation/datasets/sample_rag_eval.json](./evaluation/datasets/sample_rag_eval.json) | Provides 20 synthetic examples for evaluator smoke testing, not a domain benchmark. |
| [evaluation/eval_suite.py](./evaluation/eval_suite.py) | Adds RAGAS scoring for faithfulness, context precision, answer relevance, and correctness, with optional live FAISS retrieval. |
| [guardrails/__init__.py](./guardrails/__init__.py) | Package marker. |
| [guardrails/input_validator.py](./guardrails/input_validator.py) | Validates queries, blocks selected injection/safety patterns, redacts matching PII, and flags output length/grounding concerns. |
| [ingestion/lineage.py](./ingestion/lineage.py) | Adds chunk IDs, SHA-256 hashes, ingestion time, embedding model, and modification metadata. |
| [mcp_server.py](./mcp_server.py) | Provides four local stdio MCP tools: document Q&A, general Q&A, web research, and document listing. |
| [middleware/__init__.py](./middleware/__init__.py) | Package marker. |
| [middleware/rate_limiter.py](./middleware/rate_limiter.py) | Implements an in-memory per-session sliding-window limit and privacy-preserving audit events. |
| [observability/__init__.py](./observability/__init__.py) | Package marker. |
| [observability/experiment_tracking.py](./observability/experiment_tracking.py) | Adds opt-in MLflow parameter, metric, and artifact logging. |
| [observability/logfire_tracing.py](./observability/logfire_tracing.py) | Adds opt-in Pydantic Logfire inference tracing and nested HTTPX request spans without attaching query/document contents. |
| [observability/metrics.py](./observability/metrics.py) | Writes structured inference/audit JSONL records and tracks provider or estimated token usage. |
| [prompts/registry.py](./prompts/registry.py) | Registers the active versioned RAG prompt templates. |
| [requirements-ml.txt](./requirements-ml.txt) | Declares optional RAGAS, MLflow, and MCP SDK packages. |
| [tests/test_platform_features.py](./tests/test_platform_features.py) | Adds unit tests covering Pydantic request/response schema validation, guardrails, model selection, rate limiting, lineage, metrics, prompt registry, A/B scoring, and sample dataset. |

## 2. Files Modified/Updated

| File | Changes |
|---|---|
| [README.md](./README.md) | Documents new modules, setup, evaluation/MLflow/Logfire, guardrail limitations, Pydantic schema validation, MCP stdio setup, Docker, CI requirements, why embedding-only models are not selectable for chat, and secure Logfire token/region troubleshooting. |
| [app.py](./app.py) | Handles validation/rate-limit errors; escapes user/model/source content before HTML rendering; displays guardrail flags and retrieved chunk lineage; avoids logging raw query text; resets stale selections to an installed chat-capable model. |
| [config/settings.py](./config/settings.py) | Makes LLM models, Ollama URL, and FAISS index path environment-configurable; removes trailing whitespace from the heavy-model default and makes index path project-relative. |
| [inference/inference_engine.py](./inference/inference_engine.py) | Wraps inference with input validation, rate limiting, token/latency/model metrics, audit logging, optional MLflow, cache and retrieval metadata, prompt version, and source references. |
| [ingestion/pdf_loader.py](./ingestion/pdf_loader.py) | Calls lineage metadata helper after chunking and before indexing. |
| [guardrails/input_validator.py](./guardrails/input_validator.py) | Replaces the input dataclass with Pydantic models, validates bounded/normalized queries, and adds a typed inference-result schema. |
| [models/model_factory.py](./models/model_factory.py) | Checks Ollama capabilities and excludes embedding-only models from chat model discovery. |
| [reasoning/context_builder.py](./reasoning/context_builder.py) | Preserves the default string result and optionally returns source metadata and an exact-used-context SHA-256. |
| [reasoning/rag_pipeline.py](./reasoning/rag_pipeline.py) | Uses the prompt registry; optional metadata return includes retrieval counts, context, and source references while the default return shape remains the same. |
| [requirements.txt](./requirements.txt) | Adds base runtime declarations for PyMuPDF, Selenium, webdriver-manager, BeautifulSoup, requests, pyperclip, and Pydantic v2 for request and inference-result validation. |
| [requirements-ml.txt](./requirements-ml.txt) | Adds optional Pydantic Logfire SDK with HTTPX instrumentation. |
| [tests/test_platform_features.py](./tests/test_platform_features.py) | Adds a regression test confirming embedding-only models are excluded from the chat model list. |

## 3. Files Deleted

| File | What was removed |
|---|---|
| `gitignore` | The old misnamed ignore file was replaced by [.gitignore](./.gitignore); its existing ignore rules were retained and evaluation/MLflow exclusions added. |

## 4. Code Changes

The following are actual excerpts from the task's current source files. The “Before” examples are the prior Git version; the “After” blocks are exact current code excerpts.

### Inference entry point and metrics

Before, [inference_engine.py](./inference/inference_engine.py) exposed this `run_inference` signature directly for the route-handling implementation:

```python
def run_inference(
    query: str,
    llm,
    llm_heavy,
    session_state: dict,
    memory_on: bool = True,
    forced_mode: str = None,
    use_hybrid: bool = True,
    progress_callback=None,   # optional fn(msg: str) for UI updates
) -> dict:
```

The body of that function was moved into `_run_inference`; the new public function at [inference_engine.py:245](./inference/inference_engine.py#L245) applies validation/throttling and preserves observability even for blocked or failed requests:

```python
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
    try:
        validated = validate_input(query)
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
        )
        result["guardrail_flags"] = list(validated.findings)
        result["guardrail_flags"].extend(
            validate_output(
                result["answer"], result["route"], result.get("context_used", "")
            )
        )
        metrics.update({
            "route": result["route"],
            "cache_hit": result.get("cache_hit", False),
            "retrieval_method": result.get("retrieval_method") or None,
            "chunks_retrieved": result.get("chunks_retrieved", 0),
            "chunks_used": result.get("chunks_used", 0),
            "prompt_version": result.get("prompt_version"),
        })
        outcome = "success"
        return result
    except (InputValidationError, RateLimitExceeded) as exc:
        metrics["error"] = f"{type(exc).__name__}: {exc}"
        raise
    except Exception as exc:
        outcome = "error"
        metrics["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Inference failed")
        raise
    finally:
        metrics["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
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
        audit_query(session_id, query, metrics["route"], outcome)
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
```

The LLM proxy and structured JSONL writer added in [metrics.py](./observability/metrics.py#L10) are:

```python
class TrackedLLM:
    def __init__(self, llm):
        self._llm = llm
        self.input_tokens = 0
        self.output_tokens = 0
        self.estimated_tokens = False
        self.calls = 0

    def invoke(self, prompt: str):
        response = self._llm.invoke(prompt)
        self.calls += 1
        content = getattr(response, "content", "")
        input_tokens = _usage_count(response, "input_tokens")
        output_tokens = _usage_count(response, "output_tokens")
        if input_tokens is None:
            input_tokens = _token_count(prompt)
            self.estimated_tokens = True
        if output_tokens is None:
            output_tokens = _token_count(content if isinstance(content, str) else str(content))
            self.estimated_tokens = True
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        return response

    def get_model_name(self) -> str:
        get_model_name = getattr(self._llm, "get_model_name", None)
        if callable(get_model_name):
            return str(get_model_name())
        return str(getattr(self._llm, "model", type(self._llm).__name__))
```

```python
def record_inference(metrics: dict) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_id": str(uuid.uuid4()),
        **metrics,
    }
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock, METRICS_FILE.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
```

### Guardrails and rate limiting

The actual validation logic is in [input_validator.py](./guardrails/input_validator.py#L34):

```python
def validate_input(query: str) -> ValidatedInput:
    if not isinstance(query, str) or not query.strip():
        raise InputValidationError("Enter a non-empty query.")
    if len(query) > MAX_QUERY_LENGTH:
        raise InputValidationError(
            f"Query is too long; maximum length is {MAX_QUERY_LENGTH} characters."
        )

    if any(pattern.search(query) for pattern in _INJECTION_PATTERNS):
        raise InputValidationError("The query was blocked by the prompt-injection guardrail.")
    if any(pattern.search(query) for pattern in _TOXIC_PATTERNS):
        raise InputValidationError("The query was blocked by the safety-language guardrail.")

    findings = []
    sanitized = query
    for entity_type, pattern in _PII_PATTERNS.items():
        if pattern.search(sanitized):
            findings.append(f"pii_{entity_type}")
            sanitized = pattern.sub(f"[REDACTED_{entity_type.upper()}]", sanitized)
    return ValidatedInput(text=sanitized, findings=tuple(findings))
```

```python
def validate_output(answer: str, route: str, context: str = "") -> list[str]:
    flags = []
    if len(answer) > 20000:
        flags.append("output_length_exceeded")
    if route == "rag" and context.strip() and answer.strip():
        context_terms = set(re.findall(r"\b[a-zA-Z0-9]{4,}\b", context.lower()))
        answer_terms = set(re.findall(r"\b[a-zA-Z0-9]{4,}\b", answer.lower()))
        overlap = len(context_terms & answer_terms) / max(1, len(answer_terms))
        if overlap < 0.10 and "could not find" not in answer.lower():
            flags.append("grounding_review_required")
    return flags
```

The limiter in [rate_limiter.py](./middleware/rate_limiter.py#L14) uses a locked sliding window:

```python
def check(self, key: str) -> None:
    now = time.monotonic()
    with self._lock:
        requests = self._requests[key]
        while requests and now - requests[0] >= self.window_seconds:
            requests.popleft()
        if len(requests) >= self.limit:
            raise RateLimitExceeded(
                f"Rate limit exceeded: maximum {self.limit} queries per "
                f"{self.window_seconds} seconds."
            )
        requests.append(now)
```

The Streamlit rendering change at [app.py:293](./app.py#L293) now escapes dynamic HTML. For example:

```python
safe_message = html.escape(entry["message"]).replace("\n", "<br>")
st.markdown(f'<div class="user-bubble">{safe_message}</div>', unsafe_allow_html=True)
```

Source URLs are only linked for `http` or `https` schemes; source text/body is escaped, and the link receives `rel="noopener noreferrer"`.

### Lineage, RAG metadata, and prompt registry

Chunk lineage added in [lineage.py](./ingestion/lineage.py#L8):

```python
def attach_lineage_metadata(chunks: list, embedding_model: str = EMBEDDING_MODEL) -> None:
    ingested_at = datetime.now(timezone.utc).isoformat()
    for chunk in chunks:
        chunk.metadata.update({
            "chunk_id": str(uuid.uuid4()),
            "chunk_hash": hashlib.sha256(
                chunk.page_content.encode("utf-8")
            ).hexdigest(),
            "ingested_at": ingested_at,
            "embedding_model": embedding_model,
            "modified_at": None,
        })
```

`pdf_loader.py` calls `attach_lineage_metadata(chunks)` after chunking. Context construction now optionally returns a second value without changing the default string-return API:

```python
# Previous implementation
context = "\n\n---\n\n".join(context_parts)
return context
```

```python
# Current implementation in reasoning/context_builder.py
context = "\n\n---\n\n".join(context_parts)
return (context, source_refs) if include_sources else context
```

Each `source_refs` entry contains `source_file`, `page`, `chunk_id`, `chunk_hash`, `context_sha256`, `ingested_at`, `embedding_model`, `modified_at`, `section`, and `path`.

The RAG pipeline gets its templates through the new registry:

```python
# prompts/registry.py
PROMPT_REGISTRY = {
    "rag_v1": {
        "version": "1.0",
        "created": "2026-10-21",
        "template": RAG_PROMPT,
        "template_no_history": RAG_PROMPT_NO_HISTORY,
    },
}

ACTIVE_RAG_PROMPT = "rag_v1"
```

In [rag_pipeline.py](./reasoning/rag_pipeline.py#L39), `return_metadata=False` preserves the previous `(answer, method)` return. The actual changed prompt/return logic is:

```python
context, source_refs = build_context(docs, include_sources=True)

history_str = format_history(chat_history) if chat_history else ""
prompt_config = get_prompt_version(ACTIVE_RAG_PROMPT)
if history_str and history_str != "None":
    prompt = prompt_config["template"].format(
        history=history_str, context=context, question=query
    )
else:
    prompt = prompt_config["template_no_history"].format(
        context=context, question=query
    )

answer = llm.invoke(prompt).content.strip()
```

When metadata is requested, it returns `(answer, method, retrieved_count, used_source_count, context, source_refs)`:

```python
result = (answer, method)
return (
    *result, retrieved_count, len(source_refs), context, source_refs
) if return_metadata else result
```

### Evaluation and experiment tracking

The A/B implementation in [ab_tester.py](./evaluation/ab_tester.py#L6) accepts model names, an answer function and a score function:

```python
def run_ab_test(
    queries: Iterable[str],
    model_names: tuple[str, str],
    answer_fn: Callable[[str, str], str],
    score_fn: Callable[[str, str], float],
) -> dict:
    """Run paired queries against two models and return aggregate scores."""
    if len(model_names) != 2 or model_names[0] == model_names[1]:
        raise ValueError("A/B testing requires two distinct model names.")

    results = []
    totals = {name: 0.0 for name in model_names}
    counts = {name: 0 for name in model_names}
    for query in queries:
        candidates = {}
        for model_name in model_names:
            answer = answer_fn(model_name, query)
            score = float(score_fn(query, answer))
            if not 0.0 <= score <= 1.0:
                raise ValueError("A/B score function must return a value from 0 to 1.")
            candidates[model_name] = {"answer": answer, "score": score}
            totals[model_name] += score
            counts[model_name] += 1
        winner = max(model_names, key=lambda name: candidates[name]["score"])
        results.append({"query": query, "candidates": candidates, "winner": winner})

    averages = {
        name: totals[name] / counts[name] if counts[name] else 0.0
        for name in model_names
    }
    report = {
        "models": list(model_names),
        "average_scores": averages,
        "winner": max(model_names, key=averages.get) if results else None,
        "results": results,
    }
    log_experiment(
        run_name="paired-model-ab-test",
        params={"models": model_names, "query_count": len(results)},
        metrics={f"average_score_{name}": score for name, score in averages.items()},
    )
    return report
```

The RAGAS evaluator in [eval_suite.py](./evaluation/eval_suite.py#L66) scores the configured metrics and rejects missing or non-finite averages:

```python
result = evaluate(
    Dataset.from_list(examples),
    metrics=[
        faithfulness,
        context_precision,
        answer_relevancy,
        answer_correctness,
    ],
    llm=llm,
    embeddings=embeddings,
)
score_frame = result.to_pandas()
rows = score_frame.to_dict(orient="records")
averages = {
    metric: float(score_frame[metric].dropna().mean())
    for metric in (
        "faithfulness",
        "context_precision",
        "answer_relevancy",
        "answer_correctness",
    )
}
if any(not math.isfinite(score) for score in averages.values()):
    raise RuntimeError("RAGAS returned a missing or non-finite metric score.")
```

`--query-index` builds answer/context rows by calling the RAG pipeline. The CLI exits `1` when average faithfulness is below `--min-faithfulness` (default `0.8`).

MLflow is explicitly opt-in in [experiment_tracking.py](./observability/experiment_tracking.py#L7):

```python
if os.getenv("AI_ENABLE_MLFLOW", "false").lower() != "true":
    return
```

### MCP and configuration

The local stdio tools in [mcp_server.py](./mcp_server.py#L76) are registered with FastMCP:

```python
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
```

The server also registers `list_documents()` and ends with `mcp.run(transport="stdio")`. This is local trusted stdio, not an authenticated network server.

The changed settings in [settings.py](./config/settings.py#L1) now use:

```python
LLM_MODEL = os.getenv("AI_LLM_MODEL", "gemma3:12b")
LLM_MODEL_HEAVY = os.getenv("AI_LLM_MODEL_HEAVY", "gemma3:12b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_INDEX_PATH = os.getenv(
    "VECTOR_INDEX_PATH",
    os.path.join(PROJECT_ROOT, "faiss_index"),
)
```

## 5. Dependencies/Configuration

- **Added to `requirements.txt`:** `pymupdf`, `selenium`, `webdriver-manager`, `beautifulsoup4`, `requests`, `pyperclip`.
- **Declared in `requirements-ml.txt` (optional):** `ragas>=0.2.15,<0.3`, `mlflow>=2.17,<4`, `mcp>=1.0,<2`, plus `-r requirements.txt`.
- **Packages installed during this task:** none.
- **Environment settings:** `AI_LLM_MODEL`, `AI_LLM_MODEL_HEAVY`, `OLLAMA_URL`, and `VECTOR_INDEX_PATH` configure inference/index paths; `AI_QUERY_RATE_LIMIT` sets the in-memory per-session limit (default 30/minute); `AI_ENABLE_MLFLOW`, `MLFLOW_TRACKING_URI`, and `MLFLOW_EXPERIMENT_NAME` configure optional tracking.
- **CI variables:** `RAGAS_DATASET` and `VECTOR_INDEX_PATH` must point to a reviewed evaluation dataset and provisioned index on the self-hosted runner.
- **Pydantic:** `pydantic>=2.7,<3` is declared in `requirements.txt`.
- **Logfire:** `logfire[httpx]>=3.0,<5` is optional in `requirements-ml.txt`; enable with `AI_ENABLE_LOGFIRE=true` and a Logfire project write token in `LOGFIRE_TOKEN`.

### Follow-up: Logfire token setup and 401 troubleshooting docs

Updated the Pydantic Logfire section in `README.md` with PowerShell-specific secure token entry, project write-token permissions, the distinction from Pydantic AI Gateway keys, region matching, app restart requirements, verification in Live view, and recovery steps for 401 errors. No credentials were added to project files.

### Follow-up: embedding model selected for chat

Ollama's `/api/tags` includes embedding models as well as chat models. The selector included every tag and defaulted to the first sorted name when `gemma3:12b` was not installed. In this environment that selected `bge-m3:latest`, which Ollama reports as supporting `embedding` only. Model discovery now checks `/api/show` and only offers models with the `completion` capability; a stale session selection is reset to an available chat model.

```python
# models/model_factory.py
capabilities = model_info.get("capabilities")
if isinstance(capabilities, list):
    return "completion" in capabilities
```

```python
# app.py
if st.session_state["selected_model"] not in available_models:
    st.session_state["selected_model"] = (
        LLM_MODEL if LLM_MODEL in available_models else available_models[0]
    )
```

Added a mocked regression test for model filtering. Local Ollama inspection confirmed `bge-m3:latest` reports `embedding`, `qwen2.5-coder:3b` reports `completion,tools`, and `phi3:latest` reports `completion`.

### Follow-up: Pydantic schema validation

Pydantic is used as structured input/output validation alongside—not instead of—the existing injection, PII, safety-language, and grounding checks. `QueryInput` requires a strict string, strips whitespace, and enforces the existing 4,000-character boundary; Pydantic errors are translated to the existing user-facing `InputValidationError`.

```python
class QueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
```

`InferenceOutput` validates the typed result envelope after inference (including cached results) while allowing extra result metadata for compatibility:

```python
class InferenceOutput(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    answer: str
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
```

`run_inference` now calls `validate_inference_output` before returning its result. Added tests for whitespace normalization, response schema acceptance, and rejecting a response missing its route.

```python
# inference/inference_engine.py
result = validate_inference_output(result)
result["guardrail_flags"] = list(validated.findings)
result["guardrail_flags"].extend(
    validate_output(
        result["answer"], result["route"], result.get("context_used", "")
    )
)
```

### Follow-up: Pydantic Logfire tracing

Added `observability/logfire_tracing.py`. Tracing is disabled unless `AI_ENABLE_LOGFIRE=true`; enabling it requires `LOGFIRE_TOKEN`. The Logfire SDK configures an `ai-doc-system` service and instruments HTTPX clients. Each inference has a parent span, and safe operational metadata is attached at completion:

```python
with logfire.span("ai.inference", requested_route=requested_route) as span:
    yield InferenceTrace(logfire, span)
```

The trace attributes include route/outcome, model names, latency, token counts, cache hit, retrieval method/chunk counts, prompt version, and exception type only. It excludes query text, answers, document context, and session IDs. HTTPX request/response body capture is not enabled. The app's inference wrapper creates this span around request handling and records metrics into it. Added unit tests for default-off behavior and attribute filtering; Logfire export itself requires the user's token/account and was not exercised in offline tests.

### Follow-up: centralized guardrail enforcement

Hardened `guardrails/input_validator.py` and applied it across the main inference path, RAG retrieval, web research, chat memory, and MCP:

- Input validation now normalizes Unicode, rejects detected prompt-injection/safety phrases before routing, redacts recognized PII, and returns safe reason codes for tracing.
- Conversation history is screened before model use and stored/displayed history is redacted; newly entered text is appended only after inference validation returns its sanitized form.
- Retrieved PDF chunks, document-tree titles/bodies, source metadata, and web-research context are screened before being included in generation prompts. Detected injected documents are excluded; detected PII is redacted. RAG prompts explicitly instruct the model to treat context/history as untrusted data.
- Rewritten and model-generated retry queries are revalidated before retrieval. Required screening failures raise a guardrail error and stop the request.
- Generated answers are checked before caching/return. Detected unsafe content, PII, overlong output, or invalid response envelopes are replaced with a fixed safe response. Low lexical grounding remains an informational review signal, not proof of safety or hallucination.
- MCP preflights requests before loading models/indexes and returns a generic refusal on guardrail failures. The legacy `setup_project.py` generator is disabled because its embedded app template bypasses central inference.
- Logfire emits guardrail stage/decision/reason-code/count metadata only; it does not receive query, answer, document, or rejection text.

Added `tests/test_guardrail_enforcement.py` for benign and rejected inputs, document filtering, retry validation, unsafe/invalid output replacement, PII history scrubbing, fail-closed guardrail exceptions, MCP preflight, and legacy-generator disablement. The complete test suite passed: 28 tests.
