# 🤖 Bot with Validation — AI Document Intelligence Assistant

An advanced RAG system with self-healing answers, web research, and permanent mistake memory.

---

## ✨ Features

- **Hybrid Search** — BM25 keyword + FAISS vector search combined
- **Document Tree Navigation** — Vectorless reasoning-first retrieval
- **Web Research** — Scrapes DuckDuckGo + Gemini via Selenium (no API key needed)
- **Self-Healing** — Answers validated by Gemini, mistakes saved permanently
- **Conversation Memory** — Remembers last 3 turns
- **Answer Mode Selector** — Auto / Research / General / RAG
- **Permanent Learning** — Past mistakes stored in JSON, injected into future prompts
- **Inference Observability** — JSONL metrics and privacy-preserving query audit events
- **Evaluation & A/B Testing** — RAGAS evaluation suite, 20 synthetic examples, paired model comparison
- **Guardrails** — Injection and safety-pattern checks, PII redaction, and grounding review flags
- **Schema validation** — Pydantic validates normalized query inputs and inference response shapes
- **Data Lineage** — Chunk IDs, hashes, source/page, embedding model, and ingestion timestamps
- **Prompt Versioning** — Versioned RAG prompt templates
- **Optional MLflow** — Per-inference experiment tracking when enabled
- **Optional Pydantic Logfire** — Inference spans, safe operational attributes, and nested HTTPX request timings
- **FastMCP** — Local stdio tools for document Q&A, general Q&A, web research, and document listing
- **Docker & CI** — Streamlit container and automated Python checks

---

## 🏗️ Architecture

```
Query
  ├── Auto-routing (chat / calc / research / general / RAG)
  ├── Document Tree → structure-aware retrieval
  ├── Hybrid Search → BM25 + Vector + MMR re-ranking
  ├── Web Research  → DuckDuckGo + Gemini synthesis
  └── Self-Healing  → Gemini validates → corrects → saves lesson
```

---

## 📁 Project Structure

```
ai_doc_system/
├── app.py                    ← Main Streamlit app
├── mcp_server.py             ← Optional local stdio MCP tools
├── Dockerfile                ← Reproducible Streamlit container
├── requirements.txt
├── requirements-ml.txt       ← Optional RAGAS, MLflow, and MCP packages
├── .github/workflows/ml-quality.yml
├── agents/
│   └── agent_router.py       ← Smart query routing
├── config/
│   └── settings.py
├── evaluation/
│   ├── answer_validator.py
│   ├── self_healer.py        ← Gemini validator + permanent memory
│   ├── eval_suite.py         ← RAGAS evaluation CLI
│   ├── ab_tester.py
│   └── datasets/sample_rag_eval.json
├── guardrails/
│   └── input_validator.py
├── ingestion/
│   ├── document_tree.py      ← Hierarchical document tree builder
│   ├── pdf_loader.py
│   └── lineage.py
├── middleware/
│   └── rate_limiter.py
├── models/
│   ├── embeddings.py
│   └── llm.py
├── observability/
│   ├── metrics.py
│   └── experiment_tracking.py
├── prompts/
│   ├── templates.py
│   └── registry.py
├── reasoning/
│   └── rag_pipeline.py
├── research/
│   └── web_researcher.py     ← Selenium scraper (DuckDuckGo + Gemini)
└── retrieval/
    ├── hybrid_search.py
    ├── smart_retriever.py
    ├── tree_retriever.py
    └── vector_store.py
```

---

## 🚀 Setup

### 1. Install dependencies
```powershell
pip install -r requirements.txt
```

### 2. Install optional evaluation, tracking, and MCP dependencies
```powershell
pip install -r requirements-ml.txt
```

This extra installs RAGAS, MLflow, the MCP Python SDK, and Pydantic Logfire with HTTPX instrumentation. These integrations are optional for the Streamlit app.

### 3. Install Ollama models
```powershell
ollama pull gemma3:12b
```

The app's model selector lists chat-capable Ollama models only. Embedding-only models such as `bge-m3` are excluded because they cannot answer chat prompts.

### 4. Start Brave with debug port (for Gemini features)
```powershell
& "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" --remote-debugging-port=9222
```
Log into Gemini in that Brave window, then minimize it.

### 5. Run the app
```powershell
cd D:\ai_doc_system
streamlit run app.py
```

## Evaluation and experiments

The checked-in 20-row example dataset is synthetic and is only a starting point for the evaluator; it does not claim to represent the currently indexed medical guideline. Use representative, reviewed examples from your actual documents for meaningful quality results.

```powershell
python -m evaluation.eval_suite --dataset evaluation/datasets/sample_rag_eval.json --min-faithfulness 0.8
```

The evaluator uses the locally installed Ollama model and downloads the configured embedding model if needed. It reports faithfulness, context precision, answer relevance, and answer correctness. Use `--query-index` to generate answers and retrieved contexts from the live FAISS index; without it, provide precomputed `answer` and `contexts` fields. The command exits non-zero when faithfulness is below the threshold.

Run paired model comparisons with `evaluation.ab_tester.run_ab_test`, supplying an answer function and scorer. For automatic tracking of app inferences, set `AI_ENABLE_MLFLOW=true`; configure `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` as needed. The default tracking store is the local `mlruns/` directory.

### Pydantic Logfire tracing

Logfire is opt-in. It exports inference traces and operational metadata to your Logfire project.

1. Sign in to the same Logfire region where your project exists (`logfire-us.pydantic.dev` or `logfire-eu.pydantic.dev`).
2. Open that project, then go to **Settings → Write tokens → New write token**. Use a project write token (or a project API key with **Send telemetry** permission). A Pydantic AI Gateway/universal key is for hosted model requests and is **not** a Logfire telemetry token.
3. Install the optional SDK if it is not already installed:

   ```powershell
   pip install "logfire[httpx]>=3.0,<5"
   ```

4. Open **PowerShell** (not Command Prompt), change to the project directory, and enter the token through the secure prompt. Do not paste it into source code, README, or chat:

```powershell
Set-Location D:\ai_doc_system
$env:AI_ENABLE_LOGFIRE = "true"
$secureToken = Read-Host "Logfire project write token" -AsSecureString
$env:LOGFIRE_TOKEN = [System.Net.NetworkCredential]::new("", $secureToken).Password
$env:LOGFIRE_SERVICE_NAME = "ai-doc-system"
streamlit run app.py
```

Each inference emits an `ai.inference` span with route, model, latency, token counts, cache/retrieval details, and outcome. HTTPX spans are nested where applicable, so Ollama/API request timing is visible. Query text, answers, document context, and session IDs are not attached by this integration. HTTPX request/response body capture is not enabled. **When enabled, trace metadata is transmitted to your configured Logfire project.** Keep the write token secret; disabling `AI_ENABLE_LOGFIRE` stops this export. `LOGFIRE_TOKEN` is a Logfire write token, not an Ollama or model-provider key.

Keep the app running in that same PowerShell window after setting the variables. To verify, submit one app query and check the selected project's **Live** view for `ai.inference`.

If export reports `401 Unauthorized` / `Invalid token`:

- Stop the app and verify the token was created under the exact project whose **Live** view you are checking.
- Verify that the account/project region matches the project URL. US and EU Logfire regions are separate; use the project’s own region when creating/signing in to the account.
- Revoke any token pasted into chat or another public place, create a fresh project write token, and enter it into a new PowerShell session using the secure prompt above. A Gateway API key will not work for `LOGFIRE_TOKEN`.
- Restart Streamlit from that same PowerShell session. The Logfire SDK derives the hosted region from a valid project token; do not set a different endpoint manually.

To disable remote export, stop Streamlit and run `$env:AI_ENABLE_LOGFIRE = "false"` before restarting. The token is only set for the current PowerShell session by the example above.

## Guardrails, telemetry, and limits

- Inference metrics are appended to `logs/inference_metrics.jsonl`; token counts use provider usage when available and otherwise a character-based estimate.
- Audit events are appended to `logs/query_audit.jsonl`; accepted requests store a SHA-256 hash of the PII-redacted query and a session identifier, not the raw query. Rejected input is not hashed.
- `AI_QUERY_RATE_LIMIT` sets the in-memory, per-process/per-session sliding-window limit (default: 30 requests per minute).
- Central inference validation runs before routing, retrieval, or model invocation. It rejects detected prompt-injection/safety patterns, redacts detected email, phone, and simple self-identifying-name patterns, and screens saved conversation history before reuse. These remain local heuristics, not a dedicated PII/toxicity classifier.
- Retrieved PDF and research content is treated as untrusted data: instruction-like content is excluded before generation, detected PII is redacted, and prompts explicitly delimit document/research context as data rather than instructions. Tree titles and bodies are screened before the tree-navigation model sees them.
- Generated responses are checked for detected unsafe language, PII, and excessive length before return or caching. Rejected outputs are replaced with a fixed safe response; grounding overlap remains an informational review signal and is not itself a safety verdict.
- Logfire records guardrail stage, decision, safe reason code, and count only; it does not attach query text, answers, document text, or rejection details. The legacy `setup_project.py` generator is disabled because its embedded app template bypasses the central guardrails.
- Pydantic enforces query constraints and the inference result schema; it does not identify prompt injection, toxicity, PII, or hallucinations by itself.
- RAG grounding is flagged using lexical context overlap. It is a review signal, not proof that an answer is or is not hallucinated.
- Chunk lineage is retained as FAISS document metadata and exact context excerpts receive a SHA-256 fingerprint. Existing indexes predate these fields and need to be rebuilt by reprocessing their PDFs.

## Optional MCP server

Run `python mcp_server.py` from the project directory after installing `requirements-ml.txt`. Configure the command and absolute script path in your MCP client. It exposes `ask_documents`, `ask_general`, `research_web`, and `list_documents` over **local stdio**. This trusted local transport has no API-key authentication; do not expose it as a network service. The server uses the local Ollama endpoint and FAISS index as the app.

Example Claude Desktop entry (adjust the interpreter and project paths):

```json
{
  "mcpServers": {
    "ai-doc-system": {
      "command": "python",
      "args": ["D:\\ai_doc_system\\mcp_server.py"]
    }
  }
}
```

## Docker

```powershell
docker build -t ai-doc-assistant .
docker run --rm -p 8501:8501 -v "${PWD}\faiss_index:/app/faiss_index" ai-doc-assistant
```

Ollama must run separately on the host and have the configured model available. `OLLAMA_URL` defaults to `http://host.docker.internal:11434`. The mounted index directory persists embeddings; uploaded PDFs, local logs, experiment runs, and personal memory are excluded from the image.

## Continuous integration

`.github/workflows/ml-quality.yml` compiles Python sources and runs offline unit tests on GitHub-hosted runners for pushes and pull requests. Its full RAGAS faithfulness gate runs on trusted pushes and can also be manually dispatched, using a Windows self-hosted runner with `requirements-ml.txt` and the Ollama model. Configure repository variables `RAGAS_DATASET` (reviewed JSON with `question` and `ground_truth` fields) and `VECTOR_INDEX_PATH` (provisioned local FAISS directory). The gate fails when faithfulness is below `0.8`. Do not expose a self-hosted runner to untrusted fork pull requests.

---

## 🧠 How Self-Healing Works

```
1. The configured Ollama model generates an answer
2. Answer sent to Gemini via your logged-in Brave
3. Gemini validates: CORRECT / PARTIALLY_CORRECT / WRONG
4. If wrong → Gemini corrected answer shown (green bubble)
5. Mistake + lesson saved to self_healing_memory.json permanently
6. Next similar question → lesson injected into prompt automatically
```

---

## 📋 Requirements

- Windows 10/11
- Python 3.10+
- Ollama running locally
- Brave Browser
- 8GB RAM minimum

---

## 👤 Author

**ItsChinuCr7**
