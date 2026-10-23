# Change Handoff for Claude

Generated: 2026-10-21

This file records the implementation changes made for the AI/ML engineering feature set. Line references point to the final files in this workspace. For modified tracked files, added/deleted counts come from the file diff. For newly created files, the full file line count is listed as added.

## Summary

- Added structured inference metrics (`logs/inference_metrics.jsonl`) and privacy-preserving query audit records (`logs/query_audit.jsonl`).
- Added local input/output guardrails, per-session sliding-window rate limiting, and HTML escaping for chat and scraped-source rendering.
- Added RAGAS evaluation, a 20-row synthetic example dataset, live FAISS-index evaluation support, paired A/B testing, and optional MLflow tracking.
- Added chunk-level lineage and exact-context SHA-256 references in the Streamlit UI and MCP document answers.
- Added a versioned RAG prompt registry and environment-driven model/index configuration.
- Added a local stdio FastMCP server exposing four tools, a Dockerfile, and GitHub Actions workflows.

## File-by-file changes

| File | Final line references | Added | Removed | What changed |
|---|---:|---:|---:|---|
| [README.md](./README.md) | 16–197 | 92 | 9 | Documents setup, new modules, eval/MLflow commands, guardrail limits, MCP configuration, Docker, and CI prerequisites. |
| [app.py](./app.py) | 5–21, 293–387 | 38 | 15 | Handles expected validation/rate-limit errors, displays lineage and guardrail flags, escapes dynamic HTML, and stops logging raw query text. |
| [config/settings.py](./config/settings.py) | 1–12 | 10 | 4 | Reads model and Ollama settings from environment variables, fixes the heavy-model trailing space, and makes the FAISS path project-relative/configurable. |
| [inference/inference_engine.py](./inference/inference_engine.py) | 10–347 | 146 | 4 | Wraps inference with validation, throttling, token/model/latency metrics, audit logging, optional MLflow tracking, cache/retrieval stats, prompt version, and source lineage. |
| [ingestion/pdf_loader.py](./ingestion/pdf_loader.py) | 16, 92 | 2 | 0 | Attaches lineage to generated chunks before vector indexing. |
| [ingestion/lineage.py](./ingestion/lineage.py) | 1–19 | 19 | 0 | Creates chunk ID, SHA-256, ingestion timestamp, embedding model, and modification metadata. |
| [reasoning/context_builder.py](./reasoning/context_builder.py) | 1–43 | 18 | 2 | Optionally returns exact source references and context excerpt hashes while preserving the original string-return behavior. |
| [reasoning/rag_pipeline.py](./reasoning/rag_pipeline.py) | 15–130 | 17 | 8 | Uses the prompt registry and optionally returns retrieval counts, used context, and source references. Default return shape remains unchanged. |
| [prompts/registry.py](./prompts/registry.py) | 1–19 | 19 | 0 | Registers the active `rag_v1` template and version metadata. |
| [guardrails/input_validator.py](./guardrails/input_validator.py) | 1–66 | 66 | 0 | Validates query length, blocks selected injection/threat patterns, redacts email/phone/simple name patterns, and flags weak RAG grounding or long output. |
| [guardrails/__init__.py](./guardrails/__init__.py) | empty | 0 | 0 | Package marker. |
| [middleware/rate_limiter.py](./middleware/rate_limiter.py) | 1–50 | 50 | 0 | Implements in-memory per-session rate limiting and JSONL audit events containing query hashes rather than raw queries. |
| [middleware/__init__.py](./middleware/__init__.py) | empty | 0 | 0 | Package marker. |
| [observability/metrics.py](./observability/metrics.py) | 1–78 | 78 | 0 | Writes structured JSONL metrics and wraps LLM calls to capture provider usage or explicitly estimated token counts. |
| [observability/experiment_tracking.py](./observability/experiment_tracking.py) | 1–44 | 44 | 0 | Adds opt-in MLflow logging for inference, evaluation, and A/B runs. |
| [observability/__init__.py](./observability/__init__.py) | empty | 0 | 0 | Package marker. |
| [evaluation/eval_suite.py](./evaluation/eval_suite.py) | 1–178 | 178 | 0 | Adds the RAGAS CLI for faithfulness, context precision, answer relevance, and correctness; can evaluate prepared rows or generate answers from the configured FAISS index. |
| [evaluation/ab_tester.py](./evaluation/ab_tester.py) | 1–47 | 47 | 0 | Runs paired queries across two model names with a supplied answer function/scorer and logs aggregate scores when MLflow is enabled. |
| [evaluation/datasets/sample_rag_eval.json](./evaluation/datasets/sample_rag_eval.json) | 1–22 | 22 | 0 | Adds 20 synthetic Q&A examples; these are illustrative and not a clinical benchmark. |
| [mcp_server.py](./mcp_server.py) | 1–111 | 111 | 0 | Adds local stdio tools for document Q&A, general Q&A, web research, and indexed-document listing. |
| [requirements.txt](./requirements.txt) | 11–16 | 6 | 0 | Adds runtime declarations for PyMuPDF, Selenium, webdriver-manager, BeautifulSoup, requests, and pyperclip. |
| [requirements-ml.txt](./requirements-ml.txt) | 1–4 | 4 | 0 | Adds optional RAGAS, MLflow, and MCP SDK dependency ranges. |
| [Dockerfile](./Dockerfile) | 1–24 | 24 | 0 | Adds a non-root Python 3.12 Streamlit container using the host Ollama endpoint. |
| [.dockerignore](./.dockerignore) | 1–15 | 15 | 0 | Excludes indexes, PDFs, logs, credentials, and local state from the Docker build context. |
| [.gitignore](./.gitignore) | 1–56 | 56 | 0 | Replaces the tracked misnamed `gitignore` file and adds local evaluation/MLflow output exclusions. |
| [gitignore](./gitignore) | 1–52 | 0 | 52 | Old path removed as part of the rename to `.gitignore`; existing ignore rules were retained. |
| [.github/workflows/ml-quality.yml](./.github/workflows/ml-quality.yml) | 1–51 | 51 | 0 | Runs offline compile/unit checks on pushes and PRs, plus the configured RAGAS gate on trusted pushes/manual runs using a Windows self-hosted runner. |
| [tests/test_platform_features.py](./tests/test_platform_features.py) | 1–119 | 119 | 0 | Adds 10 unit tests for guardrails, rate limits, lineage, metrics, prompts, A/B scoring, and the sample dataset. |

## Removed or replaced

- No existing application feature was intentionally removed.
- Replaced the README's one-off package install command and older Ollama model instructions with the requirements files and current configured model.
- Removed raw query text from the app log message; audit logs retain only a SHA-256 hash.
- Replaced raw dynamic HTML interpolation for chat/source content with escaped content and HTTP(S)-only links.
- Replaced the incorrectly named `gitignore` path with `.gitignore`; retained its existing rules and added exclusions for evaluation results and MLflow runs.
- Replaced hard-coded model/index endpoint values with environment-aware settings; removed the trailing whitespace from the heavy-model name.

## Packages

- **Installed during this task:** none.
- **Added to the base manifest:** `pymupdf`, `selenium`, `webdriver-manager`, `beautifulsoup4`, `requests`, `pyperclip`.
- **Declared as optional in `requirements-ml.txt`:** `ragas>=0.2.15,<0.3`, `mlflow>=2.17,<4`, `mcp>=1.0,<2`.
- Install optional integrations with `pip install -r requirements-ml.txt`. MLflow inference logging remains opt-in with `AI_ENABLE_MLFLOW=true`.

## Verification

- 10 offline unit tests passed.
- 16 changed Python files compiled successfully.
- Mocked general inference and mocked RAG retrieval/provenance flows passed.
- Focused `git diff --check` passed; Git reported only a line-ending normalization notice for `ingestion/pdf_loader.py`.
- Full RAGAS evaluation, MCP startup, and Docker image build were not run: the optional RAGAS/MCP packages were not installed, and no live Ollama/index workflow was started.

## Configuration and limitations to tell Claude

- The MCP service is intentionally **local stdio without API-key auth**, per the selected transport. Do not expose it as a network service.
- RAGAS uses the synthetic dataset only for evaluator smoke tests. For real quality gating, configure a reviewed dataset and provision the matching FAISS index.
- The GitHub RAGAS gate requires a trusted Windows self-hosted runner, repository variables `RAGAS_DATASET` and `VECTOR_INDEX_PATH`, the optional dependencies, and a locally available Ollama model. The workflow deliberately avoids running a self-hosted runner against untrusted fork PRs.
- Token counts fall back to a documented character-based estimate when provider usage is absent. PII/toxicity and grounding checks are lightweight heuristics, not certified classifiers.
- Existing FAISS indexes need to be rebuilt from PDFs to add lineage metadata.
- These pre-existing dirty/generated files were present before implementation and were preserved, not included as task changes: tracked `__pycache__` files, `data/cache/llm_cache.json`, `faiss_index/index.faiss`, `faiss_index/index.pkl`, and `logs/app.log`.
