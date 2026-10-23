import json
import math
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

METRICS_FILE = Path(__file__).resolve().parents[1] / "logs" / "inference_metrics.jsonl"
_write_lock = threading.Lock()


def _token_count(text: str) -> int:
    return math.ceil(len(text) / 4) if text else 0


def _usage_count(response, name: str) -> int | None:
    usage = getattr(response, "usage_metadata", None) or {}
    value = usage.get(name)
    if value is None:
        response_metadata = getattr(response, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage") or {}
        value = token_usage.get(name)
        if value is None:
            value = token_usage.get(
                "prompt_tokens" if name == "input_tokens" else "completion_tokens"
            )
    return value if isinstance(value, int) and value >= 0 else None


class TrackedLLM:
    """Transparent LLM proxy that accumulates provider or estimated token usage."""

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


def record_inference(metrics: dict) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_id": metrics.get("request_id") or str(uuid.uuid4()),
        **metrics,
    }
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock, METRICS_FILE.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def record_audit_event(event: dict) -> None:
    audit_file = METRICS_FILE.with_name("query_audit.jsonl")
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock, audit_file.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
