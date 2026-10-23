import os
import threading
from contextlib import contextmanager
from typing import Generator

_configure_lock = threading.Lock()
_configured_logfire = None


class InferenceTrace:
    def __init__(self, logfire, span):
        self._logfire = logfire
        self._span = span
        self._guardrail_event_count = 0

    def record(self, metrics: dict) -> None:
        error = metrics.get("error")
        attributes = {
            "route": metrics.get("route"),
            "outcome": metrics.get(
                "outcome", "error" if error else "success"
            ),
            "error_type": error.split(":", 1)[0] if error else None,
            "model": metrics.get("model"),
            "models_used": metrics.get("models_used"),
            "latency_ms": metrics.get("latency_ms"),
            "tokens_in": metrics.get("tokens_in"),
            "tokens_out": metrics.get("tokens_out"),
            "cache_hit": metrics.get("cache_hit"),
            "retrieval_method": metrics.get("retrieval_method"),
            "chunks_retrieved": metrics.get("chunks_retrieved"),
            "chunks_used": metrics.get("chunks_used"),
            "prompt_version": metrics.get("prompt_version"),
        }
        for name, value in attributes.items():
            if value is not None:
                self._span.set_attribute(f"ai.{name}", value)

        self._logfire.info("Inference completed", **{
            name: value for name, value in attributes.items() if value is not None
        })

    def record_guardrail(
        self,
        stage: str,
        decision: str,
        reason_code: str,
        count: int = 0,
    ) -> None:
        event_id = self._guardrail_event_count
        self._guardrail_event_count += 1
        safe_fields = {
            "guardrail_stage": stage,
            "guardrail_decision": decision,
            "guardrail_reason_code": reason_code,
            "guardrail_count": count,
        }
        prefix = f"ai.guardrail_events.{event_id}"
        for name, value in safe_fields.items():
            self._span.set_attribute(f"{prefix}.{name}", value)
        self._logfire.info("Guardrail decision", **safe_fields)

    def record_safety_outcome(self, decision: str, reason_code: str) -> None:
        fields = {
            "safety_decision": decision,
            "safety_reason_code": reason_code,
        }
        for name, value in fields.items():
            self._span.set_attribute(f"ai.{name}", value)
        self._logfire.info("Overall safety outcome", **fields)


def _get_logfire():
    global _configured_logfire
    if os.getenv("AI_ENABLE_LOGFIRE", "false").lower() != "true":
        return None

    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        raise RuntimeError(
            "AI_ENABLE_LOGFIRE is true but LOGFIRE_TOKEN is not set. "
            "Create a Logfire write token or disable Logfire tracing."
        )

    with _configure_lock:
        if _configured_logfire is None:
            try:
                import logfire
            except ModuleNotFoundError as exc:
                if exc.name != "logfire":
                    raise
                raise RuntimeError(
                    "Logfire tracing is enabled but the SDK is not installed. "
                    "Install it with: pip install -r requirements-ml.txt"
                ) from exc
            logfire.configure(
                token=token,
                send_to_logfire=True,
                service_name=os.getenv("LOGFIRE_SERVICE_NAME", "ai-doc-system"),
            )
            logfire.instrument_httpx()
            _configured_logfire = logfire
    return _configured_logfire


@contextmanager
def inference_trace(
    requested_route: str,
) -> Generator[InferenceTrace | None, None, None]:
    logfire = _get_logfire()
    if logfire is None:
        yield None
        return

    with logfire.span("ai.inference", requested_route=requested_route) as span:
        yield InferenceTrace(logfire, span)


def record_guardrail_decision(
    requested_route: str,
    stage: str,
    decision: str,
    reason_code: str,
    count: int = 0,
) -> None:
    """Record a safe standalone decision for requests rejected before inference setup."""
    try:
        with inference_trace(requested_route) as trace:
            if trace is not None:
                trace.record_guardrail(stage, decision, reason_code, count)
                if decision in {"blocked", "error"}:
                    trace.record_safety_outcome(
                        "error" if decision == "error" else "blocked",
                        reason_code,
                    )
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "Guardrail telemetry unavailable (%s)", type(exc).__name__
        )
