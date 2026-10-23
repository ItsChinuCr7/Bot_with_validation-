# ============================================================
# models/model_factory.py
# Dynamic model factory.
# - Discovers all models installed in Ollama automatically
# - Creates LLM instances by name
# - No need to edit settings.py to switch models
# ============================================================

import logging
import requests
from models.base_llm import OllamaLLMWrapper
from config.settings import OLLAMA_URL, LLM_MODEL, LLM_MODEL_HEAVY

logger = logging.getLogger(__name__)

# Default params per model size category
MODEL_PARAMS = {
    "light": {"temperature": 0.7, "num_predict": 4096},
    "heavy": {"temperature": 0.7, "num_predict": 4096},
}

# Known heavy models (need more VRAM)
HEAVY_MODELS = {"llama3", "llama2", "mixtral", "gemma2", "mistral-large", "qwen2"}


def get_available_models() -> list[str]:
    """
    Query Ollama for locally installed models that support chat completion.
    Falls back to settings defaults if Ollama is unreachable.
    """
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            installed_models = [m["name"] for m in resp.json().get("models", [])]
            models = [name for name in installed_models if _supports_chat(name)]
            logger.info(f"Model factory: found {len(models)} Ollama models")
            return sorted(models) if installed_models else [LLM_MODEL, LLM_MODEL_HEAVY]
    except Exception as e:
        logger.warning(f"Could not reach Ollama API: {e}")
    return [LLM_MODEL, LLM_MODEL_HEAVY]


def _supports_chat(model_name: str) -> bool:
    """Exclude embedding-only Ollama models from the chat model selector."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/show",
            json={"name": model_name},
            timeout=3,
        )
        response.raise_for_status()
        model_info = response.json()
    except requests.RequestException as exc:
        logger.warning("Could not inspect Ollama model %s: %s", model_name, exc)
        return False

    capabilities = model_info.get("capabilities")
    if isinstance(capabilities, list):
        return "completion" in capabilities

    details = model_info.get("details") or {}
    family = str(details.get("family", "")).lower()
    return family != "bert" and "embed" not in model_name.lower()


def is_heavy_model(model_name: str) -> bool:
    """Check if a model needs heavy resources."""
    name = model_name.lower()
    return any(h in name for h in HEAVY_MODELS)


def create_llm(model_name: str = None, mode: str = "light"):
    """
    Create an LLM instance by model name.

    Args:
        model_name: Ollama model name e.g. "phi3", "llama3:8b"
                    If None, uses settings default.
        mode:       "light" or "heavy" — affects default params.

    Returns:
        OllamaLLMWrapper instance
    """
    if not model_name:
        model_name = LLM_MODEL_HEAVY if mode == "heavy" else LLM_MODEL

    params = MODEL_PARAMS.get(mode, MODEL_PARAMS["light"])
    logger.info(f"Creating LLM: {model_name} (mode={mode})")

    return OllamaLLMWrapper(
        model       = model_name,
        base_url    = OLLAMA_URL,
        temperature = params["temperature"],
        num_predict = params["num_predict"],
    )


def get_model_info(model_name: str) -> dict:
    """Get size/parameter info for a model from Ollama."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/show",
            json={"name": model_name},
            timeout=3,
        )
        if resp.status_code == 200:
            data = resp.json()
            details = data.get("details", {})
            return {
                "name":       model_name,
                "parameters": details.get("parameter_size", "unknown"),
                "family":     details.get("family", "unknown"),
                "quantization": details.get("quantization_level", "unknown"),
            }
    except Exception:
        pass
    return {"name": model_name, "parameters": "unknown", "family": "unknown", "quantization": "unknown"}