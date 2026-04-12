# ============================================================
# models/base_llm.py
# Common LLM interface — all models go through this.
# Swap the underlying model without touching anything else.
# ============================================================

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract base class for all LLM wrappers."""

    @abstractmethod
    def invoke(self, prompt: str) -> object:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass


class OllamaLLMWrapper(BaseLLM):
    """Wraps ChatOllama with the BaseLLM interface."""

    def __init__(self, model: str, base_url: str, temperature: float, num_predict: int):
        from langchain_ollama import ChatOllama
        self._model_name = model
        self._llm = ChatOllama(
            model       = model,
            base_url    = base_url,
            temperature = temperature,
            num_predict = num_predict,
        )
        logger.info(f"LLM loaded: {model} | temp={temperature} | num_predict={num_predict}")

    def invoke(self, prompt: str):
        return self._llm.invoke(prompt)

    def get_model_name(self) -> str:
        return self._model_name