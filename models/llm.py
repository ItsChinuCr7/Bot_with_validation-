# ============================================================
# models/llm.py
# LLM loader — uses model_factory for dynamic model creation.
# ============================================================

import logging
import streamlit as st
from models.model_factory import create_llm
from config.settings import LLM_MODEL, LLM_MODEL_HEAVY

logger = logging.getLogger(__name__)


@st.cache_resource
def load_llm(model_name: str = None):
    """Load and cache the light LLM."""
    name = model_name or LLM_MODEL
    logger.info(f"Loading light LLM: {name}")
    return create_llm(model_name=name, mode="light")


@st.cache_resource
def load_llm_heavy(model_name: str = None):
    """Load and cache the heavy LLM."""
    name = model_name or LLM_MODEL_HEAVY
    logger.info(f"Loading heavy LLM: {name}")
    return create_llm(model_name=name, mode="heavy")


def reload_llm(model_name: str):
    """Force reload LLM with a new model — clears Streamlit cache."""
    load_llm.clear()
    load_llm_heavy.clear()
    return load_llm(model_name)