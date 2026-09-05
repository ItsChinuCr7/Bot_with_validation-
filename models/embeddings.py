import streamlit as st
from config.settings import EMBEDDING_MODEL, OLLAMA_URL

@st.cache_resource
def load_embeddings(model_name: str = None):
    name = model_name or EMBEDDING_MODEL
    # If using local Ollama model (e.g. bge-m3)
    if "bge-m3" in name.lower() or "/" not in name:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=name,
            base_url=OLLAMA_URL,
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
