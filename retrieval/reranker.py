import math

import streamlit as st
from sentence_transformers import CrossEncoder
from guardrails.input_validator import GuardrailCheckError
from config.settings import (
    FINAL_CONTEXT_DOCS,
    MIN_RELEVANCE_SCORE,
)


@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")


def score_docs(query, docs):
    try:
        if not docs:
            return []
        reranker = load_reranker()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = [float(score) for score in reranker.predict(pairs)]
        if len(scores) != len(docs) or any(not math.isfinite(score) for score in scores):
            raise ValueError("Relevance model returned invalid scores.")
        return sorted(zip(scores, docs), key=lambda item: item[0], reverse=True)
    except Exception as exc:
        raise GuardrailCheckError("Retrieval relevance screening failed.") from exc


def rerank_docs(
    query,
    docs,
    top_k=FINAL_CONTEXT_DOCS,
    min_score: float | None = None,
):
    threshold = MIN_RELEVANCE_SCORE if min_score is None else min_score
    ranked = score_docs(query, docs)
    return [
        doc for score, doc in ranked if score >= threshold
    ][:top_k]
