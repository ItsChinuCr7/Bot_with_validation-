import streamlit as st
from sentence_transformers import CrossEncoder
from config.settings import FINAL_CONTEXT_DOCS

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

def rerank_docs(query, docs, top_k=FINAL_CONTEXT_DOCS):
    reranker = load_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]
