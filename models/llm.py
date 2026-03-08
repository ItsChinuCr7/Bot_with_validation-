import streamlit as st
from langchain_ollama import ChatOllama
from config.settings import LLM_MODEL, OLLAMA_URL

@st.cache_resource
@st.cache_resource
def load_llm():
    return ChatOllama(model="phi3", base_url=OLLAMA_URL,
                      temperature=0.1, num_predict=512)

@st.cache_resource  
def load_llm_heavy():
    return ChatOllama(model="llama3:8b-instruct-q4_0", 
                      base_url=OLLAMA_URL,
                      temperature=0.1, num_predict=1024)
