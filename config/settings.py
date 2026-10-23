import os

LLM_MODEL = os.getenv("AI_LLM_MODEL", "gemma3:12b")  # Primary model
# Heavy model — only for research/complex analysis
LLM_MODEL_HEAVY = os.getenv("AI_LLM_MODEL_HEAVY", "gemma3:12b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_INDEX_PATH = os.getenv(
    "VECTOR_INDEX_PATH",
    os.path.join(PROJECT_ROOT, "faiss_index"),
)
CHUNK_SIZE = 450
CHUNK_OVERLAP = 160
TOP_K_RETRIEVAL = 8
FINAL_CONTEXT_DOCS = 4
MAX_CHARS_PER_DOC = 1200
MAX_REASONING_DEPTH = 2
MIN_RELEVANCE_SCORE = 0.0
HIGH_RELEVANCE_SCORE = 3.0
# Add these lines
LLM_NUM_PREDICT = 2000# shorter answers = faster
LLM_TEMPERATURE = 0.5