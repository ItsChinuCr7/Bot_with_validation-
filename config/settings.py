LLM_MODEL = "phi3"# Primary model — fast, fits in VRAM
# Heavy model — only for research/complex analysis
LLM_MODEL_HEAVY = "llama3:8b-instruct-q4_0"
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_INDEX_PATH = "faiss_index"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 8
FINAL_CONTEXT_DOCS = 4
MAX_CHARS_PER_DOC = 700
MAX_REASONING_DEPTH = 2
# Add these lines
LLM_NUM_PREDICT = 256    # shorter answers = faster
LLM_TEMPERATURE = 0.1