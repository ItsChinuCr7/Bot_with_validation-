LLM_MODEL = "gemma3:12b"# Primary model — fast, fits in VRAM
# Heavy model — only for research/complex analysis
LLM_MODEL_HEAVY = "gemma3:12b "
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_INDEX_PATH = "faiss_index"
CHUNK_SIZE = 450
CHUNK_OVERLAP = 160
TOP_K_RETRIEVAL = 8
FINAL_CONTEXT_DOCS = 4
MAX_CHARS_PER_DOC = 1200
MAX_REASONING_DEPTH = 2
# Add these lines
LLM_NUM_PREDICT = 2000# shorter answers = faster
LLM_TEMPERATURE = 0.5