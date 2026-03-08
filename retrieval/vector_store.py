from langchain_community.vectorstores import FAISS
from config.settings import VECTOR_INDEX_PATH

def create_vector_store(docs, embeddings):
    vector_db = FAISS.from_documents(docs, embeddings)
    vector_db.save_local(VECTOR_INDEX_PATH)
    return vector_db

def load_vector_store(embeddings):
    return FAISS.load_local(
        VECTOR_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
