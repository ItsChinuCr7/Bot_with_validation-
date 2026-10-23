from langchain_community.vectorstores import FAISS
from config.settings import EMBEDDING_MODEL, VECTOR_INDEX_PATH

def create_vector_store(docs, embeddings):
    vector_db = FAISS.from_documents(docs, embeddings)
    vector_db.save_local(VECTOR_INDEX_PATH)
    return vector_db

def load_vector_store(embeddings):
    vector_db = FAISS.load_local(
        VECTOR_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    documents = list(vector_db.docstore._dict.values())
    indexed_models = {
        doc.metadata.get("embedding_model")
        for doc in documents
        if isinstance(doc.metadata.get("embedding_model"), str)
    }
    if indexed_models and indexed_models != {EMBEDDING_MODEL}:
        raise ValueError(
            "Saved FAISS index uses a different embedding model; reprocess the PDFs."
        )
    query_dimension = len(embeddings.embed_query("embedding dimension check"))
    if query_dimension != vector_db.index.d:
        raise ValueError(
            "Saved FAISS index dimension does not match the configured embeddings; "
            "reprocess the PDFs."
        )
    return vector_db
