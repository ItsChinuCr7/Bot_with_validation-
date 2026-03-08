from config.settings import TOP_K_RETRIEVAL

def get_mmr_retriever(vector_db):
    return vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K_RETRIEVAL,
            "fetch_k": TOP_K_RETRIEVAL * 2,
            "lambda_mult": 0.6,
        },
    )
