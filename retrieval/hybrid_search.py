from rank_bm25 import BM25Okapi

class HybridSearcher:
    def __init__(self, docs, vector_db, alpha=0.5):
        self.docs = docs
        self.vector_db = vector_db
        self.alpha = alpha
        tokenized = [doc.page_content.lower().split() for doc in docs]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, top_k=8):
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_norm = bm25_scores / bm25_max

        vector_results = self.vector_db.similarity_search_with_score(query, k=top_k * 2)
        vector_score_map = {}
        for doc, score in vector_results:
            for i, d in enumerate(self.docs):
                if d.page_content == doc.page_content:
                    vector_score_map[i] = 1.0 / (1.0 + score)
                    break

        combined = []
        for i, doc in enumerate(self.docs):
            v_score = vector_score_map.get(i, 0.0)
            b_score = float(bm25_norm[i])
            hybrid_score = self.alpha * v_score + (1 - self.alpha) * b_score
            combined.append((hybrid_score, doc))

        combined.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in combined[:top_k]]
