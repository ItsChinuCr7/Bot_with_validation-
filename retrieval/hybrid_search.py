from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearcher:
    def __init__(self, docs, vector_db, alpha=0.5):
        self.docs = docs
        self.vector_db = vector_db
        self.alpha = alpha
        tokenized = [doc.page_content.lower().split() for doc in docs]
        self.bm25 = BM25Okapi(tokenized)
        # Fast lookup mapping: content -> index
        self._content_to_index = {doc.page_content: i for i, doc in enumerate(docs)}

    def search(self, query, top_k=8):
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_max = np.max(bm25_scores) if len(bm25_scores) > 0 and np.max(bm25_scores) > 0 else 1.0
        bm25_norm = bm25_scores / bm25_max

        vector_results = self.vector_db.similarity_search_with_score(query, k=min(top_k * 2, len(self.docs)))
        vector_score_map = {}
        for doc, score in vector_results:
            idx = self._content_to_index.get(doc.page_content)
            if idx is not None:
                # Convert distance score to similarity (1 / (1 + distance))
                vector_score_map[idx] = 1.0 / (1.0 + float(score))

        combined = []
        for i, doc in enumerate(self.docs):
            v_score = vector_score_map.get(i, 0.0)
            b_score = float(bm25_norm[i])
            hybrid_score = self.alpha * v_score + (1.0 - self.alpha) * b_score
            combined.append((hybrid_score, doc))

        combined.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in combined[:top_k]]
