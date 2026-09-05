import pytest
from retrieval.hybrid_search import HybridSearcher

class MockDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

class MockVectorDB:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search_with_score(self, query, k=5):
        # Return mock distance scores
        results = []
        for i, doc in enumerate(self.docs[:k]):
            score = 0.1 * (i + 1)
            results.append((doc, score))
        return results

def test_hybrid_searcher_scoring():
    docs = [
        MockDoc("Python is an interpreted programming language."),
        MockDoc("Machine learning uses neural networks and datasets."),
        MockDoc("FastAPI and Streamlit build interactive web apps in Python."),
    ]
    vector_db = MockVectorDB(docs)
    searcher = HybridSearcher(docs, vector_db, alpha=0.5)

    results = searcher.search("Python programming", top_k=2)
    assert len(results) <= 2
    assert any("Python" in d.page_content for d in results)
