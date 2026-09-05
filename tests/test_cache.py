import pytest
import os
import time
from data.cache.cache_manager import get_cached, set_cache, clear_cache, get_cache_stats

def test_cache_set_and_get():
    clear_cache()
    query = "test unique query 123"
    route = "general"
    data = {"answer": "This is a cached test answer", "route": route}
    
    # Save to cache
    set_cache(query, route, data)
    
    # Fetch from cache
    cached = get_cached(query, route)
    assert cached is not None
    assert cached["answer"] == "This is a cached test answer"
    
    # Non-cached query returns None
    assert get_cached("another query", route) is None

def test_dynamic_routes_not_cached():
    clear_cache()
    query = "research something"
    data = {"answer": "dynamic research", "route": "research"}
    set_cache(query, "research", data)
    assert get_cached(query, "research") is None

def test_cache_stats():
    stats = get_cache_stats()
    assert "total_entries" in stats
    assert "valid_entries" in stats
