# ============================================================
# data/cache/cache_manager.py
# LLM response cache — same question = instant answer.
# Saves to disk so cache persists across restarts.
# ============================================================

import json
import hashlib
import os
import time

CACHE_FILE = os.path.join(os.path.dirname(__file__), "llm_cache.json")
CACHE_TTL  = 60 * 60 * 24 * 7   # 7 days in seconds


def _make_key(query: str, route: str) -> str:
    """Hash query + route into a cache key."""
    raw = f"{route.lower().strip()}::{query.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _load() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(cache: dict):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def get_cached(query: str, route: str) -> dict | None:
    """
    Return cached result if it exists and is not expired.
    Returns None if not found or expired.
    """
    cache = _load()
    key   = _make_key(query, route)
    entry = cache.get(key)
    if not entry:
        return None
    # Check TTL
    if time.time() - entry.get("timestamp", 0) > CACHE_TTL:
        return None
    return entry.get("result")


def set_cache(query: str, route: str, result: dict):
    """
    Save a result to cache.
    Only caches general and RAG routes — not research (web results change).
    """
    if route in ["research", "web", "calc"]:
        return  # don't cache dynamic content
    cache     = _load()
    key       = _make_key(query, route)
    cache[key] = {
        "timestamp": time.time(),
        "query":     query,
        "route":     route,
        "result":    result,
    }
    _save(cache)


def clear_cache():
    """Wipe entire cache."""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)


def get_cache_stats() -> dict:
    """Return cache size and entry count."""
    cache = _load()
    now   = time.time()
    valid = sum(1 for e in cache.values() if now - e.get("timestamp", 0) <= CACHE_TTL)
    size  = os.path.getsize(CACHE_FILE) / 1024 if os.path.exists(CACHE_FILE) else 0
    return {
        "total_entries": len(cache),
        "valid_entries": valid,
        "expired":       len(cache) - valid,
        "size_kb":       round(size, 1),
    }