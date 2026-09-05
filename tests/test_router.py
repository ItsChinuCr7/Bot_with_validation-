import pytest
from agents.agent_router import route_query, extract_research_query
from research.web_researcher import parse_user_source_override

def test_extract_research_query():
    assert extract_research_query("research quantum computing") == "quantum computing"

def test_parse_user_source_override():
    q, src = parse_user_source_override("research on wikipedia python programming")
    assert q == "python programming"
    assert src == "wikipedia"

    q2, src2 = parse_user_source_override("what is the weather?")
    assert q2 == "what is the weather?"
    assert src2 is None

def test_route_query_calc():
    assert route_query("calculate 45 * 12") == "calc"
    assert route_query("what is the value of 2 + 2") == "calc"

def test_route_query_chat():
    assert route_query("hello there!") == "chat"
    assert route_query("who are you?") == "chat"

def test_route_query_research():
    assert route_query("research the latest quantum computing breakthroughs") == "research"

def test_route_query_forced_mode():
    assert route_query("any query here", forced_mode="rag") == "rag"
    assert route_query("any query here", forced_mode="general") == "general"

def test_route_query_no_docs_fallback():
    assert route_query("explain transformers", has_docs=False) == "general"
