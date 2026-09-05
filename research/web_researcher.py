# ============================================================
# research/web_researcher.py
#
# High-efficiency, fast HTTP-based web researcher.
# Replaces slow Selenium browser spawns with fast, multithreaded
# HTTP requests via requests + BeautifulSoup.
#
# Sources:
#   duckduckgo  -> fast HTML instant search + scrape
#   wikipedia   -> direct Wikipedia API / page fetch
#   bing        -> search results + article scrape
# ============================================================

import re
import html
import logging
from urllib.parse import quote_plus, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

MAX_CHARS_PER_PAGE = 3000
PAGE_TIMEOUT = 6
MAX_WORKERS = 4

ALL_SOURCES = ["duckduckgo", "wikipedia", "bing"]

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_SOURCE_PROMPT = """You are a research routing agent.
Pick the best 2-3 sources for this query from: duckduckgo, wikipedia, bing

Rules:
- Factual/encyclopedic -> wikipedia
- Current events / recent news -> duckduckgo, bing
- Technical / coding -> duckduckgo, bing
- Simple well-known fact -> just wikipedia
- Only use ALL if explicitly requested

Return ONLY a comma-separated list. No explanation.

Query: {query}
Sources:"""


# ── Query Cleaners and Parsers ─────────────────────────────

def extract_research_query(query: str) -> str:
    """Strip 'research' keyword and source overrides to get clean topic."""
    clean = re.sub(r"\bresearch\b", "", query, flags=re.IGNORECASE)
    clean = re.sub(r"\bon\s+(wikipedia|bing|duckduckgo|all)\b", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip(" ,:-")
    return clean if clean else query


def parse_user_source_override(query: str):
    """
    Detect if user specified a source override.
    e.g. "research on wikipedia quantum computing" -> ("quantum computing", "wikipedia")
    """
    m = re.search(r"research\s+on\s+(wikipedia|bing|duckduckgo|all)\s+(.+)", query, re.IGNORECASE)
    if m:
        return m.group(2).strip(), m.group(1).strip()

    m2 = re.search(r"research\s+on\s+([\w\s,and]+?)\s+(?:about|for|:|–|-)\s*(.+)", query, re.IGNORECASE)
    if m2:
        return m2.group(2).strip(), m2.group(1).strip()

    return query, None


def decide_sources(llm, query: str, user_override: str = None) -> list:
    """LLM picks best sources; user override takes priority."""
    if user_override:
        ov = user_override.lower()
        if "all" in ov:
            return ALL_SOURCES
        matched = [s for s in ALL_SOURCES if s in ov]
        if matched:
            return matched
    try:
        raw = llm.invoke(_SOURCE_PROMPT.format(query=query)).content.strip().lower()
        sources = [s.strip() for s in raw.split(",") if s.strip() in ALL_SOURCES]
        return sources if sources else ["duckduckgo", "wikipedia"]
    except Exception:
        return ["duckduckgo", "wikipedia"]


# ── Text & HTML Extraction ─────────────────────────────────

def _extract_text_from_html(html_content: str) -> str:
    """Extract clean readable text and tables from HTML."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript", "svg"]):
        tag.decompose()

    parts = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        lines = []
        for row in rows:
            cells = row.find_all(["td", "th"])
            line = " | ".join(c.get_text(strip=True) for c in cells)
            if line.strip():
                lines.append(line)
        if lines:
            parts.append("[TABLE]\n" + "\n".join(lines) + "\n[/TABLE]")
        table.decompose()

    main_el = soup.find("article") or soup.find("main") or soup.find("body") or soup
    text = main_el.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        parts.append(text)

    full = "\n\n".join(parts)
    return full[:MAX_CHARS_PER_PAGE]


def _fetch_url_content(url: str, timeout: int = PAGE_TIMEOUT) -> str:
    """Fetch URL body content via fast HTTP GET."""
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        if resp.status_code == 200:
            return _extract_text_from_html(resp.text)
    except Exception as e:
        logger.debug(f"Failed to fetch {url}: {e}")
    return ""


# ── Scrapers ───────────────────────────────────────────────

def _duckduckgo_search(query: str, max_results: int = 2) -> list:
    """Fast DuckDuckGo search using HTML endpoint."""
    results = []
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        resp = requests.post(url, data={"q": query}, headers=DEFAULT_HEADERS, timeout=PAGE_TIMEOUT)
        if resp.status_code != 200:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=PAGE_TIMEOUT)

        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.find_all("a", class_="result__url")
            urls = []
            for a in links:
                href = a.get("href", "")
                if "uddg=" in href:
                    raw_url = href.split("uddg=")[-1].split("&")[0]
                    target_url = unquote(raw_url)
                    if target_url.startswith("http") and "duckduckgo.com" not in target_url:
                        urls.append(target_url)
                elif href.startswith("http") and "duckduckgo.com" not in href:
                    urls.append(href)

            with ThreadPoolExecutor(max_workers=max_results) as executor:
                future_to_url = {executor.submit(_fetch_url_content, u): u for u in urls[:max_results]}
                for future in as_completed(future_to_url):
                    u = future_to_url[future]
                    content = future.result()
                    if content and len(content) > 120:
                        results.append({"source": "duckduckgo", "url": u, "content": content})
    except Exception as e:
        logger.warning(f"DuckDuckGo search error: {e}")
    return results


def _wikipedia_search(query: str, max_results: int = 2) -> list:
    """Direct Wikipedia API search + fetch."""
    results = []
    try:
        api_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote_plus(query)}"
            f"&utf8=&format=json&srlimit={max_results}"
        )
        resp = requests.get(api_url, headers=DEFAULT_HEADERS, timeout=PAGE_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            search_items = data.get("query", {}).get("search", [])
            for item in search_items:
                title = item.get("title")
                snippet = html.unescape(re.sub(r"<[^>]+>", "", item.get("snippet", "")))
                page_url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
                
                full_content = _fetch_url_content(page_url)
                content = full_content if len(full_content) > 200 else f"{title}\n{snippet}"
                results.append({"source": "wikipedia", "url": page_url, "content": content})
    except Exception as e:
        logger.warning(f"Wikipedia search error: {e}")
    return results


def _bing_search(query: str, max_results: int = 2) -> list:
    """Bing search scraper."""
    results = []
    try:
        url = f"https://www.bing.com/search?q={quote_plus(query)}"
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=PAGE_TIMEOUT)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.select("li.b_algo h2 a")
            urls = []
            for a in links:
                href = a.get("href", "")
                if href.startswith("http") and "bing.com" not in href and "microsoft.com" not in href:
                    urls.append(href)

            with ThreadPoolExecutor(max_workers=max_results) as executor:
                future_to_url = {executor.submit(_fetch_url_content, u): u for u in urls[:max_results]}
                for future in as_completed(future_to_url):
                    u = future_to_url[future]
                    content = future.result()
                    if content and len(content) > 120:
                        results.append({"source": "bing", "url": u, "content": content})
    except Exception as e:
        logger.warning(f"Bing search error: {e}")
    return results


# ── Main Entry Point ───────────────────────────────────────

def run_research(query: str, sources: list) -> dict:
    """
    Run high-efficiency parallel research across requested sources.

    Returns:
        {
            "results": list of {source, url, content},
            "context": str (combined text),
            "final_answer": str,
            "sources": list of URLs,
            "error": str or None
        }
    """
    scraper_map = {
        "duckduckgo": _duckduckgo_search,
        "wikipedia":  _wikipedia_search,
        "bing":       _bing_search,
    }

    all_results = []
    clean_query = extract_research_query(query)

    try:
        logger.info(f"Running high-efficiency research on sources: {sources}")
        with ThreadPoolExecutor(max_workers=min(len(sources), MAX_WORKERS)) as executor:
            future_to_source = {}
            for src in sources:
                if src in scraper_map:
                    future_to_source[executor.submit(scraper_map[src], clean_query)] = src

            for future in as_completed(future_to_source):
                try:
                    res = future.result()
                    if res:
                        all_results.extend(res)
                except Exception as e:
                    logger.warning(f"Scraper error: {e}")

        if not all_results:
            return {
                "results": [],
                "context": "",
                "final_answer": "",
                "sources": [],
                "error": "No content found. Try rephrasing or choosing different sources.",
            }

        parts = [f"[{r['source'].upper()} — {r['url']}]\n{r['content']}" for r in all_results]
        combined = "\n\n---\n\n".join(parts)

        return {
            "results": all_results,
            "context": combined,
            "final_answer": combined,
            "sources": [r["url"] for r in all_results],
            "error": None,
        }

    except Exception as e:
        logger.error(f"Research pipeline error: {e}")
        return {
            "results": [],
            "context": "",
            "final_answer": "",
            "sources": [],
            "error": f"Research failed: {e}",
        }