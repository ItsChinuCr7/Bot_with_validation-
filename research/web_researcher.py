# ============================================================
# research/web_researcher.py
#
# Selenium-based web researcher using Brave browser
#
# Sources:
#   duckduckgo  → search + scrape top result pages
#   wikipedia   → direct article scrape
#   bing        → additional results
#   gemini      → opens Gemini in YOUR Brave (already logged in)
#
# Smart routing: LLM auto-picks sources per query
# User override: "research on gemini ..." / "research on all ..."
# ============================================================

import re
import time
from urllib.parse import quote_plus

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ── Config ────────────────────────────────────────────────
BRAVE_PATH        = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
BRAVE_PROFILE_DIR = r"C:\Users\chinm\AppData\Local\BraveSoftware\Brave-Browser\User Data"

MAX_CHARS_PER_PAGE = 3000
GEMINI_WAIT_SECS   = 20
PAGE_LOAD_TIMEOUT  = 10
SCROLL_PAUSE       = 1.2

ALL_SOURCES = ["duckduckgo", "wikipedia", "bing", "gemini"]

_SOURCE_PROMPT = """You are a research routing agent.
Pick the best 2-3 sources for this query from: duckduckgo, wikipedia, bing, gemini

Rules:
- Factual/encyclopedic -> wikipedia
- Current events / recent news -> duckduckgo, bing
- Deep explanations / opinions / summaries -> gemini
- Technical / coding -> duckduckgo, gemini
- Simple well-known fact -> just wikipedia
- Only use ALL if explicitly requested

Return ONLY a comma-separated list. No explanation.

Query: {query}
Sources:"""


# ── Public helpers (imported by app.py) ──────────────────

def extract_research_query(query: str) -> str:
    """Strip 'research' keyword and source overrides to get clean topic."""
    clean = re.sub(r"\bresearch\b", "", query, flags=re.IGNORECASE)
    clean = re.sub(r"\bon\s+(gemini|wikipedia|bing|duckduckgo|all)\b", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip(" ,:-")
    return clean if clean else query


def parse_user_source_override(query: str):
    """
    Detect if user specified a source override.
    "research on gemini quantum computing" -> ("quantum computing", "gemini")
    Returns (clean_query, override | None)
    """
    m = re.search(r"research\s+on\s+(gemini|wikipedia|bing|duckduckgo|all)\s+(.+)", query, re.IGNORECASE)
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
        raw     = llm.invoke(_SOURCE_PROMPT.format(query=query)).content.strip().lower()
        sources = [s.strip() for s in raw.split(",") if s.strip() in ALL_SOURCES]
        return sources if sources else ["duckduckgo", "wikipedia"]
    except Exception:
        return ["duckduckgo", "wikipedia"]


# ── Browser setup ─────────────────────────────────────────

def _build_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    options.binary_location = BRAVE_PATH

    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,900")
    options.add_argument("--log-level=3")
    options.add_argument("--mute-audio")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    # Use a fresh temp session every time - avoids "profile locked" if Brave is open
    import tempfile, os as _os
    _tmp = _os.path.join(tempfile.gettempdir(), "brave_research_tmp")
    _os.makedirs(_tmp, exist_ok=True)
    options.add_argument(f"--user-data-dir={_tmp}")
    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver


# ── Text extraction ───────────────────────────────────────

def _extract(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]):
        tag.decompose()
    parts = []
    for table in soup.find_all("table"):
        rows  = table.find_all("tr")
        lines = []
        for row in rows:
            cells = row.find_all(["td", "th"])
            line  = " | ".join(c.get_text(strip=True) for c in cells)
            if line.strip():
                lines.append(line)
        if lines:
            parts.append("[TABLE]\n" + "\n".join(lines) + "\n[/TABLE]")
        table.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        parts.append(text)
    return "\n\n".join(parts)[:MAX_CHARS_PER_PAGE]


def _get(driver, url: str) -> str:
    try:
        driver.get(url)
        time.sleep(SCROLL_PAUSE)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
        time.sleep(0.4)
        return driver.page_source
    except Exception:
        return ""


# ── Source scrapers ───────────────────────────────────────

def _duckduckgo(driver, query: str) -> list:
    results = []
    try:
        driver.get(f"https://duckduckgo.com/?q={quote_plus(query)}&ia=web")
        time.sleep(2.5)
        links = driver.find_elements(By.CSS_SELECTOR, "a[data-testid='result-title-a']")
        if not links:
            links = driver.find_elements(By.CSS_SELECTOR, "article h2 a")
        urls = []
        for a in links[:4]:
            href = a.get_attribute("href")
            if href and href.startswith("http") and "duckduckgo.com" not in href:
                urls.append(href)
        for url in urls[:2]:
            html = _get(driver, url)
            if html:
                text = _extract(html)
                if len(text) > 150:
                    results.append({"source": "duckduckgo", "url": url, "content": text})
    except Exception as e:
        print(f"[DDG] {e}")
    return results


def _wikipedia(driver, query: str) -> list:
    results = []
    try:
        driver.get(f"https://en.wikipedia.org/wiki/Special:Search?search={quote_plus(query)}")
        time.sleep(2)
        urls = []
        if "/wiki/" in driver.current_url and "Special:Search" not in driver.current_url:
            urls.append(driver.current_url)
        else:
            for a in driver.find_elements(By.CSS_SELECTOR, ".mw-search-result-heading a")[:2]:
                href = a.get_attribute("href")
                if href:
                    urls.append(href)
        for url in urls[:2]:
            html = _get(driver, url)
            if html:
                text = _extract(html)
                if len(text) > 150:
                    results.append({"source": "wikipedia", "url": url, "content": text})
    except Exception as e:
        print(f"[Wiki] {e}")
    return results


def _bing(driver, query: str) -> list:
    results = []
    try:
        driver.get(f"https://www.bing.com/search?q={quote_plus(query)}")
        time.sleep(2)
        urls = []
        for a in driver.find_elements(By.CSS_SELECTOR, "li.b_algo h2 a")[:3]:
            href = a.get_attribute("href")
            if href and href.startswith("http") and "bing.com" not in href:
                urls.append(href)
        for url in urls[:2]:
            html = _get(driver, url)
            if html:
                text = _extract(html)
                if len(text) > 150:
                    results.append({"source": "bing", "url": url, "content": text})
    except Exception as e:
        print(f"[Bing] {e}")
    return results


def _gemini(driver, query: str) -> list:
    results = []
    try:
        driver.get("https://gemini.google.com/app")
        time.sleep(5)
        input_box = None
        for sel in [
            "rich-textarea div[contenteditable='true']",
            "div[contenteditable='true']",
            "div.ql-editor",
            "textarea",
        ]:
            try:
                input_box = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
                )
                if input_box:
                    break
            except Exception:
                continue
        if not input_box:
            print("[Gemini] Input box not found")
            return results

        input_box.click()
        time.sleep(0.5)
        input_box.send_keys(query)
        time.sleep(0.5)
        input_box.send_keys(Keys.RETURN)
        print(f"[Gemini] Waiting {GEMINI_WAIT_SECS}s...")
        time.sleep(GEMINI_WAIT_SECS)

        response_text = ""
        for sel in [
            "div.response-container-content",
            "div.model-response-text",
            "message-content div.markdown",
            "div[class*='response-content']",
            "div.presented-response-container",
        ]:
            try:
                els = driver.find_elements(By.CSS_SELECTOR, sel)
                if els:
                    text = els[-1].text.strip()
                    if len(text) > 50:
                        response_text = text
                        break
            except Exception:
                continue

        if not response_text:
            try:
                paras = driver.find_elements(By.CSS_SELECTOR, "p[data-path-to-node]")
                response_text = " ".join(p.text for p in paras if p.text.strip())
            except Exception:
                pass

        if response_text:
            print(f"[Gemini] Extracted {len(response_text)} chars")
            results.append({
                "source":  "gemini",
                "url":     "https://gemini.google.com/app",
                "content": response_text[:MAX_CHARS_PER_PAGE],
            })
        else:
            print("[Gemini] No response found")
    except Exception as e:
        print(f"[Gemini] {e}")
    return results


# ── Main entry point ──────────────────────────────────────

def run_research(query: str, sources: list) -> dict:
    """
    Run research across all requested sources.

    Returns dict with keys:
        results      → list of {source, url, content}
        context      → combined text for LLM
        final_answer → same as context (used by app.py)
        sources      → list of URLs scraped
        error        → str or None
    """
    driver      = None
    all_results = []
    headless    = "gemini" not in sources  # Gemini needs visible browser for login

    try:
        print(f"[Research] Launching Brave (headless={headless})...")
        driver = _build_driver(headless=headless)

        scraper_map = {
            "duckduckgo": _duckduckgo,
            "wikipedia":  _wikipedia,
            "bing":       _bing,
            "gemini":     _gemini,
        }

        for source in sources:
            if source in scraper_map:
                print(f"[Research] → {source.upper()}")
                all_results.extend(scraper_map[source](driver, query))

        if not all_results:
            return {
                "results": [], "context": "", "final_answer": "",
                "sources": [],
                "error": "No content found. Try rephrasing or choosing different sources.",
            }

        parts    = [f"[{r['source'].upper()} — {r['url']}]\n{r['content']}" for r in all_results]
        combined = "\n\n---\n\n".join(parts)

        return {
            "results":      all_results,
            "context":      combined,
            "final_answer": combined,
            "sources":      [r["url"] for r in all_results],
            "error":        None,
        }

    except Exception as e:
        return {
            "results": [], "context": "", "final_answer": "",
            "sources": [], "error": f"Research failed: {e}",
        }

    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass