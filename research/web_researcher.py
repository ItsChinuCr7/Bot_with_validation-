# ============================================================
# research/web_researcher.py
# Sources: DuckDuckGo + Gemini
# Final answer: Gemini synthesizes everything in one output
# ============================================================

import time
import re
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

# ── Paths ─────────────────────────────────────────────────
BRAVE_PATH    = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
BRAVE_PROFILE = r"C:\Users\chinm\AppData\Local\BraveSoftware\Brave-Browser"

# ── Config ────────────────────────────────────────────────
MAX_CHARS_PER_PAGE  = 3000   # per scraped page
GEMINI_WAIT_SECS    = 40     # max wait for Gemini response
GEMINI_MIN_WAIT     = 20     # minimum wait before checking
PAGE_LOAD_TIMEOUT   = 10
SCROLL_PAUSE        = 1.2

ALL_SOURCES = ["duckduckgo", "gemini"]

# ── Source selection prompt ───────────────────────────────
SOURCE_SELECT_PROMPT = """You are a research routing agent.
Available sources: duckduckgo, gemini

Rules:
- For factual/technical/scientific → both
- For current events/news → duckduckgo
- For deep explanations/opinions → gemini
- Default → both

Return ONLY comma-separated source names.
Query: {query}
Sources:"""


def decide_sources(llm, query: str, user_override: str = None) -> list:
    if user_override:
        override = user_override.lower()
        if "all" in override:
            return ALL_SOURCES
        matched = [s for s in ALL_SOURCES if s in override]
        if matched:
            return matched
    try:
        result  = llm.invoke(SOURCE_SELECT_PROMPT.format(query=query)).content.strip().lower()
        sources = [s.strip() for s in result.split(",") if s.strip() in ALL_SOURCES]
        return sources if sources else ALL_SOURCES
    except Exception:
        return ALL_SOURCES


def parse_user_source_override(query: str):
    pattern = r"research\s+on\s+(gemini|duckduckgo|all)\s+(.+)"
    match   = re.search(pattern, query, re.IGNORECASE)
    if match:
        return match.group(2).strip(), match.group(1).strip()
    return query, None


def extract_research_query(query: str) -> str:
    clean = re.sub(r"\bresearch\b", "", query, flags=re.IGNORECASE).strip()
    clean = re.sub(r"\s+", " ", clean).strip(" ,:-")
    return clean if clean else query


# ── Browser ───────────────────────────────────────────────
def _build_brave_driver(headless: bool = False) -> webdriver.Chrome:
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
    options.add_argument(f"--user-data-dir={BRAVE_PROFILE}")
    options.add_argument("--profile-directory=Default")
    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver


# ── Text Extractor ────────────────────────────────────────
def _extract_text_and_tables(html: str, max_chars: int = MAX_CHARS_PER_PAGE) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer",
                     "header", "aside", "form", "iframe", "noscript"]):
        tag.decompose()
    output_parts = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        lines = []
        for row in rows:
            cells = row.find_all(["td", "th"])
            line  = " | ".join(c.get_text(strip=True) for c in cells)
            if line.strip():
                lines.append(line)
        if lines:
            output_parts.append("[TABLE]\n" + "\n".join(lines) + "\n[/TABLE]")
        table.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        output_parts.append(text)
    return "\n\n".join(output_parts)[:max_chars]


def _safe_get(driver, url: str) -> str:
    try:
        driver.get(url)
        time.sleep(SCROLL_PAUSE)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
        time.sleep(0.5)
        return driver.page_source
    except Exception:
        return ""


# ── Gemini Input Helper ───────────────────────────────────
def _find_gemini_input(driver) -> object:
    """Find Gemini's input box using multiple selectors."""
    selectors = [
        "rich-textarea div[contenteditable='true']",
        "div[contenteditable='true'][data-placeholder]",
        "div.ql-editor",
        "textarea",
    ]
    for sel in selectors:
        try:
            box = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
            )
            if box:
                return box
        except Exception:
            continue
    return None


def _wait_for_gemini_response(driver, min_wait: int = GEMINI_MIN_WAIT,
                               max_wait: int = GEMINI_WAIT_SECS) -> str:
    """
    Wait for Gemini to finish its full response.
    - Always waits at least min_wait seconds
    - Then checks if text has stopped growing (response complete)
    - Waits up to max_wait seconds total
    """
    print(f"[Gemini] Waiting minimum {min_wait}s for full response...")
    time.sleep(min_wait)  # always wait minimum first

    waited   = min_wait
    interval = 3
    response_text = ""

    while waited < max_wait:
        time.sleep(interval)
        waited += interval

        for sel in [
            "div.response-container-content",
            "div.model-response-text",
            "p[data-path-to-node]",
            "div[class*='response-content']",
        ]:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, sel)
                if elements:
                    text = elements[-1].text.strip()
                    if len(text) > 100:
                        # Check if text stopped growing (Gemini finished)
                        prev_len = len(text)
                        time.sleep(3)
                        elements2 = driver.find_elements(By.CSS_SELECTOR, sel)
                        if elements2:
                            new_len = len(elements2[-1].text.strip())
                            if new_len == prev_len:
                                response_text = elements2[-1].text.strip()
                                print(f"[Gemini] Response complete ({new_len} chars, {waited}s)")
                                return response_text
            except Exception:
                continue

        print(f"[Gemini] Still generating... ({waited}s elapsed)")

    # Return whatever we have even if not fully done
    for sel in ["div.response-container-content", "div.model-response-text", "p[data-path-to-node]"]:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, sel)
            if elements:
                text = elements[-1].text.strip()
                if text:
                    print(f"[Gemini] Timeout — returning partial response ({len(text)} chars)")
                    return text
        except Exception:
            continue

    return ""


def _send_to_gemini(driver, prompt: str, navigate: bool = False) -> str:
    """
    Send a prompt to Gemini and wait for full response.
    Always opens a fresh new Gemini chat to avoid stale page state.
    """
    # Always navigate to a fresh Gemini chat — avoids "input box not found"
    # after a previous conversation changed the page state
    print("[Gemini] Opening fresh Gemini chat...")
    driver.get("https://gemini.google.com/app")
    time.sleep(5)  # give page time to fully load

    # Retry finding input box up to 3 times
    input_box = None
    for attempt in range(3):
        input_box = _find_gemini_input(driver)
        if input_box:
            break
        print(f"[Gemini] Input box not found, retrying ({attempt+1}/3)...")
        time.sleep(3)
        driver.refresh()
        time.sleep(4)

    if not input_box:
        print("[Gemini] Could not find input box after 3 attempts")
        return ""

    input_box.click()
    time.sleep(0.5)

    # Clipboard paste — entire prompt in ONE shot, never chunked
    try:
        import pyperclip
        pyperclip.copy(prompt)
        time.sleep(0.3)
        input_box.send_keys(Keys.CONTROL + "a")
        input_box.send_keys(Keys.DELETE)
        time.sleep(0.3)
        input_box.send_keys(Keys.CONTROL + "v")
        time.sleep(1.0)
        print(f"[Gemini] Pasted prompt via clipboard ({len(prompt)} chars)")
    except Exception as e:
        print(f"[Gemini] Clipboard failed: {e} — trying JS")
        try:
            driver.execute_script(
                "arguments[0].focus();"
                "document.execCommand('selectAll', false, null);"
                "document.execCommand('insertText', false, arguments[1]);",
                input_box, prompt
            )
            time.sleep(0.8)
        except Exception as e2:
            print(f"[Gemini] JS also failed: {e2}")
            return ""

    time.sleep(0.5)
    input_box.send_keys(Keys.RETURN)

    return _wait_for_gemini_response(driver)


# ── DuckDuckGo Scraper ────────────────────────────────────
def _scrape_duckduckgo(driver, query: str) -> list:
    results = []
    urls    = []
    try:
        driver.get(f"https://duckduckgo.com/?q={quote_plus(query)}&ia=web")
        time.sleep(2.5)

        links = driver.find_elements(By.CSS_SELECTOR, "a[data-testid='result-title-a']")
        for link in links[:4]:
            href = link.get_attribute("href")
            if href and href.startswith("http") and "duckduckgo.com" not in href:
                urls.append(href)

        if not urls:
            links = driver.find_elements(By.CSS_SELECTOR, "article h2 a")
            for link in links[:4]:
                href = link.get_attribute("href")
                if href and href.startswith("http") and "duckduckgo.com" not in href:
                    urls.append(href)

        for url in urls[:3]:
            html = _safe_get(driver, url)
            if html:
                text = _extract_text_and_tables(html)
                if len(text) > 150:
                    results.append({
                        "source":  "duckduckgo",
                        "url":     url,
                        "content": text,
                    })
    except Exception as e:
        print(f"[DDG error] {e}")
    return results


# ── Gemini Research Scraper ───────────────────────────────
def _scrape_gemini(driver, query: str) -> list:
    """
    Ask Gemini the research query directly.
    Wait for FULL response before returning.
    """
    results = []
    try:
        response = _send_to_gemini(driver, query, navigate=True)
        if response:
            results.append({
                "source":  "gemini",
                "url":     "https://gemini.google.com/app",
                "content": response,
            })
    except Exception as e:
        print(f"[Gemini research error] {e}")
    return results


# ── Gemini Final Synthesizer ──────────────────────────────
def _gemini_final_answer(driver, query: str, all_data: str) -> str:
    """
    Send ALL extracted data back to Gemini for final synthesis.
    """
    data_snippet = all_data[:5000]

    synthesis_prompt = (
        f"TASK: Synthesize the research data below into one final answer.\n"
        f"The data is already provided below — do not ask for more.\n\n"
        f"QUESTION: {query}\n\n"
        f"--- RESEARCH DATA START ---\n"
        f"{data_snippet}\n"
        f"--- RESEARCH DATA END ---\n\n"
        f"Write a comprehensive answer (max 4000 chars) based ONLY on the data above. "
        f"Include key facts, cite sources, use clear sections."
    )

    print(f"[Gemini] all_data length received: {len(all_data)} chars")
    print(f"[Gemini] data_snippet length: {len(data_snippet)} chars")
    print(f"[Gemini] Synthesis prompt length: {len(synthesis_prompt)} chars")
    print(f"[Gemini] First 200 chars of data going to Gemini: {data_snippet[:200]}")
    print("[Gemini] Sending all data for final synthesis...")
    response = _send_to_gemini(driver, synthesis_prompt, navigate=False)
    return response


# ── Main Research Function ────────────────────────────────
def run_research(query: str, sources: list) -> dict:
    """
    Full research pipeline:
    1. Scrape DuckDuckGo and/or Gemini
    2. Send ALL data back to Gemini for final synthesis
    3. Return structured results + final answer
    """
    driver      = None
    all_results = []

    try:
        print(f"[Research] Starting Brave browser...")
        driver = _build_brave_driver(headless=False)

        # ── Step 1: Scrape all sources ────────────────────
        for source in sources:
            print(f"[Research] Scraping: {source}")
            if source == "duckduckgo":
                results = _scrape_duckduckgo(driver, query)
            elif source == "gemini":
                results = _scrape_gemini(driver, query)
            else:
                continue
            all_results.extend(results)
            print(f"[Research] {source}: got {len(results)} result(s)")

        if not all_results:
            return {
                "results":      [],
                "context":      "",
                "final_answer": "",
                "sources":      [],
                "error":        "No content found. Try rephrasing.",
            }

        # ── Step 2: Build combined context ───────────────
        parts = []
        for i, r in enumerate(all_results):
            parts.append(
                f"[SOURCE {i+1} — {r['source'].upper()}]\n"
                f"URL: {r['url']}\n"
                f"CONTENT:\n{r['content']}"
            )
        combined_data = "\n\n" + ("=" * 60) + "\n\n".join(parts)

        # ── Step 3: Send to Gemini for final synthesis ────
        print("[Research] Sending all data to Gemini for synthesis...")
        final_answer = _gemini_final_answer(driver, query, combined_data)

        if not final_answer:
            # Fallback: just return raw combined data
            final_answer = combined_data[:4000]

        return {
            "results":      all_results,
            "context":      combined_data,
            "final_answer": final_answer,
            "sources":      [r["url"] for r in all_results],
            "error":        None,
        }

    except Exception as e:
        return {
            "results":      [],
            "context":      "",
            "final_answer": "",
            "sources":      [],
            "error":        str(e),
        }

    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass