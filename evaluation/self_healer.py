# ============================================================
# evaluation/self_healer.py
#
# Validates AI answers using Gemini via your open Brave browser.
# Brave must be running with --remote-debugging-port=9222
# ============================================================

import json
import os
import re
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

MEMORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "self_healing_memory.json")
WAIT_SECS   = 60
MIN_WAIT    = 20

# Portable Brave browser path fallback
BRAVE_PATH = os.getenv(
    "BRAVE_PATH",
    r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
    if os.name == "nt" else "/usr/bin/brave-browser"
)
DEBUG_PORT  = os.getenv("BRAVE_DEBUG_PORT", "127.0.0.1:9222")

VALIDATION_PROMPT = """You are an AI answer validator.

QUESTION: {question}

AI ANSWER: {answer}

Validate this answer. Respond in EXACT format below (no extra text):

VERDICT: [CORRECT / PARTIALLY_CORRECT / WRONG]
ACCURACY_SCORE: [0-10]
WHAT_IS_WRONG: [specific errors, or None]
CORRECT_ANSWER: [the accurate complete answer]
KEY_CORRECTIONS: [what to remember to avoid this mistake next time]"""


# ── Memory ────────────────────────────────────────────────

class HealingMemory:
    def __init__(self, memory_file: str = MEMORY_FILE):
        self.memory_file = memory_file
        self.memories    = self._load()

    def _load(self) -> list:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def save(self, entry: dict) -> None:
        self.memories.append(entry)
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[HealingMemory] Save error: {e}")

    def find_similar_mistake(self, query: str) -> dict | None:
        query_words = set(query.lower().split())
        best_match, best_score = None, 0
        for mem in self.memories:
            if mem.get("verdict") == "CORRECT":
                continue
            past_words = set(mem.get("question", "").lower().split())
            if not past_words:
                continue
            overlap = len(query_words & past_words)
            union   = len(query_words | past_words)
            score   = overlap / union if union > 0 else 0
            if score > 0.5 and score > best_score:
                best_score, best_match = score, mem
        return best_match

    def get_stats(self) -> dict:
        total   = len(self.memories)
        correct = sum(1 for m in self.memories if m.get("verdict") == "CORRECT")
        wrong   = sum(1 for m in self.memories if m.get("verdict") == "WRONG")
        partial = sum(1 for m in self.memories if m.get("verdict") == "PARTIALLY_CORRECT")
        return {
            "total":    total,
            "correct":  correct,
            "wrong":    wrong,
            "partial":  partial,
            "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
        }

    def get_lessons(self, limit: int = 5) -> str:
        wrong = [
            m for m in self.memories[-20:]
            if m.get("verdict") in ["WRONG", "PARTIALLY_CORRECT"]
            and m.get("key_corrections")
        ]
        if not wrong:
            return ""
        lines = ["PAST MISTAKES TO AVOID:"]
        for m in wrong[-limit:]:
            lines.append(f"- Q: {m['question'][:80]}")
            lines.append(f"  Lesson: {m['key_corrections'][:150]}")
        return "\n".join(lines)


# ── Attach to Brave ───────────────────────────────────────

def _attach_to_brave() -> webdriver.Chrome:
    """
    Attach Selenium to your already-running Brave browser.
    Start Brave first with:
        brave.exe --remote-debugging-port=9222
    Or double-click start_brave.bat
    """
    options = Options()
    options.binary_location = BRAVE_PATH
    options.add_experimental_option("debuggerAddress", DEBUG_PORT)
    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=options)
    print(f"[SelfHealer] Attached to Brave at {DEBUG_PORT} ✅")
    return driver


# ── Send to Gemini and wait ───────────────────────────────

def _send_to_gemini(driver, prompt: str) -> str:
    """Navigate to Gemini, paste prompt, wait for full response."""
    driver.get("https://gemini.google.com/app")
    time.sleep(6)

    # Find input
    input_box = None
    for sel in [
        "rich-textarea div[contenteditable='true']",
        "div[contenteditable='true']",
        "div.ql-editor",
        "textarea",
    ]:
        try:
            input_box = WebDriverWait(driver, 6).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
            )
            if input_box:
                print(f"[SelfHealer] Input found: {sel}")
                break
        except Exception:
            continue

    if not input_box:
        try:
            els = driver.find_elements(By.XPATH, "//*[@contenteditable='true']")
            if els:
                input_box = els[-1]
        except Exception:
            pass

    if not input_box:
        print("[SelfHealer] Could not find Gemini input box")
        return ""

    # Paste prompt via clipboard
    input_box.click()
    time.sleep(0.4)
    try:
        import pyperclip
        pyperclip.copy(prompt)
        time.sleep(0.3)
        input_box.send_keys(Keys.CONTROL + "a")
        input_box.send_keys(Keys.DELETE)
        time.sleep(0.2)
        input_box.send_keys(Keys.CONTROL + "v")
        time.sleep(0.8)
        print(f"[SelfHealer] Pasted {len(prompt)} chars via clipboard")
    except Exception:
        # Fallback: type directly
        input_box.send_keys(prompt[:500])
        time.sleep(0.5)

    input_box.send_keys(Keys.RETURN)
    print(f"[SelfHealer] Submitted. Waiting {MIN_WAIT}s minimum...")
    time.sleep(MIN_WAIT)

    # Poll until response stops growing
    waited   = MIN_WAIT
    interval = 3
    response_selectors = [
        "div.response-container-content",
        "div.model-response-text",
        "p[data-path-to-node]",
        "div.markdown",
        "message-content",
    ]

    while waited < WAIT_SECS:
        time.sleep(interval)
        waited += interval
        for sel in response_selectors:
            try:
                els = driver.find_elements(By.CSS_SELECTOR, sel)
                if not els:
                    continue
                if "data-path-to-node" in sel:
                    text = "\n".join(e.text.strip() for e in els if e.text.strip())
                else:
                    text = els[-1].text.strip()
                if len(text) < 50:
                    continue
                prev = len(text)
                time.sleep(3)
                els2 = driver.find_elements(By.CSS_SELECTOR, sel)
                if els2:
                    if "data-path-to-node" in sel:
                        new_text = "\n".join(e.text.strip() for e in els2 if e.text.strip())
                    else:
                        new_text = els2[-1].text.strip()
                    if len(new_text) == prev:
                        print(f"[SelfHealer] Done: {prev} chars ({waited}s)")
                        return new_text
            except Exception:
                continue
        print(f"[SelfHealer] Waiting... ({waited}s)")

    # Timeout — return whatever exists
    for sel in response_selectors:
        try:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            if els:
                text = els[-1].text.strip()
                if text:
                    return text
        except Exception:
            continue
    return ""


# ── Parse Gemini's validation response ───────────────────

def _parse_response(response: str) -> dict:
    def extract(label):
        m = re.search(rf"{label}:\s*(.+?)(?=\n[A-Z_]+:|$)", response, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    verdict        = extract("VERDICT").upper()
    accuracy_str   = extract("ACCURACY_SCORE")
    what_is_wrong  = extract("WHAT_IS_WRONG")
    correct_answer = extract("CORRECT_ANSWER")
    key_corrections= extract("KEY_CORRECTIONS")

    try:
        accuracy = float(re.search(r"\d+\.?\d*", accuracy_str).group())
    except Exception:
        accuracy = 5.0

    if verdict not in ["CORRECT", "PARTIALLY_CORRECT", "WRONG"]:
        verdict = "CORRECT" if accuracy >= 8 else ("PARTIALLY_CORRECT" if accuracy >= 5 else "WRONG")

    return {
        "verdict":          verdict,
        "accuracy_score":   accuracy,
        "what_is_wrong":    what_is_wrong,
        "correct_answer":   correct_answer,
        "key_corrections":  key_corrections,
    }


# ── Main ──────────────────────────────────────────────────

def self_heal(
    question:        str,
    original_answer: str,
    memory:          HealingMemory,
    context:         str = "",
    llm=None,
) -> dict:
    """
    Validate answer via Gemini running in your Brave.
    Brave must be open with --remote-debugging-port=9222
    """
    driver = None
    try:
        driver = _attach_to_brave()
    except Exception as e:
        msg = (
            f"Could not attach to Brave: {e}\n"
            "Fix: start Brave with debug port using start_brave.bat"
        )
        print(f"[SelfHealer] {msg}")
        return {
            "final_answer":   original_answer,
            "verdict":        "UNKNOWN",
            "accuracy_score": 0,
            "was_corrected":  False,
            "what_was_wrong": "",
            "gemini_said":    msg,
        }

    try:
        prompt   = VALIDATION_PROMPT.format(
            question=question,
            answer=original_answer[:800],
        )
        response = _send_to_gemini(driver, prompt)

        if not response:
            return {
                "final_answer":   original_answer,
                "verdict":        "UNKNOWN",
                "accuracy_score": 0,
                "was_corrected":  False,
                "what_was_wrong": "",
                "gemini_said":    "Gemini gave no response. Is Brave logged in to Gemini?",
            }

        parsed         = _parse_response(response)
        verdict        = parsed["verdict"]
        correct_answer = parsed["correct_answer"]
        what_is_wrong  = parsed["what_is_wrong"]
        key_corr       = parsed["key_corrections"]
        accuracy       = parsed["accuracy_score"]

        was_corrected = False
        if verdict in ["WRONG", "PARTIALLY_CORRECT"] and correct_answer and len(correct_answer) > 20:
            final_answer  = correct_answer
            was_corrected = True
        else:
            final_answer = original_answer

        memory.save({
            "timestamp":       datetime.now().isoformat(),
            "question":        question,
            "original_answer": original_answer[:300],
            "verdict":         verdict,
            "accuracy_score":  accuracy,
            "what_is_wrong":   what_is_wrong[:300],
            "correct_answer":  correct_answer[:300],
            "key_corrections": key_corr[:300],
            "was_corrected":   was_corrected,
            "validated_by":    "gemini",
        })
        print(f"[SelfHealer] Verdict: {verdict} ({accuracy}/10) | Corrected: {was_corrected}")

        parts = [f"Verdict: {verdict} ({accuracy}/10) — Gemini validated"]
        if what_is_wrong and what_is_wrong.lower() not in ["none", ""]:
            parts.append(f"Issues: {what_is_wrong[:200]}")
        if was_corrected:
            parts.append("✅ Answer corrected by Gemini")
        if key_corr:
            parts.append(f"Lesson saved: {key_corr[:150]}")

        return {
            "final_answer":   final_answer,
            "verdict":        verdict,
            "accuracy_score": accuracy,
            "was_corrected":  was_corrected,
            "what_was_wrong": what_is_wrong,
            "gemini_said":    "\n".join(parts),
        }

    finally:
        pass  # Never quit — Brave belongs to the user