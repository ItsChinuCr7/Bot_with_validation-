# ============================================================
# evaluation/self_healer.py
# Self-Healing using Gemini via debuggerAddress (attaches to
# your already-open Brave — keeps all logins)
# ============================================================

import json, os, time, re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

MEMORY_FILE  = "self_healing_memory.json"
BRAVE_PATH   = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
DEBUG_PORT   = "127.0.0.1:9222"
WAIT_SECS    = 60
MIN_WAIT     = 20

VALIDATION_PROMPT = """You are an AI answer validator.

QUESTION: {question}

AI ANSWER: {answer}

Validate this answer and respond in EXACT format:

VERDICT: [CORRECT / PARTIALLY_CORRECT / WRONG]
ACCURACY_SCORE: [0-10]
WHAT_IS_WRONG: [specific errors or None]
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
            "total": total, "correct": correct,
            "wrong": wrong, "partial": partial,
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


# ── Connect to existing Brave (debuggerAddress) ───────────
def _attach_to_brave() -> webdriver.Chrome:
    """
    Attach to your already-open Brave browser.
    Brave must be started with --remote-debugging-port=9222
    """
    options = Options()
    options.binary_location = BRAVE_PATH
    options.add_experimental_option("debuggerAddress", DEBUG_PORT)
    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=options)
    print(f"[SelfHealer] Attached to Brave at {DEBUG_PORT}")
    return driver


# ── Send prompt to Gemini ─────────────────────────────────
def _send_to_gemini_and_wait(driver, prompt: str) -> str:
    """
    Navigate to Gemini, send prompt, wait for FULL response.
    Uses your existing logged-in Brave session.
    """
    # Open fresh Gemini chat
    driver.get("https://gemini.google.com/app")
    time.sleep(5)

    # Find input — try multiple selectors
    input_box = None
    selectors = [
        "div.ql-editor[contenteditable='true']",
        "rich-textarea div[contenteditable='true']",
        "div[contenteditable='true']",
        "textarea",
        "p[data-placeholder]",
    ]
    for sel in selectors:
        try:
            input_box = WebDriverWait(driver, 6).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
            )
            if input_box:
                print(f"[SelfHealer] Found input with: {sel}")
                break
        except Exception:
            continue

    if not input_box:
        # Last resort — find any contenteditable
        try:
            elements  = driver.find_elements(By.XPATH, "//*[@contenteditable='true']")
            if elements:
                input_box = elements[-1]
                print("[SelfHealer] Found input via XPath")
        except Exception:
            pass

    if not input_box:
        print("[SelfHealer] Could not find Gemini input box")
        return ""

    # Click and clear
    input_box.click()
    time.sleep(0.5)

    # Use pyperclip clipboard — entire prompt copied and pasted in ONE shot
    # This prevents Gemini receiving text as multiple messages
    try:
        import pyperclip
        pyperclip.copy(prompt)
        time.sleep(0.3)
        input_box.send_keys(Keys.CONTROL + "a")
        input_box.send_keys(Keys.DELETE)
        time.sleep(0.3)
        input_box.send_keys(Keys.CONTROL + "v")
        time.sleep(1.0)  # wait for paste to complete fully
        print(f"[SelfHealer] Pasted prompt via clipboard ({len(prompt)} chars)")
    except Exception as e:
        print(f"[SelfHealer] Clipboard failed: {e} — trying JS insert")
        try:
            driver.execute_script(
                "arguments[0].focus();"
                "document.execCommand('selectAll', false, null);"
                "document.execCommand('insertText', false, arguments[1]);",
                input_box, prompt
            )
            time.sleep(0.8)
        except Exception as e2:
            print(f"[SelfHealer] JS insert also failed: {e2}")
            return ""

    # Submit
    input_box.send_keys(Keys.RETURN)
    print(f"[SelfHealer] Prompt sent ({len(prompt)} chars). Waiting for response...")

    # Wait — minimum MIN_WAIT seconds always
    time.sleep(MIN_WAIT)

    # Then poll until text stops growing
    waited   = MIN_WAIT
    interval = 3
    response_selectors = [
        "div.response-container-content",
        "div.model-response-text",
        "div[data-response-id]",
        "p[data-path-to-node]",
        "div.markdown",
        "message-content",
    ]

    while waited < WAIT_SECS:
        time.sleep(interval)
        waited += interval

        for sel in response_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, sel)
                if not elements:
                    continue
                text = elements[-1].text.strip()
                if len(text) < 50:
                    continue

                # Check text stopped growing = Gemini finished
                prev_len = len(text)
                time.sleep(3)
                elements2 = driver.find_elements(By.CSS_SELECTOR, sel)
                if elements2:
                    new_text = elements2[-1].text.strip()
                    if len(new_text) == prev_len and prev_len > 50:
                        print(f"[SelfHealer] Gemini done ({prev_len} chars, {waited}s)")
                        return new_text
            except Exception:
                continue

        print(f"[SelfHealer] Still waiting... ({waited}s)")

    # Timeout — return whatever is there
    for sel in response_selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, sel)
            if elements:
                text = elements[-1].text.strip()
                if text:
                    print(f"[SelfHealer] Timeout — returning partial ({len(text)} chars)")
                    return text
        except Exception:
            continue

    return ""


# ── Parse Gemini Response ─────────────────────────────────
def _parse_response(response: str) -> dict:
    def extract(label, text):
        m = re.search(rf"{label}:\s*(.+?)(?=\n[A-Z_]+:|$)", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    verdict        = extract("VERDICT", response).upper()
    accuracy_str   = extract("ACCURACY_SCORE", response)
    what_is_wrong  = extract("WHAT_IS_WRONG", response)
    correct_answer = extract("CORRECT_ANSWER", response)
    key_corrections= extract("KEY_CORRECTIONS", response)

    try:
        accuracy = float(re.search(r'\d+\.?\d*', accuracy_str).group())
    except Exception:
        accuracy = 5.0

    if verdict not in ["CORRECT", "PARTIALLY_CORRECT", "WRONG"]:
        verdict = "CORRECT" if accuracy >= 8 else ("PARTIALLY_CORRECT" if accuracy >= 5 else "WRONG")

    return {
        "verdict":         verdict,
        "accuracy_score":  accuracy,
        "what_is_wrong":   what_is_wrong,
        "correct_answer":  correct_answer,
        "key_corrections": key_corrections,
    }


# ── Main Self-Heal ────────────────────────────────────────
def self_heal(
    question:        str,
    original_answer: str,
    memory:          HealingMemory,
    context:         str = "",
    llm=None,        # not used but kept for compatibility
) -> dict:
    """
    Validates answer using Gemini via your existing Brave session.
    Brave must be running with --remote-debugging-port=9222
    """
    driver = None
    try:
        driver = _attach_to_brave()
    except Exception as e:
        print(f"[SelfHealer] Could not attach to Brave: {e}")
        print("[SelfHealer] Make sure Brave is running with --remote-debugging-port=9222")
        return {
            "final_answer":   original_answer,
            "verdict":        "UNKNOWN",
            "accuracy_score": 0,
            "was_corrected":  False,
            "what_was_wrong": "",
            "gemini_said":    f"Could not connect to Brave. Start Brave with debug port 9222.",
        }

    try:
        prompt   = VALIDATION_PROMPT.format(
            question=question,
            answer=original_answer[:800],
        )
        response = _send_to_gemini_and_wait(driver, prompt)

        if not response:
            return {
                "final_answer":   original_answer,
                "verdict":        "UNKNOWN",
                "accuracy_score": 0,
                "was_corrected":  False,
                "what_was_wrong": "",
                "gemini_said":    "Gemini did not respond. Check Brave is open and logged in.",
            }

        validation     = _parse_response(response)
        verdict        = validation["verdict"]
        correct_answer = validation["correct_answer"].strip()
        what_is_wrong  = validation["what_is_wrong"].strip()
        key_corr       = validation["key_corrections"].strip()
        accuracy       = validation["accuracy_score"]

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
        print(f"[SelfHealer] Verdict: {verdict} ({accuracy}/10), corrected: {was_corrected}")

        parts = [f"Verdict: {verdict} ({accuracy}/10) — validated by Gemini"]
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
        # Don't quit driver — we attached to existing Brave, don't close it!
        pass