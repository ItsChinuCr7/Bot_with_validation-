# ============================================================
# evaluation/self_healer.py
#
# Self-Healing RAG System
#
# Flow:
# 1. My system generates an answer
# 2. Send (Question + Answer) to Gemini via Selenium
# 3. Gemini validates and corrects
# 4. Parse Gemini's feedback (correct/wrong/corrections)
# 5. Save mistakes permanently to a JSON memory file
# 6. Return corrected answer to user
#
# Permanent learning:
# - Mistakes saved to self_healing_memory.json
# - Loaded on every startup
# - Checked BEFORE answering (prevents repeat mistakes)
# ============================================================

import json
import os
import time
import re
from datetime import datetime
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

MEMORY_FILE = "self_healing_memory.json"

# How confident the answer must be before skipping validation
# 0 = always validate, 1 = never validate
VALIDATION_THRESHOLD = 0.6

# ── Gemini Validation Prompt ──────────────────────────────
VALIDATION_PROMPT_TEMPLATE = """You are an AI answer validator and teacher.

I have a RAG (document AI) system that gave an answer. Please validate it.

QUESTION ASKED:
{question}

ANSWER MY SYSTEM GAVE:
{answer}

CONTEXT MY SYSTEM USED:
{context}

Please respond in this EXACT format:

VERDICT: [CORRECT / PARTIALLY_CORRECT / WRONG]

ACCURACY_SCORE: [0-10]

WHAT_IS_WRONG:
[List specific errors, hallucinations, missing info, or "None" if correct]

CORRECT_ANSWER:
[The accurate, complete answer to the question]

KEY_CORRECTIONS:
[Bullet points of what the system should learn/remember to avoid this mistake]

REASON_FOR_ERROR:
[Why did the system make this mistake? e.g. wrong chunk retrieved, hallucination, misunderstanding]"""


# ── Memory Manager ────────────────────────────────────────
class HealingMemory:
    """
    Permanent storage of past mistakes and corrections.
    Saved as JSON file — persists across sessions.
    """

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
        """
        Check if a similar question was answered wrongly before.
        Returns the correction if found.
        """
        query_words = set(query.lower().split())
        best_match  = None
        best_score  = 0

        for mem in self.memories:
            if mem.get("verdict") in ["CORRECT"]:
                continue  # skip correct memories

            past_words = set(mem.get("question", "").lower().split())
            if not past_words:
                continue

            # Jaccard similarity
            overlap = len(query_words & past_words)
            union   = len(query_words | past_words)
            score   = overlap / union if union > 0 else 0

            if score > 0.5 and score > best_score:
                best_score  = score
                best_match  = mem

        return best_match

    def get_stats(self) -> dict:
        total    = len(self.memories)
        correct  = sum(1 for m in self.memories if m.get("verdict") == "CORRECT")
        wrong    = sum(1 for m in self.memories if m.get("verdict") == "WRONG")
        partial  = sum(1 for m in self.memories if m.get("verdict") == "PARTIALLY_CORRECT")
        return {
            "total":    total,
            "correct":  correct,
            "wrong":    wrong,
            "partial":  partial,
            "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
        }

    def get_lessons(self, limit: int = 5) -> str:
        """
        Get recent correction lessons to inject into LLM prompt.
        This prevents repeat mistakes.
        """
        wrong_memories = [
            m for m in self.memories[-20:]
            if m.get("verdict") in ["WRONG", "PARTIALLY_CORRECT"]
            and m.get("key_corrections")
        ]

        if not wrong_memories:
            return ""

        lessons = ["PAST MISTAKES TO AVOID:"]
        for mem in wrong_memories[-limit:]:
            lessons.append(f"- Q: {mem['question'][:80]}")
            lessons.append(f"  Lesson: {mem['key_corrections'][:150]}")

        return "\n".join(lessons)


# ── Gemini Validator via Selenium ─────────────────────────
def validate_with_gemini(
    driver,
    question: str,
    answer: str,
    context: str = "",
    wait_secs: int = 40,
) -> dict:
    """
    Send question + answer to Gemini for validation.
    Uses existing Selenium driver (Brave, already logged in).

    Returns parsed validation dict.
    """
    prompt = VALIDATION_PROMPT_TEMPLATE.format(
        question=question,
        answer=answer[:800],
        context=context[:500] if context else "Not available",
    )

    try:
        driver.get("https://gemini.google.com/app")
        time.sleep(4)

        # Find input box
        input_box = None
        selectors = [
            "rich-textarea div[contenteditable='true']",
            "div[contenteditable='true'][data-placeholder]",
            "div.ql-editor",
            "textarea",
        ]
        for sel in selectors:
            try:
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC
                input_box = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
                )
                if input_box:
                    break
            except Exception:
                continue

        if not input_box:
            return {"error": "Could not find Gemini input box"}

        # Type the validation prompt
        input_box.click()
        time.sleep(0.5)
        # Clear first
        input_box.send_keys(Keys.CONTROL + "a")
        input_box.send_keys(Keys.DELETE)
        time.sleep(0.3)
        input_box.send_keys(prompt)
        time.sleep(0.5)
        input_box.send_keys(Keys.RETURN)

        # Smart wait for response
        print(f"[SelfHealer] Waiting for Gemini validation (max {wait_secs}s)...")
        waited   = 0
        interval = 3
        response_text = ""

        while waited < wait_secs:
            time.sleep(interval)
            waited += interval

            for sel in [
                "div.response-container-content",
                "div.model-response-text",
                "p[data-path-to-node]",
            ]:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, sel)
                    if elements:
                        text = elements[-1].text.strip()
                        if len(text) > 100 and "VERDICT:" in text:
                            prev_len = len(text)
                            time.sleep(2)
                            elements2 = driver.find_elements(By.CSS_SELECTOR, sel)
                            if elements2:
                                new_len = len(elements2[-1].text.strip())
                                if new_len >= prev_len:
                                    response_text = elements2[-1].text.strip()
                                    print(f"[SelfHealer] Gemini responded ({new_len} chars)")
                                    break
                except Exception:
                    continue

            if response_text:
                break
            print(f"[SelfHealer] Still waiting... ({waited}s)")

        if not response_text:
            return {"error": "Gemini did not respond in time"}

        return _parse_gemini_validation(response_text)

    except Exception as e:
        return {"error": str(e)}


def _parse_gemini_validation(response: str) -> dict:
    """Parse Gemini's structured validation response."""

    def extract(label: str, text: str) -> str:
        pattern = rf"{label}:\s*(.+?)(?=\n[A-Z_]+:|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    verdict          = extract("VERDICT", response).upper()
    accuracy_str     = extract("ACCURACY_SCORE", response)
    what_is_wrong    = extract("WHAT_IS_WRONG", response)
    correct_answer   = extract("CORRECT_ANSWER", response)
    key_corrections  = extract("KEY_CORRECTIONS", response)
    reason_for_error = extract("REASON_FOR_ERROR", response)

    # Parse accuracy score
    try:
        accuracy = float(re.search(r'\d+\.?\d*', accuracy_str).group())
    except Exception:
        accuracy = 5.0

    # Normalize verdict
    if verdict not in ["CORRECT", "PARTIALLY_CORRECT", "WRONG"]:
        if accuracy >= 8:
            verdict = "CORRECT"
        elif accuracy >= 5:
            verdict = "PARTIALLY_CORRECT"
        else:
            verdict = "WRONG"

    return {
        "verdict":          verdict,
        "accuracy_score":   accuracy,
        "what_is_wrong":    what_is_wrong,
        "correct_answer":   correct_answer,
        "key_corrections":  key_corrections,
        "reason_for_error": reason_for_error,
        "raw_response":     response[:500],
    }


# ── Main Self-Healing Function ────────────────────────────
def self_heal(
    driver,
    question: str,
    original_answer: str,
    context: str,
    memory: HealingMemory,
) -> dict:
    """
    Full self-healing pipeline.

    1. Validate answer with Gemini
    2. Save result to permanent memory
    3. Return corrected answer + validation info

    Returns:
    {
        "final_answer":   str,   # corrected or original
        "verdict":        str,   # CORRECT / PARTIALLY_CORRECT / WRONG
        "accuracy_score": float,
        "was_corrected":  bool,
        "what_was_wrong": str,
        "gemini_said":    str,   # full correction for display
    }
    """

    # Validate with Gemini
    validation = validate_with_gemini(driver, question, original_answer, context)

    if "error" in validation:
        print(f"[SelfHealer] Validation error: {validation['error']}")
        return {
            "final_answer":   original_answer,
            "verdict":        "UNKNOWN",
            "accuracy_score": 0,
            "was_corrected":  False,
            "what_was_wrong": "",
            "gemini_said":    f"Validation failed: {validation['error']}",
        }

    verdict       = validation.get("verdict", "UNKNOWN")
    correct_ans   = validation.get("correct_answer", "").strip()
    what_is_wrong = validation.get("what_is_wrong", "").strip()
    key_corr      = validation.get("key_corrections", "").strip()
    accuracy      = validation.get("accuracy_score", 5.0)

    # Decide final answer
    was_corrected = False
    if verdict in ["WRONG", "PARTIALLY_CORRECT"] and correct_ans and len(correct_ans) > 20:
        final_answer  = correct_ans
        was_corrected = True
    else:
        final_answer  = original_answer

    # Save to permanent memory
    memory_entry = {
        "timestamp":       datetime.now().isoformat(),
        "question":        question,
        "original_answer": original_answer[:300],
        "verdict":         verdict,
        "accuracy_score":  accuracy,
        "what_is_wrong":   what_is_wrong[:300],
        "correct_answer":  correct_ans[:300],
        "key_corrections": key_corr[:300],
        "reason_for_error": validation.get("reason_for_error", "")[:200],
        "was_corrected":   was_corrected,
    }
    memory.save(memory_entry)
    print(f"[SelfHealer] Saved to memory. Verdict: {verdict}, Score: {accuracy}/10")

    # Build Gemini feedback summary for display
    gemini_said_parts = [f"Verdict: {verdict} ({accuracy}/10)"]
    if what_is_wrong and what_is_wrong.lower() != "none":
        gemini_said_parts.append(f"Issues: {what_is_wrong[:200]}")
    if was_corrected:
        gemini_said_parts.append("Answer was corrected by Gemini")
    if key_corr:
        gemini_said_parts.append(f"Lesson saved: {key_corr[:150]}")

    return {
        "final_answer":   final_answer,
        "verdict":        verdict,
        "accuracy_score": accuracy,
        "was_corrected":  was_corrected,
        "what_was_wrong": what_is_wrong,
        "gemini_said":    "\n".join(gemini_said_parts),
    }