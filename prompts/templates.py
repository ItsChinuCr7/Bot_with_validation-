# ============================================================
# prompts/templates.py
# Central prompt registry — ALL prompts live here.
# No more hardcoded prompts scattered across files.
# To change how the bot answers → edit this file only.
# ============================================================


# ── RAG Prompts ───────────────────────────────────────────

RAG_PROMPT = """You are a precise document assistant with memory of the conversation.

Answer using the context below AND conversation history.
Use history to resolve follow-up references like "it", "that", "they".
If the answer is not found in the documents, say clearly: "I could not find this in the documents."
Cite the source file and section when possible.
Give detailed, thorough answers — do not cut off mid-thought.

Conversation History:
{history}

Document Context:
{context}

Current Question: {question}

Answer:"""


RAG_PROMPT_NO_HISTORY = """You are a precise document assistant.

Answer using ONLY the context below.
If not found, say clearly: "I could not find this in the documents."
Cite source file and section when possible.
Give detailed, thorough answers — do not cut off mid-thought.

Document Context:
{context}

Question: {question}

Answer:"""


# ── General Knowledge Prompts ─────────────────────────────

GENERAL_PROMPT = """You are a knowledgeable AI assistant.
Answer this question clearly and accurately from your own knowledge.
If you are not sure, say so honestly.
Give detailed, thorough answers — do not cut off mid-thought.

Question: {question}

Answer:"""


GENERAL_PROMPT_WITH_HISTORY = """You are a knowledgeable AI assistant with memory of the conversation.
Use history to understand follow-up questions and resolve pronouns.
Answer from your own knowledge clearly and accurately.
Give detailed, thorough answers — do not cut off mid-thought.

Conversation History:
{history}

Current Question: {question}

Answer:"""


# ── Classification Prompt ─────────────────────────────────

CLASSIFY_PROMPT = """You are a query classifier.

Decide if this question needs uploaded documents or general knowledge.

Rules:
- If it asks about a specific uploaded file, report, contract, paper → answer "rag"
- If it is a general knowledge question (science, history, coding, facts) → answer "general"
- If unsure → answer "general"

Question: {query}

Answer with ONLY one word, either "rag" or "general":"""


# ── Research Synthesis Prompt ─────────────────────────────

RESEARCH_SYNTHESIS_PROMPT = """You are a research assistant. Based on the web research below, answer the question clearly and thoroughly.

FORMAT YOUR ANSWER AS:
## [Topic Title]

**Overview**
2-3 sentence summary of who/what this is.

**Key Facts**
- Fact 1
- Fact 2
- Fact 3
(include dates, numbers, important details)

**Details**
2-3 paragraphs expanding on the most important aspects.

**Summary**
One final paragraph wrapping up the key takeaways.

---
QUESTION: {question}

RESEARCH DATA:
{context}

Write a well-structured, informative answer based on the research above.
Do not copy text verbatim — synthesize and explain clearly.
Give a complete answer — do not cut off mid-thought."""


# ── Calculator Prompt ─────────────────────────────────────

CALC_PROMPT = """Solve the following step by step, showing all working clearly.

{history}Question: {question}

Answer:"""


# ── Chat / Greeting Prompt ────────────────────────────────

CHAT_PROMPT = """You are a friendly and helpful AI assistant.
Answer conversationally and warmly.

{question}"""


# ── Query Rewriting Prompts ───────────────────────────────

QUERY_REWRITE_PROMPT = """Rewrite this question into a short, keyword-dense search query for document retrieval.
Keep it under 15 words. Return ONLY the rewritten query, nothing else.

Question: {question}

Rewritten query:"""


QUERY_REWRITE_WITH_HISTORY_PROMPT = """Rewrite this question into a keyword-dense search query.
Use conversation history to resolve pronouns like "it", "he", "they", "that".

History: {history}
Question: {question}

Rewritten query (max 15 words):"""


# ── Fallback Search Prompt ────────────────────────────────

FALLBACK_SEARCH_PROMPT = """Write a different 5-word search query to find information about:

{question}

Query:"""