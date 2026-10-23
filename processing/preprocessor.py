# ============================================================
# processing/preprocessor.py
# Cleans raw PDF text before it goes into chunks/embeddings.
# Better input = better retrieval accuracy.
# ============================================================

import re
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline for raw PDF text.
    Runs all cleaners in sequence.
    """
    if not text or not text.strip():
        return ""

    text = _remove_page_artifacts(text)
    text = _fix_broken_lines(text)
    text = _normalize_whitespace(text)
    text = _remove_headers_footers(text)
    text = _fix_encoding_artifacts(text)
    text = _normalize_punctuation(text)

    return text.strip()


def clean_documents(docs: list) -> list:
    """
    Apply clean_text to a list of LangChain Document objects.
    Drops documents that are empty after cleaning.
    """
    from langchain_core.documents import Document

    cleaned = []
    dropped = 0

    for doc in docs:
        clean = clean_text(doc.page_content)
        if len(clean) < 30:          # too short to be useful
            dropped += 1
            continue
        cleaned.append(Document(
            page_content=clean,
            metadata=doc.metadata,
        ))

    if dropped:
        logger.info(f"Preprocessor: dropped {dropped} empty/short chunks")
    logger.info(f"Preprocessor: cleaned {len(cleaned)} documents")
    return cleaned


# ── Individual cleaners ────────────────────────────────────

def _remove_page_artifacts(text: str) -> str:
    """Remove page numbers, 'Page X of Y', standalone numbers."""
    # Page X / Page X of Y
    text = re.sub(r'\bPage\s+\d+\s*(of\s*\d+)?\b', '', text, flags=re.IGNORECASE)
    # Standalone numbers on their own line (page numbers)
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    return text


def _fix_broken_lines(text: str) -> str:
    """
    PDF text extraction often breaks mid-sentence across lines.
    Join lines that don't end with sentence-ending punctuation.
    """
    lines  = text.split('\n')
    result = []
    i      = 0

    while i < len(lines):
        line = lines[i].rstrip()

        # Empty line = paragraph break, keep it
        if not line.strip():
            result.append('')
            i += 1
            continue

        # If line ends without terminal punctuation and next line exists
        # and next line starts lowercase → broken sentence, join
        if (i + 1 < len(lines)
                and line
                and not re.search(r'[.!?:;)\]"\']\s*$', line)
                and len(lines[i + 1].strip()) > 0
                and lines[i + 1].strip()[0].islower()):
            result.append(line + ' ' + lines[i + 1].strip())
            i += 2
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs. Preserve paragraph breaks."""
    # Collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces/tabs into one
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove space at start of lines
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
    return text


def _remove_headers_footers(text: str) -> str:
    """
    Remove repeated short lines that appear to be headers/footers
    (e.g. document title repeated on every page).
    """
    lines   = text.split('\n')
    # Count line frequency
    from collections import Counter
    counts  = Counter(l.strip() for l in lines if len(l.strip()) > 3)
    # Lines appearing 4+ times in a doc are likely headers/footers
    repeated = {line for line, count in counts.items() if count >= 4}
    cleaned  = [l for l in lines if l.strip() not in repeated]
    return '\n'.join(cleaned)


def _fix_encoding_artifacts(text: str) -> str:
    """Fix common PDF encoding issues and unicode garbage."""
    replacements = {
        '\ufb01': 'fi',   # ﬁ ligature
        '\ufb02': 'fl',   # ﬂ ligature
        '\u2019': "'",    # right single quote
        '\u2018': "'",    # left single quote
        '\u201c': '"',    # left double quote
        '\u201d': '"',    # right double quote
        '\u2013': '-',    # en dash
        '\u2014': '--',   # em dash
        '\u00a0': ' ',    # non-breaking space
        '\u2022': '-',    # bullet point
        '\u25cf': '-',    # black circle bullet
        '\xad':   '',     # soft hyphen
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Remove other non-printable characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text


def _normalize_punctuation(text: str) -> str:
    """Fix spacing around punctuation."""
    # Space before punctuation (except opening brackets)
    text = re.sub(r'\s([.,;:!?)\]])', r'\1', text)
    # Multiple punctuation marks
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r'-{3,}', '--', text)
    return text