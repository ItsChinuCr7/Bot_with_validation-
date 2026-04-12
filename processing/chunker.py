# ============================================================
# processing/chunker.py
# Isolated chunking logic — pulled out of pdf_loader.py
# so it can be tuned, tested, or swapped independently.
# ============================================================

import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def chunk_documents(docs: list) -> list:
    """
    Split a list of Documents into chunks using
    RecursiveCharacterTextSplitter.

    Separators tried in order:
      paragraph → sentence → clause → word → character
    """
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function = len,
    )

    chunks = splitter.split_documents(docs)
    logger.info(f"Chunker: {len(docs)} docs → {len(chunks)} chunks "
                f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def chunk_text(text: str) -> list[str]:
    """
    Split a plain string into chunks.
    Returns list of strings (not Documents).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)