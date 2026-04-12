# ============================================================
# ingestion/pdf_loader.py
# Loads PDFs, splits into chunks, AND builds document trees
# ============================================================

import tempfile
import os
import logging
import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.document_tree import DocumentTree
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from processing.preprocessor import clean_text, clean_documents
from processing.chunker import chunk_documents

logger = logging.getLogger(__name__)


def load_and_split_pdfs(uploaded_files: list) -> tuple:
    """
    Load PDFs and return both:
    - chunks: for vector/hybrid search
    - trees:  for structure-aware tree navigation

    Returns:
        (chunks, document_trees)
    """
    all_chunks = []
    all_trees  = {}
    tmp_paths  = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_paths.append((tmp.name, file.name))

    for tmp_path, original_name in tmp_paths:
        try:
            # ── Extract text chunks ──────────────────────
            pdf   = fitz.open(tmp_path)
            pages = []
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = page.get_text("text")
                cleaned = clean_text(text)
                if cleaned:
                    pages.append(Document(
                        page_content=cleaned,
                        metadata={
                            "source_file": original_name,
                            "page":        page_num + 1,
                        }
                    ))
            pdf.close()

            # ── Build document tree ──────────────────────
            tree = DocumentTree(source_name=original_name)
            tree.build_from_pdf(tmp_path)

            # Fallback if tree is empty
            if not tree.all_nodes and pages:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                chunks_for_tree = splitter.split_documents(pages)
                tree.build_from_chunks(chunks_for_tree)

            all_trees[original_name] = tree
            all_chunks.extend(pages)

            node_count = len(tree.all_nodes)
            logger.info(f"PDF Loader: {original_name} — {len(pages)} pages, {node_count} tree nodes")

        except Exception as e:
            logger.error(f"PDF Loader error {original_name}: {e}", exc_info=True)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if not all_chunks:
        return [], {}

    # Clean then chunk
    all_chunks = clean_documents(all_chunks)
    chunks     = chunk_documents(all_chunks)
    logger.info(f"PDF Loader: {len(chunks)} chunks, {len(all_trees)} tree(s) ready")
    return chunks, all_trees