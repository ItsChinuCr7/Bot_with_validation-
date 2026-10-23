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
from ingestion.lineage import attach_lineage_metadata

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
    processing_error = None

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
            if not pages:
                raise ValueError(
                    "No extractable text found; scanned PDFs require OCR."
                )

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
            logger.info(
                "PDF Loader: %d pages, %d tree nodes",
                len(pages),
                node_count,
            )

        except Exception as exc:
            logger.error("PDF Loader failed (%s)", type(exc).__name__)
            processing_error = exc
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if processing_error is not None:
        raise RuntimeError("Could not process an uploaded PDF.") from processing_error
    if not all_chunks:
        raise RuntimeError("No extractable text was found in the uploaded PDFs.")

    # Clean then chunk
    all_chunks = clean_documents(all_chunks)
    chunks     = chunk_documents(all_chunks)
    if not chunks:
        raise RuntimeError("No usable text chunks were extracted from the uploaded PDFs.")
    attach_lineage_metadata(chunks)
    logger.info(
        "PDF Loader: %d chunks, %d document tree(s) ready",
        len(chunks),
        len(all_trees),
    )
    return chunks, all_trees