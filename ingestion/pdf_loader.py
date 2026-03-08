# ============================================================
# ingestion/pdf_loader.py
# Loads PDFs, splits into chunks, AND builds document trees
# ============================================================

import tempfile
import os
import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.document_tree import DocumentTree
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


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
                if text.strip():
                    pages.append(Document(
                        page_content=text,
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
            print(f"[PDF Loader] {original_name}: {len(pages)} pages, {node_count} tree nodes")

        except Exception as e:
            print(f"[PDF Loader] Error loading {original_name}: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if not all_chunks:
        return [], {}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_chunks)
    print(f"[PDF Loader] Total: {len(chunks)} chunks, {len(all_trees)} document tree(s)")
    return chunks, all_trees