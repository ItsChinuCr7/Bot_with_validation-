# ============================================================
# ingestion/document_tree.py
#
# Builds a hierarchical tree from documents.
# Each heading/section becomes a node.
# Enables structure-aware navigation instead of pure similarity.
#
# Tree structure:
#   DocumentTree
#   └── Node(title, content, level, children, page)
#       └── Node(...)
#           └── Node(...)
# ============================================================

import re
from dataclasses import dataclass, field
from typing import Optional
import fitz  # pymupdf


@dataclass
class TreeNode:
    """A single node in the document tree."""
    title:    str
    content:  str
    level:    int        # 1=chapter, 2=section, 3=subsection
    page:     int
    source:   str
    children: list = field(default_factory=list)
    parent:   Optional[object] = field(default=None, repr=False)

    def full_path(self) -> str:
        """Return full path like: Chapter 1 > Section 1.2 > Subsection 1.2.3"""
        parts = []
        node = self
        while node:
            parts.append(node.title)
            node = node.parent
        return " > ".join(reversed(parts))

    def to_dict(self) -> dict:
        return {
            "title":    self.title,
            "content":  self.content[:200],
            "level":    self.level,
            "page":     self.page,
            "source":   self.source,
            "path":     self.full_path(),
            "children": [c.title for c in self.children],
        }


class DocumentTree:
    """
    Hierarchical tree representation of a document.
    Built by detecting headings and section structure.
    """

    def __init__(self, source_name: str):
        self.source   = source_name
        self.root     = TreeNode(
            title="Document Root",
            content="",
            level=0,
            page=0,
            source=source_name,
        )
        self.all_nodes: list[TreeNode] = []

    def _detect_heading_level(self, text: str, font_size: float = 0) -> int:
        """
        Detect if a line is a heading and what level.
        Returns 0 if not a heading, 1-3 for heading levels.
        """
        text = text.strip()
        if not text or len(text) > 200:
            return 0

        # Numbered headings: 1. / 1.1 / 1.1.1
        if re.match(r'^\d+\.\s+\w', text):
            return 1
        if re.match(r'^\d+\.\d+\s+\w', text):
            return 2
        if re.match(r'^\d+\.\d+\.\d+\s+\w', text):
            return 3

        # ALL CAPS headings
        if text.isupper() and 3 < len(text) < 80:
            return 1

        # Font size based (if available from PDF extraction)
        if font_size > 16:
            return 1
        if font_size > 13:
            return 2
        if font_size > 11:
            return 3

        # Common heading patterns
        heading_patterns = [
            r'^(Chapter|CHAPTER|Section|SECTION|Part|PART)\s+\w',
            r'^(Introduction|Conclusion|Abstract|Summary|Overview|Background)',
            r'^(References|Bibliography|Appendix)',
        ]
        for pattern in heading_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return 1

        return 0

    def build_from_pdf(self, pdf_path: str) -> None:
        """Build tree from PDF using font sizes and text patterns."""
        try:
            pdf = fitz.open(pdf_path)
            current_content = []
            current_node    = self.root
            last_level      = 0

            for page_num in range(len(pdf)):
                page   = pdf[page_num]
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text      = span.get("text", "").strip()
                            font_size = span.get("size", 0)

                            if not text:
                                continue

                            level = self._detect_heading_level(text, font_size)

                            if level > 0:
                                # Save accumulated content to current node
                                if current_content:
                                    current_node.content += " ".join(current_content)
                                    current_content = []

                                # Create new node
                                new_node = TreeNode(
                                    title=text,
                                    content="",
                                    level=level,
                                    page=page_num + 1,
                                    source=self.source,
                                )

                                # Find correct parent
                                parent = self._find_parent(level)
                                new_node.parent = parent
                                parent.children.append(new_node)
                                self.all_nodes.append(new_node)
                                current_node = new_node
                                last_level   = level
                            else:
                                current_content.append(text + " ")

            # Save remaining content
            if current_content:
                current_node.content += " ".join(current_content)

            pdf.close()

        except Exception as e:
            print(f"[DocumentTree] Error building tree: {e}")

    def _find_parent(self, level: int) -> TreeNode:
        """Find appropriate parent node for a given heading level."""
        if level == 1 or not self.all_nodes:
            return self.root

        # Walk back through nodes to find parent
        for node in reversed(self.all_nodes):
            if node.level < level:
                return node

        return self.root

    def build_from_chunks(self, chunks: list) -> None:
        """
        Fallback: build a flat tree from LangChain document chunks.
        Used when PDF font-based detection isn't available.
        """
        for chunk in chunks:
            text   = chunk.page_content
            source = chunk.metadata.get("source_file", "unknown")
            page   = chunk.metadata.get("page", 0)

            # Try to detect a heading in first line
            first_line = text.split("\n")[0].strip()
            level      = self._detect_heading_level(first_line)

            if level > 0:
                node = TreeNode(
                    title=first_line,
                    content=text,
                    level=level,
                    page=page,
                    source=source,
                )
            else:
                # Create generic section node
                node = TreeNode(
                    title=f"Section (p.{page})",
                    content=text,
                    level=3,
                    page=page,
                    source=source,
                )

            parent = self._find_parent(node.level)
            node.parent = parent
            parent.children.append(node)
            self.all_nodes.append(node)

    def get_outline(self) -> str:
        """
        Return a text outline of the document structure.
        This is fed to the LLM so it can reason about where to look.
        """
        lines = [f"Document: {self.source}", "Structure:"]

        def _walk(node, indent=0):
            if node.title != "Document Root":
                lines.append("  " * indent + f"[L{node.level}] {node.title} (p.{node.page})")
            for child in node.children:
                _walk(child, indent + 1)

        _walk(self.root)
        return "\n".join(lines)

    def find_nodes_by_title(self, keywords: list) -> list:
        """Find nodes whose titles match given keywords."""
        results = []
        for node in self.all_nodes:
            title_lower = node.title.lower()
            if any(kw.lower() in title_lower for kw in keywords):
                results.append(node)
        return results

    def get_node_with_context(self, node: TreeNode, include_children: bool = True) -> str:
        """Get a node's content plus its children's content."""
        parts = [f"[{node.full_path()}]\n{node.content}"]
        if include_children:
            for child in node.children:
                parts.append(f"[{child.full_path()}]\n{child.content}")
        return "\n\n".join(parts)