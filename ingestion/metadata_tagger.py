"""
ingestion/metadata_tagger.py
Enriches each chunk with structured metadata needed downstream:

  - section_heading : inferred from text heuristics (lines that look like headings)
  - modality        : "text" | "table" | "figure"
  - word_count      : quick quality signal
  - char_count      : used for confidence scoring in validator

Design note:
  Tagging at ingest time (not retrieval time) keeps the retrieval path fast
  and allows the Retrieval Planner to filter by modality without re-scanning text.
"""

import re
import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Heading patterns: lines that are short, may be ALL CAPS or Title Case,
# optionally preceded by numbering like "1.", "1.1", "Chapter 2"
_HEADING_RE = re.compile(
    r"^(?:\d+[\.\d]*\s+)?[A-Z][A-Za-z0-9 \-:,]{3,80}$"
)

# Table signals: presence of pipe characters or structured whitespace grids
_TABLE_SIGNAL_RE = re.compile(r"(\|.+\|)|(\t.+\t.+\t)")

# Figure signals: captions starting with "Figure", "Fig.", "Chart", "Plot"
_FIGURE_SIGNAL_RE = re.compile(
    r"\b(figure|fig\.|chart|plot|diagram|image|illustration)\b",
    re.IGNORECASE,
)


def _detect_modality(text: str) -> str:
    if _TABLE_SIGNAL_RE.search(text):
        return "table"
    if _FIGURE_SIGNAL_RE.search(text):
        return "figure"
    return "text"


def _extract_section_heading(text: str) -> str:
    """
    Return the first line that looks like a section heading,
    or 'Unknown' if none found.
    """
    for line in text.splitlines():
        line = line.strip()
        if line and _HEADING_RE.match(line):
            return line[:120]   # cap length
    return "Unknown"


def tag_chunks(chunks: List[Document]) -> List[Document]:
    """
    In-place enrichment of chunk metadata.
    Returns the same list for chaining convenience.
    """
    for chunk in chunks:
        text = chunk.page_content

        chunk.metadata["modality"] = _detect_modality(text)
        chunk.metadata["section_heading"] = _extract_section_heading(text)
        chunk.metadata["word_count"] = len(text.split())
        chunk.metadata["char_count"] = len(text)

    logger.info(
        "Tagged %d chunks. Modality breakdown — text: %d, table: %d, figure: %d",
        len(chunks),
        sum(1 for c in chunks if c.metadata["modality"] == "text"),
        sum(1 for c in chunks if c.metadata["modality"] == "table"),
        sum(1 for c in chunks if c.metadata["modality"] == "figure"),
    )
    return chunks
