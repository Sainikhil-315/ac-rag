"""
ingestion/table_extractor.py
Extracts real tables from PDF pages using pdfplumber's structural table detection,
producing retrieval-ready Document chunks tagged modality="table".

Why this exists (see GAPS_AND_IMPROVEMENTS.md item C.14):
  metadata_tagger.py's _detect_modality() only looks for pipe/tab characters in
  already-chunked plain text. PyMuPDF's page-text extraction flattens PDF tables
  into space-separated prose with no such markers, so that heuristic tagged ZERO
  chunks as "table" on the actual AC-RAG paper — confirmed empirically — even
  though the document contains a system-comparison matrix (Table 1) and a
  parameter/value config table (Table 9) that a jury member would reasonably
  expect "modality=table" filtering to surface.

Design (mirrors figure_extractor.py's pattern):
  - pdfplumber's page.extract_tables() detects table structure directly from PDF
    layout (cell boundaries), independent of how PyMuPDF's plain-text extraction
    happens to flatten the same content. No LLM call needed — table content is
    already structured, unlike figures, so this is free and fast rather than
    the paid GPT-4o Vision call figure extraction requires.
  - Each detected table is rendered as a clean markdown pipe table, which also
    gives it pipe characters — so it would now ALSO satisfy the original regex
    heuristic if ever re-chunked, though this path bypasses that heuristic by
    tagging modality="table" directly.
  - Skips tables with fewer than MIN_TABLE_ROWS rows (avoids false positives on
    pdfplumber misreading a two-column layout as a 2-row "table").
  - Only processes PDFs — other formats aren't handled by pdfplumber here.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

MIN_TABLE_ROWS = 3  # below this, more likely a false-positive layout artifact


def _make_table_chunk_id(source: str, page: int, table_index: int) -> str:
    raw = f"{source}::page{page}::table{table_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _table_to_markdown(rows: List[List[Optional[str]]]) -> str:
    """Render a pdfplumber table (list of row lists, cells may be None) as markdown."""
    cleaned = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in rows]
    if not cleaned:
        return ""

    header, body = cleaned[0], cleaned[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in body:
        # Pad/truncate rows that don't match header width (pdfplumber occasionally
        # misdetects a merged cell) rather than dropping the row entirely.
        row = (row + [""] * len(header))[: len(header)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_tables(file_path: str | Path) -> List[Document]:
    """
    Extract all real tables from a PDF and return them as Document chunks.

    Each Document has:
      - page_content : the table rendered as markdown (embeds and displays cleanly)
      - metadata     : source, file_name, page, table_index, modality="table",
                       chunk_id, word_count, char_count, section_heading="Table"

    Non-PDF files are silently ignored (returns empty list) — same contract as
    figure_extractor.extract_figures().
    """
    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        return []

    try:
        import pdfplumber
    except ImportError:
        logger.warning("[TableExtractor] pdfplumber not installed — table extraction skipped.")
        return []

    table_docs: List[Document] = []
    table_global_index = 0

    try:
        pdf = pdfplumber.open(str(path))
    except Exception as e:
        logger.error("[TableExtractor] Cannot open '%s': %s", path.name, e)
        return []

    logger.info("[TableExtractor] Scanning '%s' (%d pages) for tables...", path.name, len(pdf.pages))

    try:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                raw_tables = page.extract_tables()
            except Exception as e:
                logger.warning("[TableExtractor] extract_tables failed on page %d: %s", page_num, e)
                continue

            for raw_table in raw_tables:
                if len(raw_table) < MIN_TABLE_ROWS:
                    logger.debug(
                        "[TableExtractor] Skipping %d-row table on page %d (below MIN_TABLE_ROWS=%d)",
                        len(raw_table), page_num, MIN_TABLE_ROWS,
                    )
                    continue

                markdown = _table_to_markdown(raw_table)
                if not markdown.strip():
                    continue

                chunk_id = _make_table_chunk_id(str(path), page_num, table_global_index)
                doc = Document(
                    page_content=markdown,
                    metadata={
                        "source":          str(path),
                        "file_name":       path.name,
                        "file_type":       ".pdf",
                        "page":            page_num,
                        "table_index":     table_global_index,
                        "modality":        "table",
                        "section_heading": "Table",
                        "chunk_id":        chunk_id,
                        "chunk_index":     table_global_index,
                        "word_count":      len(markdown.split()),
                        "char_count":      len(markdown),
                    },
                )
                table_docs.append(doc)
                table_global_index += 1
                logger.info(
                    "[TableExtractor] Table %d extracted from page %d (%d rows)",
                    table_global_index, page_num, len(raw_table),
                )
    finally:
        pdf.close()

    logger.info("[TableExtractor] Extracted %d table(s) from '%s'", len(table_docs), path.name)
    return table_docs
