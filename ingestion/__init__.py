"""
ingestion package
Exposes the full ingest pipeline as a single callable: ingest_documents()
"""

from ingestion.loader import load_document, load_directory
from ingestion.chunker import chunk_documents
from ingestion.metadata_tagger import tag_chunks
from ingestion.figure_extractor import extract_figures
from ingestion.table_extractor import extract_tables

__all__ = [
    "load_document", "load_directory", "chunk_documents",
    "tag_chunks", "extract_figures", "extract_tables", "ingest_documents",
]


def ingest_documents(source):
    """
    Full ingestion pipeline: load → chunk → tag → figure extraction → table extraction.

    For PDF files, figures are extracted separately using GPT-4o Vision and merged
    with the text chunks (modality="figure"). Tables are extracted structurally via
    pdfplumber and merged as clean markdown (modality="table") — see
    ingestion/table_extractor.py for why this exists: metadata_tagger's regex-based
    table heuristic detects zero real PDF tables, since PyMuPDF's plain-text
    extraction flattens table structure with no pipe/tab markers left to match.

    Note: table content is NOT missing from the flattened text chunks the way figure
    content is (PyMuPDF drops images entirely but does extract table text, just
    unstructured) — so a table's content may now appear twice in the index: once as
    noisy flattened prose (modality="text") and once as clean structured markdown
    (modality="table"). This is intentional redundancy, not a bug — different
    retrieval angles for the same content, and context_refiner's semantic dedup
    (cosine > 0.85) will collapse near-duplicates where the two representations
    end up similar enough.

    Args:
        source: path to a single file or a directory.

    Returns:
        List[Document] — tagged, chunked, metadata-enriched documents including
                         GPT-4o Vision figure descriptions and structured tables.
    """
    from pathlib import Path
    path = Path(source)

    if path.is_dir():
        raw_docs = load_directory(path)
    else:
        raw_docs = load_document(path)

    chunks = chunk_documents(raw_docs)
    tagged = tag_chunks(chunks)

    if path.is_dir():
        pdf_files = list(path.rglob("*.pdf"))
    else:
        pdf_files = [path] if path.suffix.lower() == ".pdf" else []

    # Figure chunks are already sized correctly — skip the text splitter
    figure_chunks = []
    table_chunks = []
    for pdf_path in pdf_files:
        figure_chunks.extend(extract_figures(pdf_path))
        table_chunks.extend(extract_tables(pdf_path))

    if figure_chunks or table_chunks:
        import logging
        logging.getLogger(__name__).info(
            "Merged %d figure chunk(s) and %d table chunk(s) into ingest output",
            len(figure_chunks), len(table_chunks),
        )

    return tagged + figure_chunks + table_chunks
