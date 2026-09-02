"""
ingestion/loader.py
Loads raw documents from disk into a unified list of LangChain Documents.
Supports: PDF, DOCX, TXT, HTML, Markdown.
Each document carries raw metadata: source path, file type, page number.
"""

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)

from config.settings import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


# Map extension → loader class
_LOADER_MAP = {
    ".pdf":   PyMuPDFLoader,
    ".docx":  Docx2txtLoader,
    ".txt":   TextLoader,
    ".html":  UnstructuredHTMLLoader,
    ".md":    UnstructuredMarkdownLoader,
}


def load_document(file_path: str | Path) -> List[Document]:
    """
    Load a single file and return a list of LangChain Documents.
    PyMuPDF returns one Document per page (good for page-level metadata).
    Other loaders return one Document per file.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    loader_cls = _LOADER_MAP[ext]
    loader = loader_cls(str(path))
    docs = loader.load()

    # Normalise metadata: ensure 'source' and 'file_type' are always present
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata["file_type"] = ext
        doc.metadata["file_name"] = path.name

    logger.info("Loaded %d page(s) from '%s'", len(docs), path.name)
    return docs


def load_directory(directory: str | Path) -> List[Document]:
    """
    Recursively load all supported documents from a directory.
    Returns a flat list of Documents across all files.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_docs: List[Document] = []
    files = [f for f in directory.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        logger.warning("No supported documents found in '%s'", directory)
        return all_docs

    for file_path in sorted(files):
        try:
            docs = load_document(file_path)
            all_docs.extend(docs)
        except Exception as e:
            logger.error("Failed to load '%s': %s", file_path.name, e)

    logger.info("Total documents loaded: %d from %d file(s)", len(all_docs), len(files))
    return all_docs
