"""
ingestion/chunker.py
Splits loaded Documents into retrieval-sized chunks.

Strategy:
  - RecursiveCharacterTextSplitter as the base splitter.
    It respects paragraph → sentence → word boundaries in order,
    which preserves semantic coherence better than fixed-size splits.
  - Chunk size ~512 tokens (proxied as ~2048 chars at ~4 chars/token).
  - Overlap of 64 tokens (~256 chars) to avoid losing cross-boundary context.

Each chunk inherits parent metadata and gets a unique chunk_id.
"""

import hashlib
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# Character-level proxy: ~4 chars per token
_CHARS_PER_TOKEN = 4
_CHUNK_SIZE_CHARS = CHUNK_SIZE * _CHARS_PER_TOKEN
_CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP * _CHARS_PER_TOKEN


def _make_chunk_id(source: str, index: int) -> str:
    """Deterministic chunk ID based on source file and position."""
    raw = f"{source}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split a list of Documents into chunks.
    Returns a flat list of chunk Documents with enriched metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE_CHARS,
        chunk_overlap=_CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],  # hierarchy: paragraph → sentence → word
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks: List[Document] = []
    global_index = 0

    for doc in docs:
        splits = splitter.split_documents([doc])

        for local_index, chunk in enumerate(splits):
            chunk.metadata["chunk_id"] = _make_chunk_id(
                chunk.metadata.get("source", "unknown"), global_index
            )
            chunk.metadata["chunk_index"] = global_index
            chunk.metadata["local_chunk_index"] = local_index
            chunk.metadata["total_chunks_in_doc"] = len(splits)
            all_chunks.append(chunk)
            global_index += 1

    logger.info(
        "Chunked %d document(s) into %d chunks (size≈%d tokens, overlap≈%d tokens)",
        len(docs), len(all_chunks), CHUNK_SIZE, CHUNK_OVERLAP,
    )
    return all_chunks
