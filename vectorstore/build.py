"""
vectorstore/build.py
One-shot script: ingest all documents from data/raw/ → embed → save index.

Run this once before starting the pipeline:
    python -m vectorstore.build

Re-run whenever you add new documents to data/raw/.
"""

import logging
import sys
from pathlib import Path

# Ensure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger
from config.settings import DATA_RAW_DIR, VECTORSTORE_DIR
from ingestion import ingest_documents
from vectorstore.store import VectorStoreManager

logger = setup_logger("ac_rag.build")


def build_index(source_dir: Path = DATA_RAW_DIR) -> VectorStoreManager:
    """
    Full build pipeline:
      1. Ingest documents from source_dir (load + chunk + tag)
      2. Embed all chunks
      3. Build FAISS index
      4. Save to disk

    Returns the ready VectorStoreManager (can be used immediately for queries).
    """
    logger.info("=== AC-RAG Vector Store Build ===")
    logger.info("Source  : %s", source_dir)
    logger.info("Index   : %s", VECTORSTORE_DIR)

    # Step 1: Ingest
    chunks = ingest_documents(source_dir)
    if not chunks:
        logger.error("No chunks produced. Add documents to data/raw/ and retry.")
        sys.exit(1)

    logger.info("Chunks ready for embedding: %d", len(chunks))

    # Step 2 + 3: Embed and build
    vsm = VectorStoreManager()
    vsm.build(chunks)

    # Step 4: Save
    vsm.save()
    logger.info("Build complete.")
    return vsm


if __name__ == "__main__":
    build_index()
