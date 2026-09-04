"""
Fast dev build: ingest the AC-RAG paper as text + tables only.
Skips GPT-4o Vision figure extraction (which is slow during dev iteration).

For the full build with figures, use `python -m vectorstore.build`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger
from ingestion.loader import load_directory
from ingestion.chunker import chunk_documents
from ingestion.metadata_tagger import tag_chunks
from ingestion.table_extractor import extract_tables
from vectorstore.store import VectorStoreManager
from config.settings import DATA_RAW_DIR, VECTORSTORE_DIR

logger = setup_logger("ac_rag.fast_build")


def main():
    print(f"=== Fast build from {DATA_RAW_DIR} ===")
    # Step 1: load + chunk + tag (text only)
    raw = load_directory(DATA_RAW_DIR)
    chunks = chunk_documents(raw)
    chunks = tag_chunks(chunks)
    print(f"  Text chunks: {len(chunks)}")

    # Step 2: add tables (no LLM call, structural extraction)
    pdfs = list(DATA_RAW_DIR.rglob("*.pdf"))
    table_chunks = []
    for pdf in pdfs:
        table_chunks.extend(extract_tables(pdf))
    print(f"  Table chunks: {len(table_chunks)}")

    all_chunks = chunks + table_chunks
    print(f"  Total: {len(all_chunks)} chunks (no figures)")

    if not all_chunks:
        print("No chunks to index!")
        return

    vsm = VectorStoreManager()
    vsm.build(all_chunks)
    vsm.save()
    print(f"  Index saved to {VECTORSTORE_DIR}")
    print(f"  Total vectors: {vsm._store.index.ntotal}")


if __name__ == "__main__":
    main()
