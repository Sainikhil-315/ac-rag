"""
tests/conftest.py
Shared pytest fixtures for the AC-RAG test suite.

No LLM or embedding API calls happen in this suite — every node under test either
has a pure-function core extracted for direct testing, or its LLM/embedding calls
are monkeypatched. This keeps the suite free, fast, and runnable in CI without
API keys.
"""

import sys
from pathlib import Path

# Ensure project root is importable when running `pytest` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from pipeline.state import initial_state


@pytest.fixture
def base_state():
    """A fresh ACRagState for a simple query, as produced by initial_state()."""
    return initial_state("What is the retrieval range (k)?")


def make_doc(content: str, chunk_id: str = "c1", score: float = None, **extra):
    """Build a retrieved-doc dict matching the shape retriever.py produces."""
    doc = {
        "content": content,
        "chunk_id": chunk_id,
        "source": "test.pdf",
        "page": 1,
        "section_heading": "Test Section",
        "modality": "text",
        "retrieved_by_query": "test query",
        "score": score,
    }
    doc.update(extra)
    return doc
