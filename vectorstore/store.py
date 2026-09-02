"""
vectorstore/store.py
VectorStoreManager — the single interface for all vector DB operations.

Responsibilities:
  1. Build a new index from a list of Documents
  2. Persist index to disk (FAISS) or collection (Chroma)
  3. Load a previously saved index
  4. Similarity search (cosine)
  5. MMR search (Maximum Marginal Relevance) — used by the Retriever node

Design decisions:
  - FAISS chosen as default: fast, local, no server needed, good for research.
  - Chroma available as alternative when persistent HTTP server is preferred.
  - MMR is exposed here so the Retrieval Planner can call it directly with
    dynamic k values without touching FAISS internals.
  - All metadata is preserved inside the vector store (FAISS supports this
    via the docstore).
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import (
    VECTORSTORE_BACKEND,
    VECTORSTORE_DIR,
    RETRIEVAL_K_DEFAULT,
    MMR_FETCH_K_MULTIPLIER,
    MMR_LAMBDA_MULT,
)
from vectorstore.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Unified interface for building, saving, loading, and querying a vector store.

    Usage:
        # Build from scratch
        vsm = VectorStoreManager()
        vsm.build(chunks)
        vsm.save()

        # Load existing
        vsm = VectorStoreManager()
        vsm.load()

        # Query
        results = vsm.mmr_search(query, k=6)
    """

    def __init__(self, embedding_model: Optional[Embeddings] = None):
        self.embeddings = embedding_model or get_embedding_model()
        self._store = None   # underlying LangChain vectorstore object
        self._backend = VECTORSTORE_BACKEND.lower()
        self._index_path = Path(VECTORSTORE_DIR)

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, documents: List[Document]) -> None:
        """
        Embed all documents and build the in-memory vector index.
        Call save() afterwards to persist to disk.
        """
        if not documents:
            raise ValueError("Cannot build vector store: document list is empty.")

        logger.info(
            "Building %s index from %d chunks...", self._backend.upper(), len(documents)
        )

        if self._backend == "faiss":
            from langchain_community.vectorstores import FAISS
            self._store = FAISS.from_documents(documents, self.embeddings)

        elif self._backend == "chroma":
            from langchain_community.vectorstores import Chroma
            self._store = Chroma.from_documents(
                documents,
                self.embeddings,
                persist_directory=str(self._index_path),
            )
        else:
            raise ValueError(f"Unknown VECTORSTORE_BACKEND: '{self._backend}'")

        logger.info("Vector index built. Total vectors: %d", self._store.index.ntotal
                    if self._backend == "faiss" else len(documents))

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the index to disk."""
        self._assert_built()
        self._index_path.mkdir(parents=True, exist_ok=True)

        if self._backend == "faiss":
            self._store.save_local(str(self._index_path))
            logger.info("FAISS index saved to '%s'", self._index_path)

        elif self._backend == "chroma":
            # Chroma auto-persists when persist_directory is set; explicit call for safety
            self._store.persist()
            logger.info("Chroma index persisted to '%s'", self._index_path)

    def load(self) -> None:
        """Load a previously saved index from disk."""
        if not self._index_path.exists():
            raise FileNotFoundError(
                f"No saved index found at '{self._index_path}'. "
                "Run vectorstore/build.py first."
            )

        logger.info("Loading %s index from '%s'...", self._backend.upper(), self._index_path)

        if self._backend == "faiss":
            from langchain_community.vectorstores import FAISS
            self._store = FAISS.load_local(
                str(self._index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,  # required by LangChain ≥0.2
            )

        elif self._backend == "chroma":
            from langchain_community.vectorstores import Chroma
            self._store = Chroma(
                persist_directory=str(self._index_path),
                embedding_function=self.embeddings,
            )

        logger.info("Vector index loaded.")

    # ── Incremental update ────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to an existing index (incremental update).
        If no index exists yet, builds one from scratch.
        Call save() afterwards to persist the updated index.
        """
        if not documents:
            return

        if self._store is None:
            logger.info("No existing index — building new index from %d chunks.", len(documents))
            self.build(documents)
            return

        logger.info("Adding %d chunks to existing %s index…", len(documents), self._backend.upper())

        if self._backend == "faiss":
            self._store.add_documents(documents)
        elif self._backend == "chroma":
            self._store.add_documents(documents)
        else:
            raise ValueError(f"Unknown VECTORSTORE_BACKEND: '{self._backend}'")

        logger.info("Index updated. Total vectors: %d",
                    self._store.index.ntotal if self._backend == "faiss" else "?")

    # ── Search ────────────────────────────────────────────────────────────────

    def similarity_search(
        self, query: str, k: int = RETRIEVAL_K_DEFAULT
    ) -> List[Document]:
        """
        Standard cosine similarity search.
        Returns top-k documents ordered by relevance.
        """
        self._assert_built()
        results = self._store.similarity_search(query, k=k)
        logger.debug("Similarity search k=%d → %d results", k, len(results))
        return results

    def similarity_search_with_score(
        self, query: str, k: int = RETRIEVAL_K_DEFAULT
    ) -> List[Tuple[Document, float]]:
        """
        Similarity search that also returns the cosine distance score.
        Used by the Validator node to filter low-confidence passages.
        Returns list of (Document, score) pairs; lower score = more similar in FAISS L2.
        Scores are normalised to [0, 1] similarity (1 = perfect match).
        """
        self._assert_built()
        raw = self._store.similarity_search_with_score(query, k=k)

        # FAISS returns L2 distance (lower = better). Convert to similarity ∈ [0, 1].
        # similarity = 1 / (1 + distance)  — monotone, bounded
        normalised = [
            (doc, round(1.0 / (1.0 + float(dist)), 4))
            for doc, dist in raw
        ]
        logger.debug(
            "Scored search k=%d → scores: %s",
            k, [s for _, s in normalised],
        )
        return normalised

    def mmr_search(
        self,
        query: str,
        k: int = RETRIEVAL_K_DEFAULT,
        fetch_k: Optional[int] = None,
        lambda_mult: float = MMR_LAMBDA_MULT,
    ) -> List[Document]:
        """
        MMR (Maximum Marginal Relevance) search.
        Balances relevance and diversity to avoid redundant passages.

        Args:
            query      : search query string
            k          : number of final documents to return
            fetch_k    : candidate pool size (default: k * MMR_FETCH_K_MULTIPLIER)
            lambda_mult: 0 = max diversity, 1 = max relevance
        """
        self._assert_built()
        fetch_k = fetch_k or (k * MMR_FETCH_K_MULTIPLIER)

        results = self._store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
        logger.debug("MMR search k=%d, fetch_k=%d → %d results", k, fetch_k, len(results))
        return results

    def as_retriever(self, search_type: str = "mmr", k: int = RETRIEVAL_K_DEFAULT):
        """
        Returns a LangChain Retriever object.
        Useful for compatibility with LangChain chains/tools.
        """
        self._assert_built()
        search_kwargs = {"k": k}
        if search_type == "mmr":
            search_kwargs["fetch_k"] = k * MMR_FETCH_K_MULTIPLIER
            search_kwargs["lambda_mult"] = MMR_LAMBDA_MULT
        return self._store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _assert_built(self) -> None:
        if self._store is None:
            raise RuntimeError(
                "Vector store not initialised. Call build() or load() first."
            )

    @property
    def is_ready(self) -> bool:
        return self._store is not None
