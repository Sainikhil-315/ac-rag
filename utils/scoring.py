"""
utils/scoring.py
Embedding-based scoring utilities shared across pipeline nodes.

Responsibilities:
  - Score a list of passages against a query using cosine similarity
  - Compute pairwise cosine similarity matrix between passages
    (used by context refiner for semantic deduplication)

Design:
  Cosine similarity is computed in numpy after getting embeddings from
  the same model used to build the index — this ensures the similarity
  space is consistent with the retrieval space.

  Batching: passages are embedded in a single batch call where possible
  to minimise API round trips.
"""

import logging
from typing import List, Tuple

import numpy as np

from vectorstore.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def score_passages_against_query(
    query: str,
    passages: List[str],
) -> List[float]:
    """
    Embed the query and each passage, then compute cosine similarity.
    Returns a list of similarity scores in [0, 1], one per passage.

    Uses the same embedding model as the vector store to ensure
    similarity scores are on the same scale as retrieval scores.
    """
    if not passages:
        return []

    model = get_embedding_model()

    # Embed query
    query_emb = np.array(model.embed_query(query))

    # Embed all passages in one batch
    passage_embs = np.array(model.embed_documents(passages))

    scores = [
        round(_cosine_similarity(query_emb, passage_embs[i]), 4)
        for i in range(len(passages))
    ]

    logger.debug(
        "Scored %d passages. min=%.3f max=%.3f mean=%.3f",
        len(scores), min(scores), max(scores), float(np.mean(scores))
    )
    return scores


def pairwise_cosine_matrix(texts: List[str]) -> np.ndarray:
    """
    Compute an N×N cosine similarity matrix for a list of texts.
    Used by context refiner to detect redundant passages.
    Returns np.ndarray of shape (N, N) with values in [0, 1].
    """
    if not texts:
        return np.array([])

    model = get_embedding_model()
    embs = np.array(model.embed_documents(texts))

    # Normalise rows to unit length
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    embs_norm = embs / norms

    sim_matrix = embs_norm @ embs_norm.T
    return np.clip(sim_matrix, 0.0, 1.0)
