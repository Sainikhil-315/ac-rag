"""
pipeline/retrieval/hybrid.py
Hybrid Dense + BM25 Retrieval Fusion.

Combines candidates retrieved via dense vector search (FAISS) and lexical BM25 search.
Supports Reciprocal Rank Fusion (RRF) and Score Normalization / Linear Fusion.
Preserves full document provenance, metadata, and attribution.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional

from config.settings import (
    HYBRID_FUSION_METHOD,
    HYBRID_DENSE_WEIGHT,
    HYBRID_BM25_WEIGHT,
    RRF_K,
)

logger = logging.getLogger(__name__)


def _get_doc_id(doc: Dict[str, Any]) -> str:
    """
    Get a unique identifier for a doc based on chunk_id or content hash.
    """
    meta = doc.get("metadata", {})
    if "chunk_id" in meta and meta["chunk_id"]:
        return str(meta["chunk_id"])
    content = doc.get("content") or doc.get("page_content") or ""
    source = meta.get("source", "")
    return hashlib.md5(f"{source}:{content[:200]}".encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    top_k: int = 6,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
    rrf_k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """
    Combine dense and BM25 search results using Reciprocal Rank Fusion (RRF).

    RRF_Score(d) = (w_dense / (rrf_k + rank_dense)) + (w_bm25 / (rrf_k + rank_bm25))

    Returns:
        Deduplicated list of top_k docs with updated 'score' field reflecting RRF score.
    """
    doc_map: Dict[str, Dict[str, Any]] = {}
    rrf_scores: Dict[str, float] = {}

    # Process dense results
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = _get_doc_id(doc)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        score_contrib = dense_weight / (rrf_k + rank)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score_contrib

    # Process BM25 results
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = _get_doc_id(doc)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        score_contrib = bm25_weight / (rrf_k + rank)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score_contrib

    # Sort doc_ids by RRF score descending
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)

    max_rrf = rrf_scores[sorted_doc_ids[0]] if sorted_doc_ids else 1.0

    fused_results: List[Dict[str, Any]] = []
    for doc_id in sorted_doc_ids[:top_k]:
        original_doc = doc_map[doc_id]
        raw_rrf = rrf_scores[doc_id]
        norm_score = raw_rrf / (max_rrf if max_rrf > 0 else 1.0)

        # Merge metadata cleanly
        new_doc = {
            "content": original_doc.get("content") or original_doc.get("page_content", ""),
            "metadata": dict(original_doc.get("metadata", {})),
            "score": float(norm_score),
            "rrf_score": float(raw_rrf),
            "retrieval_method": "hybrid_rrf",
        }
        fused_results.append(new_doc)

    return fused_results


def linear_fusion(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    top_k: int = 6,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
) -> List[Dict[str, Any]]:
    """
    Combine dense and BM25 results using linear score combination.
    """
    doc_map: Dict[str, Dict[str, Any]] = {}
    combined_scores: Dict[str, float] = {}

    for doc in dense_results:
        doc_id = _get_doc_id(doc)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        raw_score = doc.get("score", 0.0)
        combined_scores[doc_id] = combined_scores.get(doc_id, 0.0) + (dense_weight * raw_score)

    for doc in bm25_results:
        doc_id = _get_doc_id(doc)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        raw_score = doc.get("score", 0.0)
        combined_scores[doc_id] = combined_scores.get(doc_id, 0.0) + (bm25_weight * raw_score)

    sorted_ids = sorted(combined_scores.keys(), key=lambda did: combined_scores[did], reverse=True)

    max_score = combined_scores[sorted_ids[0]] if sorted_ids else 1.0

    fused_results: List[Dict[str, Any]] = []
    for doc_id in sorted_ids[:top_k]:
        original_doc = doc_map[doc_id]
        score_val = combined_scores[doc_id]
        norm_score = score_val / (max_score if max_score > 0 else 1.0)

        new_doc = {
            "content": original_doc.get("content") or original_doc.get("page_content", ""),
            "metadata": dict(original_doc.get("metadata", {})),
            "score": float(norm_score),
            "retrieval_method": "hybrid_linear",
        }
        fused_results.append(new_doc)

    return fused_results


def hybrid_fuse_results(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    top_k: int = 6,
    method: str = HYBRID_FUSION_METHOD,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
) -> List[Dict[str, Any]]:
    """
    Main entry point for fusing dense vector search and BM25 lexical search candidates.
    """
    if not dense_results and not bm25_results:
        return []
    if not bm25_results:
        return dense_results[:top_k]
    if not dense_results:
        return bm25_results[:top_k]

    if method == "linear":
        return linear_fusion(dense_results, bm25_results, top_k, dense_weight, bm25_weight)
    else:
        return reciprocal_rank_fusion(dense_results, bm25_results, top_k, dense_weight, bm25_weight)
