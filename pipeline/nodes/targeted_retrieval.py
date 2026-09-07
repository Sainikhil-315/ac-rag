"""
pipeline/nodes/targeted_retrieval.py
Targeted Retrieval Node.

Executes adaptive targeted retrieval specifically focusing on unresolved / missing evidence requirements.

Algorithm:
  1. Inspect state["missing_requirements"] and state["evidence_requirements"]
  2. For each missing requirement, construct targeted query & select modality/strategy
  3. Retrieve candidate passages (dense or BM25)
  4. Merge with existing state["retrieved_docs"] and state["retrieved_evidence"]
  5. Increment retry_count / retrieval_attempts counter
"""

import logging
from typing import Any, Dict, List

from config.settings import MAX_RETRIEVAL_ROUNDS, USE_TARGETED_RETRIEVAL
from pipeline.state import ACRagState
from vectorstore.store import VectorStoreManager

logger = logging.getLogger(__name__)


def make_targeted_retrieval_node(vsm: VectorStoreManager):
    """
    Factory that binds VectorStoreManager to the targeted retrieval node.
    """

    def targeted_retrieval_node(state: ACRagState) -> ACRagState:
        """
        LangGraph node: Targeted Retrieval.
        Reads:  state["missing_requirements"], state["evidence_requirements"], state["retrieved_docs"]
        Writes: state["retrieved_docs"], state["retrieved_evidence"], state["targeted_queries"]
        """
        missing_ids = state.get("missing_requirements") or []
        reqs = state.get("evidence_requirements") or []
        existing_docs = list(state.get("retrieved_docs") or [])
        existing_evidence = list(state.get("retrieved_evidence") or [])
        retry_count = state.get("retry_count", 0)

        log_entry: Dict[str, Any] = {
            "stage": "targeted_retrieval",
            "status": "started",
            "details": {
                "missing_ids": missing_ids,
                "retry_count": retry_count,
            },
        }

        if not missing_ids or not USE_TARGETED_RETRIEVAL:
            logger.info("[TargetedRetrieval] No missing requirements or disabled — skipping.")
            log_entry["status"] = "completed"
            log_entry["details"]["skipped"] = True
            return {
                **state,
                "stage_logs": state["stage_logs"] + [log_entry],
            }

        req_map = {r["id"]: r for r in reqs if isinstance(r, dict)}
        targeted_queries = []
        new_docs = []

        seen_chunk_ids = {d.get("chunk_id") for d in existing_docs if d.get("chunk_id")}

        for req_id in missing_ids:
            req = req_map.get(req_id)
            if not req:
                continue

            target_query = req.get("requirement") or state.get("query")
            keywords = req.get("keywords") or []
            if keywords:
                target_query = f"{target_query} {' '.join(keywords)}"

            req_type = req.get("requirement_type", "fact")
            modality = req.get("expected_modality", "all")

            # Choose strategy based on requirement type
            if req_type in ("numeric", "value", "procedure"):
                strategy = "bm25"
                retrieved_raw = vsm.bm25_search(target_query, k=4, modality_filter=modality)
                candidates = [
                    {
                        "content": d.page_content,
                        "metadata": d.metadata,
                        "chunk_id": d.metadata.get("chunk_id", "unknown"),
                        "source": d.metadata.get("source", "unknown"),
                        "page": d.metadata.get("page"),
                        "section_heading": d.metadata.get("section_heading", "Unknown"),
                        "modality": d.metadata.get("modality", "text"),
                        "retrieved_by_query": target_query,
                        "score": d.metadata.get("score", 0.7),
                    }
                    for d in retrieved_raw
                ]
            else:
                strategy = "hybrid"
                retrieved_raw = vsm.hybrid_search(target_query, k=4, modality_filter=modality)
                candidates = [
                    {
                        "content": d.page_content,
                        "metadata": d.metadata,
                        "chunk_id": d.metadata.get("chunk_id", "unknown"),
                        "source": d.metadata.get("source", "unknown"),
                        "page": d.metadata.get("page"),
                        "section_heading": d.metadata.get("section_heading", "Unknown"),
                        "modality": d.metadata.get("modality", "text"),
                        "retrieved_by_query": target_query,
                        "score": d.metadata.get("score", 0.7),
                    }
                    for d in retrieved_raw
                ]

            targeted_queries.append({
                "requirement_id": req_id,
                "target_query": target_query,
                "strategy": strategy,
                "modality": modality,
                "count_retrieved": len(candidates),
            })

            for cand in candidates:
                cid = cand.get("chunk_id")
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    new_docs.append(cand)

        logger.info(
            "[TargetedRetrieval] Retrieved %d new unique passages for missing requirements %s",
            len(new_docs), missing_ids
        )

        merged_docs = existing_docs + new_docs

        # Re-build EvidenceItem list
        merged_evidence = []
        for idx, d in enumerate(merged_docs, start=1):
            meta = d.get("metadata", {})
            score_val = d.get("score") or meta.get("score") or 0.8
            merged_evidence.append({
                "id": f"E{idx}",
                "content": d["content"],
                "source": meta.get("source", d.get("source", "unknown")),
                "page": meta.get("page", d.get("page")),
                "section": meta.get("section_heading", d.get("section_heading", "Unknown")),
                "chunk_id": d.get("chunk_id", "unknown"),
                "modality": meta.get("modality", d.get("modality", "text")),
                "retrieval_method": meta.get("retrieval_method", "targeted"),
                "retrieval_score": float(score_val),
                "rerank_score": float(score_val),
            })

        log_entry["status"] = "completed"
        log_entry["details"].update({
            "new_passages": len(new_docs),
            "total_passages": len(merged_docs),
        })

        return {
            **state,
            "retrieved_docs": merged_docs,
            "retrieved_evidence": merged_evidence,
            "targeted_queries": (state.get("targeted_queries") or []) + targeted_queries,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    return targeted_retrieval_node
