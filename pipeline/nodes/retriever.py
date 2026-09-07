"""
pipeline/nodes/retriever.py
Retriever Node — executes retrieval according to the RetrievalPlan.

Responsibilities:
  1. Single-query MMR retrieval (standard path)
  2. Multi-query retrieval: run each sub-query, merge, deduplicate (complex path)
  3. Modality filtering: keep only chunks matching the planned modality
  4. Serialise Document objects to dicts for state storage (LangGraph state must be JSON-serialisable)

Design:
  - Uses VectorStoreManager (loaded once and passed in via dependency injection,
    not instantiated per query — avoids reloading the FAISS index on every call).
  - Deduplication by chunk_id ensures no duplicate passages even when
    multiple sub-queries retrieve overlapping chunks.
  - Retrieved docs stored as dicts (not Document objects) for clean state serialisation.
"""

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document

from config.settings import RETRIEVAL_K_DEFAULT, RETRIEVAL_K_MAX
from pipeline.state import ACRagState, RetrievalPlan
from vectorstore.store import VectorStoreManager

logger = logging.getLogger(__name__)


def _escalate_plan_for_retry(plan: RetrievalPlan, retry_count: int) -> RetrievalPlan:
    """
    Widen the search on a retry instead of repeating the exact same MMR call.

    Bug this fixes: when the Validator rejects a retrieval, the graph routes
    straight back to this node with the SAME query and SAME plan. MMR search
    is deterministic for identical inputs, so without this escalation every
    retry returns the identical docs/scores and the pipeline always exhausts
    MAX_RETRIES on validator failures — confirmed empirically (evaluation run
    2026-08-19: 7/15 test queries failed this exact way, see
    GAPS_AND_IMPROVEMENTS.md item A.6).

    Each retry widens k/fetch_k by 50% (capped at RETRIEVAL_K_MAX) and lowers
    lambda_mult (more diversity, less pure-relevance clustering) so a retry
    actually has a chance of surfacing different passages.
    """
    if retry_count <= 0:
        return plan

    widened_k = min(RETRIEVAL_K_MAX, round(plan["k"] * (1 + 0.5 * retry_count)))
    widened_fetch_k = min(RETRIEVAL_K_MAX * 3, round(plan["fetch_k"] * (1 + 0.5 * retry_count)))
    widened_lambda = max(0.1, plan["lambda_mult"] - 0.15 * retry_count)

    return {
        **plan,
        "k": widened_k,
        "fetch_k": widened_fetch_k,
        "lambda_mult": round(widened_lambda, 2),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _doc_to_dict(doc: Document, query: str) -> Dict[str, Any]:
    """Serialise a LangChain Document to a plain dict for state storage."""
    return {
        "content": doc.page_content,
        "metadata": doc.metadata,
        "chunk_id": doc.metadata.get("chunk_id", "unknown"),
        "source": doc.metadata.get("source", "unknown"),
        "page": doc.metadata.get("page", None),
        "section_heading": doc.metadata.get("section_heading", "Unknown"),
        "modality": doc.metadata.get("modality", "text"),
        "retrieved_by_query": query,   # attribution: which query found this chunk
        "score": None,                 # filled by Validator node
    }


def _apply_modality_filter(docs: List[Dict], modality: str) -> List[Dict]:
    """Filter docs to the requested modality. 'all' skips filtering."""
    if modality == "all":
        return docs
    filtered = [d for d in docs if d["modality"] == modality]
    if not filtered:
        logger.warning(
            "[Retriever] Modality filter '%s' removed all docs. Falling back to all modalities.",
            modality
        )
        return docs   # fallback: return unfiltered to avoid empty context
    return filtered


def _deduplicate(docs: List[Dict]) -> List[Dict]:
    """Remove duplicate chunks by chunk_id, preserving first occurrence."""
    seen = set()
    unique = []
    for doc in docs:
        cid = doc["chunk_id"]
        if cid not in seen:
            seen.add(cid)
            unique.append(doc)
    return unique


# ── Core retrieval ────────────────────────────────────────────────────────────

def _retrieve_single(
    vsm: VectorStoreManager,
    query: str,
    plan: RetrievalPlan,
) -> List[Dict]:
    """Run search for a single query string using planned strategy (hybrid, bm25, or dense)."""
    strategy = plan.get("strategy", "hybrid")
    k = plan["k"]
    modality = plan.get("modality_filter", "all")

    if strategy == "bm25":
        docs = vsm.bm25_search(query=query, k=k, modality_filter=modality)
    elif strategy == "hybrid":
        l_weight = plan.get("lexical_weight", 0.5)
        d_weight = plan.get("dense_weight", 0.5)
        docs = vsm.hybrid_search(
            query=query,
            k=k,
            fetch_k=plan["fetch_k"],
            lambda_mult=plan["lambda_mult"],
            modality_filter=modality,
            dense_weight=d_weight,
            bm25_weight=l_weight,
        )
    else:
        docs = vsm.mmr_search(
            query=query,
            k=k,
            fetch_k=plan["fetch_k"],
            lambda_mult=plan["lambda_mult"],
        )

    return [_doc_to_dict(doc, query) for doc in docs]


def _retrieve_multi(
    vsm: VectorStoreManager,
    queries: List[str],
    plan: RetrievalPlan,
) -> List[Dict]:
    """
    Run search for each sub-query, merge results, deduplicate.
    k is distributed across sub-queries (each gets k//n, min 2).
    Final list is deduplicated and capped at plan["k"].
    """
    n = len(queries)
    per_query_k = max(2, plan["k"] // n)
    sub_plan = {**plan, "k": per_query_k, "fetch_k": per_query_k * 3}

    all_docs: List[Dict] = []
    for q in queries:
        results = _retrieve_single(vsm, q, sub_plan)
        all_docs.extend(results)
        logger.debug("[Retriever] Sub-query '%s' → %d docs", q[:60], len(results))

    unique = _deduplicate(all_docs)
    return unique[: plan["k"]]


# ── Node function ─────────────────────────────────────────────────────────────

def make_retriever_node(vsm: VectorStoreManager):
    """
    Factory that binds a loaded VectorStoreManager to the retriever node.
    """

    def retriever_node(state: ACRagState) -> ACRagState:
        """
        LangGraph node: Retriever.
        Reads:  state["query"], state["decomposed_queries"], state["retrieval_plan"]
        Writes: state["retrieved_docs"], state["retrieved_evidence"], state["retrieval_attempts"]
        """
        query = state.get("rewritten_query") or state["query"]
        plan: RetrievalPlan = state.get("retrieval_plan") or _default_plan()
        sub_queries: List[str] = state.get("decomposed_queries") or []

        retrieval_attempts = state.get("retrieval_attempts", 0) + 1
        retry_count = state.get("retry_count", 0)
        original_k = plan["k"]
        plan = _escalate_plan_for_retry(plan, retry_count)
        if plan["k"] != original_k:
            logger.info(
                "[Retriever] Retry #%d — widening search: k=%d fetch_k=%d lambda_mult=%.2f",
                retry_count, plan["k"], plan["fetch_k"], plan["lambda_mult"],
            )

        log_entry = {
            "stage": "retriever",
            "status": "started",
            "details": {
                "query": query,
                "k": plan["k"],
                "strategy": plan.get("strategy", "hybrid"),
                "modality": plan["modality_filter"],
                "multi_query": plan["use_multi_query"],
                "num_sub_queries": len(sub_queries),
                "retry_count": retry_count,
            },
        }
        logger.info(
            "[Retriever] k=%d | strategy=%s | modality=%s | multi_query=%s",
            plan["k"], plan.get("strategy", "hybrid"), plan["modality_filter"], plan["use_multi_query"]
        )

        try:
            # Choose retrieval strategy
            if plan["use_multi_query"] and sub_queries:
                raw_docs = _retrieve_multi(vsm, sub_queries, plan)
                logger.info("[Retriever] Multi-query retrieval: %d unique docs", len(raw_docs))
            else:
                raw_docs = _retrieve_single(vsm, query, plan)
                logger.info("[Retriever] Single-query retrieval: %d docs", len(raw_docs))

            # Apply modality filter
            filtered = _apply_modality_filter(raw_docs, plan["modality_filter"])

            logger.info("[Retriever] After modality filter: %d docs", len(filtered))

            # Build EvidenceItem list for state contract
            evidence_items = []
            for idx, d in enumerate(filtered, start=1):
                meta = d.get("metadata", {})
                score_val = meta.get("score") or d.get("score") or 0.8
                evidence_items.append({
                    "id": f"E{idx}",
                    "content": d["content"],
                    "source": meta.get("source", d.get("source", "unknown")),
                    "page": meta.get("page", d.get("page")),
                    "section": meta.get("section_heading", d.get("section_heading", "Unknown")),
                    "chunk_id": d.get("chunk_id", "unknown"),
                    "modality": meta.get("modality", d.get("modality", "text")),
                    "retrieval_method": meta.get("retrieval_method", plan.get("strategy", "hybrid")),
                    "retrieval_score": float(score_val),
                    "rerank_score": float(score_val),
                })

            log_entry["status"] = "completed"
            log_entry["details"]["docs_retrieved"] = len(filtered)

            return {
                **state,
                "retrieved_docs": filtered,
                "retrieved_evidence": evidence_items,
                "retrieval_attempts": retrieval_attempts,
                "stage_logs": state["stage_logs"] + [log_entry],
            }

        except Exception as e:
            logger.error("[Retriever] Failed: %s", e)
            log_entry["status"] = "failed"
            log_entry["details"]["error"] = str(e)
            return {
                **state,
                "error": f"Retriever failed: {e}",
                "stage_logs": state["stage_logs"] + [log_entry],
            }

    return retriever_node


def _default_plan() -> RetrievalPlan:
    """Fallback plan if planner was skipped (ablation mode)."""
    from config.settings import MMR_FETCH_K_MULTIPLIER, MMR_LAMBDA_MULT
    return RetrievalPlan(
        k=RETRIEVAL_K_DEFAULT,
        fetch_k=RETRIEVAL_K_DEFAULT * MMR_FETCH_K_MULTIPLIER,
        lambda_mult=MMR_LAMBDA_MULT,
        modality_filter="all",
        use_multi_query=False,
        retrieval_depth="standard",
        strategy="hybrid",
        lexical_weight=0.5,
        dense_weight=0.5,
    )

