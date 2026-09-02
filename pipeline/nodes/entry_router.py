"""
pipeline/nodes/entry_router.py
Entry Router Node — routes queries based on vector store relevance.

If the query has sufficient similarity to indexed documents → "rag"
Otherwise → "unknown" (pipeline responds: "I don't know about that")
"""

import logging
from typing import List, Tuple

from pipeline.state import ACRagState
from vectorstore.store import VectorStoreManager
from config.settings import ROUTER_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


def _similarity_route(vsm: VectorStoreManager, query: str) -> str:
    """
    Quick k=3 similarity check against the vector store.
    Returns "rag" if the query is relevant to indexed documents, "unknown" otherwise.
    """
    try:
        results: List[Tuple] = vsm.similarity_search_with_score(query, k=3)
        if not results:
            logger.info("[EntryRouter] Similarity returned 0 results → unknown")
            return "unknown"

        avg_score = sum(score for _, score in results) / len(results)
        top_score = results[0][1]

        logger.info(
            "[EntryRouter] Similarity → top=%.3f avg=%.3f threshold=%.3f",
            top_score, avg_score, ROUTER_SIMILARITY_THRESHOLD,
        )

        if top_score >= ROUTER_SIMILARITY_THRESHOLD and avg_score >= (ROUTER_SIMILARITY_THRESHOLD * 0.7):
            return "rag"

        return "unknown"

    except Exception as e:
        logger.warning("[EntryRouter] Similarity check failed (%s) → defaulting to rag", e)
        return "rag"


def make_entry_router_node(vsm: VectorStoreManager):
    """
    Factory that binds the loaded VectorStoreManager to the entry router.
    """

    def entry_router_node(state: ACRagState) -> ACRagState:
        """
        LangGraph node: Entry Router.
        Reads:  state["query"]
        Writes: state["route"]  → "rag" | "unknown"
        """
        query = state["query"].strip()
        log_entry = {"stage": "entry_router", "status": "started", "details": {"query": query}}

        route = _similarity_route(vsm, query)
        logger.info("[EntryRouter] Query routed → %s", route)

        log_entry["status"] = "completed"
        log_entry["details"]["route"] = route

        return {
            **state,
            "route": route,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    return entry_router_node
