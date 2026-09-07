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
    Quick similarity check against vector store + BM25/lexical pre-check.
    Dense similarity acts as a ROUTING PRIOR, not definitive evidence sufficiency.
    Queries with low embedding similarity but exact lexical match route to RAG.
    """
    try:
        results: List[Tuple] = vsm.similarity_search_with_score(query, k=3)
        top_score = results[0][1] if results else 0.0
        avg_score = (sum(score for _, score in results) / len(results)) if results else 0.0

        logger.info(
            "[EntryRouter] Similarity → top=%.3f avg=%.3f threshold=%.3f",
            top_score, avg_score, ROUTER_SIMILARITY_THRESHOLD,
        )

        if top_score >= ROUTER_SIMILARITY_THRESHOLD and avg_score >= (ROUTER_SIMILARITY_THRESHOLD * 0.7):
            return "rag"

        # Dense similarity was low — check BM25/lexical signals before rejecting
        bm25_results = vsm.bm25_search(query, k=1)
        if bm25_results:
            top_bm25_score = float(bm25_results[0].metadata.get("score", 0.0))
            if top_bm25_score > 0.0:
                logger.info(
                    "[EntryRouter] Low dense score (%.3f), but BM25 matched with score %.3f → routing to RAG",
                    top_score, top_bm25_score
                )
                return "rag"

        # Check for exact token identifiers, uppercase constants, numbers, or factual intent words
        import re
        has_uppercase_identifier = bool(re.search(r'\b[A-Z0-9_]{3,}\b', query))
        has_numeric = bool(re.search(r'\b\d+\b', query))
        is_factual_intent = any(q_word in query.lower() for q_word in [
            "what", "how", "where", "which", "when", "who", "define", "explain", "value", "config", "setting"
        ])

        if (has_uppercase_identifier or has_numeric or is_factual_intent) and not _is_chitchat(query):
            logger.info("[EntryRouter] Factual/lexical query structure detected → routing to RAG")
            return "rag"

        return "unknown"

    except Exception as e:
        logger.warning("[EntryRouter] Similarity check failed (%s) → defaulting to rag", e)
        return "rag"


def _is_chitchat(query: str) -> bool:
    clean_q = query.lower().strip("?!. ")
    chitchat_phrases = {"hi", "hello", "hey", "who are you", "what is your name", "how are you", "good morning", "good evening"}
    return clean_q in chitchat_phrases


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
