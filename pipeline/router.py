"""
pipeline/router.py
All conditional edge functions for the LangGraph pipeline.

Each function receives the current state and returns a string node name
telling LangGraph which node to route to next.

Centralising routing logic here keeps graph.py clean and makes
routing decisions easy to test independently.

Route map:
  after_entry_router     → "rag" | "unknown"
  after_query_analyzer   → "retrieval_planner" | "end_error"
  after_retrieval_planner→ "retriever"
  after_retriever        → "validator"         | "end_error"
  after_validator        → "context_refiner"   | "retriever" (retry) | "end_max_retries"
  after_context_refiner  → "generator"
  after_generator        → "critic"
  after_critic           → "end_success"
                         | "query_analyzer"    (content retry)
                         | "generator"         (format retry)
                         | "end_max_retries"
"""

import logging
from pipeline.state import ACRagState
from config.settings import MAX_RETRIES, USE_RETRIEVAL_PLANNER, USE_CRITIC

logger = logging.getLogger(__name__)


def route_after_entry_router(state: ACRagState) -> str:
    route = state.get("route", "rag")
    logger.info("[Router] entry_router → %s", route)
    if route == "unknown":
        return "unknown"
    return "rag"


def route_after_max_retries(state: ACRagState) -> str:
    """RAG exhausted retries — terminate."""
    logger.warning("[Router] RAG max retries exhausted → end")
    return "end"


def route_after_query_analyzer(state: ACRagState) -> str:
    if state.get("error"):
        logger.warning("[Router] query_analyzer error → end_error")
        return "end_error"
    return "retrieval_planner"


def route_after_retrieval_planner(state: ACRagState) -> str:
    # Planner always feeds retriever (ablation is handled inside the node itself)
    return "retriever"


def route_after_retriever(state: ACRagState) -> str:
    if state.get("error"):
        logger.warning("[Router] retriever error → end_error")
        return "end_error"

    docs = state.get("retrieved_docs") or []
    if not docs:
        retry = state.get("retry_count", 0)
        if retry >= MAX_RETRIES:
            logger.warning("[Router] retriever returned 0 docs, max retries hit → end_max_retries")
            return "end_max_retries"
        logger.info("[Router] retriever returned 0 docs, routing back to query_analyzer (retry %d)", retry + 1)
        return "query_analyzer"   # re-analyse query before re-retrieving

    return "validator"


def route_after_validator(state: ACRagState) -> str:
    if state.get("error"):
        return "end_error"

    passed = state.get("validation_passed", False)
    if passed:
        # context_refiner_node handles the USE_CONTEXT_REFINER ablation bypass
        # internally (bug fix 2026-08-20 — routing around the node entirely used
        # to leave refined_context as None, see context_refiner.py for detail).
        return "context_refiner"

    # Validation failed — retry retrieval
    retry = state.get("retry_count", 0)
    if retry >= MAX_RETRIES:
        logger.warning("[Router] validation failed, max retries hit → end_max_retries")
        return "end_max_retries"

    logger.info("[Router] validation failed (retry %d) → retriever", retry + 1)
    return "retriever"


def route_after_context_refiner(state: ACRagState) -> str:
    return "generator"


def route_after_generator(state: ACRagState) -> str:
    if state.get("error"):
        return "end_error"
    if USE_CRITIC:
        return "critic"
    return "end_success"   # ablation: skip critic


def route_after_critic(state: ACRagState) -> str:
    passed = state.get("critic_passed", False)

    if passed:
        logger.info("[Router] critic passed → end_success")
        return "end_success"

    retry = state.get("retry_count", 0)
    if retry >= MAX_RETRIES:
        logger.warning("[Router] critic failed, max retries hit → end_max_retries")
        return "end_max_retries"

    reason = state.get("retry_reason", "content")
    if reason == "format":
        logger.info("[Router] critic format issue (retry %d) → generator", retry + 1)
        return "generator"

    # Default: content issue → full restart from query_analyzer
    logger.info("[Router] critic content issue (retry %d) → query_analyzer", retry + 1)
    return "query_analyzer"
