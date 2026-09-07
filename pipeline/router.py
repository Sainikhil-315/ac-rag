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
from config.settings import MAX_RETRIES, MAX_RETRIEVAL_ROUNDS, USE_RETRIEVAL_PLANNER, USE_CRITIC

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
        return "query_analyzer"

    return "validator"


def route_after_validator(state: ACRagState) -> str:
    if state.get("error"):
        return "end_error"

    passed = state.get("validation_passed", False)
    missing_reqs = state.get("missing_requirements") or []
    retrieval_attempts = state.get("retrieval_attempts", 1)
    retry = state.get("retry_count", 0)

    if passed:
        return "context_refiner"

    if retry >= MAX_RETRIES:
        logger.warning("[Router] validator coverage failed, max retries hit → end_max_retries")
        return "end_max_retries"

    # If missing requirements exist and we have retrieval budget left, run targeted retrieval
    if missing_reqs and retrieval_attempts < MAX_RETRIEVAL_ROUNDS:
        logger.info("[Router] validator coverage incomplete (missing %d reqs, attempt %d) → targeted_retrieval", len(missing_reqs), retrieval_attempts)
        return "targeted_retrieval"

    logger.info("[Router] validator failed (retry %d) → retriever", retry + 1)
    return "retriever"



def route_after_targeted_retrieval(state: ACRagState) -> str:
    """Targeted retrieval feeds back to validator for evidence coverage re-assessment."""
    return "validator"


def route_after_context_refiner(state: ACRagState) -> str:
    return "generator"


def route_after_generator(state: ACRagState) -> str:
    if state.get("error"):
        return "end_error"
    return "claim_extractor"


def route_after_claim_extractor(state: ACRagState) -> str:
    if state.get("error"):
        return "end_error"
    return "claim_verifier"


def route_after_claim_verifier(state: ACRagState) -> str:
    if state.get("error"):
        return "end_error"
    return "contradiction_detector"


def route_after_contradiction_detector(state: ACRagState) -> str:
    if state.get("error"):
        return "end_error"
    if USE_CRITIC:
        return "critic"
    return "end_success"


def route_after_critic(state: ACRagState) -> str:
    passed = state.get("critic_passed", False)
    abstained = state.get("abstained", False)

    if passed or abstained:
        logger.info("[Router] critic completed (passed=%s, abstained=%s) → end_success", passed, abstained)
        return "end_success"

    retry = state.get("retry_count", 0)
    if retry >= MAX_RETRIES:
        logger.warning("[Router] critic failed, max retries hit → end_max_retries")
        return "end_max_retries"

    reason = state.get("retry_reason", "content")
    if reason == "format":
        logger.info("[Router] critic format issue (retry %d) → generator", retry + 1)
        return "generator"
    elif reason in ("unsupported_claim", "coverage") and state.get("retrieval_attempts", 1) < MAX_RETRIEVAL_ROUNDS:
        logger.info("[Router] critic %s issue (retry %d) → targeted_retrieval", reason, retry + 1)
        return "targeted_retrieval"

    logger.info("[Router] critic content issue (retry %d) → query_analyzer", retry + 1)
    return "query_analyzer"

