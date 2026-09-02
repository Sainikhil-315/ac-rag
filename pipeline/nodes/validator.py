"""
pipeline/nodes/validator.py
Validator Node — Evidence Validation (full implementation).

Algorithm:
  1. Embed query and all retrieved passages
  2. Compute raw cosine similarity (query ↔ each passage) — NOT the same scale as
     the Entry Router's FAISS L2-distance-based score, see config/settings.py
  3. Attach score to each passage
  4. Discard passages with score < threshold (0.20 for normal queries; relaxed to
     0.05 for intent="summarization" — see config/settings.py for why: a broad
     "what is this document about?" query has structurally low similarity to any
     ONE passage, which isn't the same thing as that passage being irrelevant)
  5. If surviving passages < min_valid_passages → set validation_passed=False
     (graph router will retry retrieval, widening k each time — see retriever.py).
     min_valid_passages is also relaxed to 1 for summarization intent — a document
     that chunks into a single passage (any short document) can never satisfy a
     floor of 2, regardless of how relevant that one passage is.
  6. Log score distribution for research analysis

Ablation:
  USE_VALIDATOR=False → all docs pass with score=1.0 (set in settings.py)
"""

import logging
from typing import Any, Dict, List

from config.settings import (
    EVIDENCE_SCORE_THRESHOLD,
    MIN_VALID_PASSAGES,
    SUMMARIZATION_SCORE_THRESHOLD,
    SUMMARIZATION_MIN_VALID_PASSAGES,
    USE_VALIDATOR,
)
from pipeline.state import ACRagState
from utils.scoring import score_passages_against_query

logger = logging.getLogger(__name__)


def _score_and_filter(
    query: str,
    docs: List[Dict[str, Any]],
    score_threshold: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Score all docs and filter below threshold.
    Returns (filtered_docs, stats_dict).
    """
    texts = [doc["content"] for doc in docs]
    scores = score_passages_against_query(query, texts)

    # Attach scores
    scored = [{**doc, "score": score} for doc, score in zip(docs, scores)]

    # Log individual scores for research traceability
    for doc in scored:
        logger.debug(
            "[Validator] chunk=%s score=%.4f section=%s",
            doc.get("chunk_id", "?"), doc["score"], doc.get("section_heading", "?")
        )

    # Filter
    passed = [doc for doc in scored if doc["score"] >= score_threshold]
    failed = [doc for doc in scored if doc["score"] < score_threshold]

    stats = {
        "total": len(scored),
        "passed": len(passed),
        "failed": len(failed),
        "threshold": score_threshold,
        "score_min": round(min(scores), 4) if scores else 0.0,
        "score_max": round(max(scores), 4) if scores else 0.0,
        "score_mean": round(sum(scores) / len(scores), 4) if scores else 0.0,
    }

    return passed, stats


def validator_node(state: ACRagState) -> ACRagState:
    """
    LangGraph node: Validator (full implementation).
    Reads:  state["query"], state["retrieved_docs"]
    Writes: state["scored_docs"], state["validation_passed"]
    """
    query = state["query"]
    docs = state.get("retrieved_docs") or []

    log_entry: Dict[str, Any] = {
        "stage": "validator",
        "status": "started",
        "details": {"docs_in": len(docs)},
    }

    if not docs:
        logger.warning("[Validator] No documents to validate.")
        log_entry["status"] = "completed"
        log_entry["details"]["validation_passed"] = False
        log_entry["details"]["reason"] = "empty_docs"
        return {
            **state,
            "scored_docs": [],
            "validation_passed": False,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    # ── Ablation bypass ───────────────────────────────────────────────────────
    if not USE_VALIDATOR:
        logger.info("[Validator] Ablation: validator disabled — passing all %d docs", len(docs))
        scored = [{**doc, "score": 1.0} for doc in docs]
        log_entry["status"] = "completed"
        log_entry["details"].update({"validation_passed": True, "ablation_skip": True})
        return {
            **state,
            "scored_docs": scored,
            "validation_passed": True,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    # ── Score and filter ──────────────────────────────────────────────────────
    # Summarization queries get a relaxed bar — see module docstring / config/settings.py.
    is_summarization = state.get("intent") == "summarization"
    score_threshold = SUMMARIZATION_SCORE_THRESHOLD if is_summarization else EVIDENCE_SCORE_THRESHOLD
    min_passages = SUMMARIZATION_MIN_VALID_PASSAGES if is_summarization else MIN_VALID_PASSAGES

    try:
        passed_docs, stats = _score_and_filter(query, docs, score_threshold)

        validation_passed = len(passed_docs) >= min_passages

        logger.info(
            "[Validator] %d/%d passages passed (threshold=%.2f, intent=%s). "
            "scores: min=%.3f max=%.3f mean=%.3f. valid=%s",
            stats["passed"], stats["total"], score_threshold,
            state.get("intent"),
            stats["score_min"], stats["score_max"], stats["score_mean"],
            validation_passed,
        )

        if not validation_passed:
            logger.warning(
                "[Validator] Only %d passage(s) passed minimum threshold %d — will retry retrieval.",
                len(passed_docs), min_passages,
            )

        log_entry["status"] = "completed"
        log_entry["details"].update({**stats, "validation_passed": validation_passed})

        return {
            **state,
            "scored_docs": passed_docs,
            "validation_passed": validation_passed,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    except Exception as e:
        logger.error("[Validator] Scoring failed: %s", e)
        log_entry["status"] = "failed"
        log_entry["details"]["error"] = str(e)
        # On scoring failure, pass docs through unscored rather than crashing pipeline
        scored_fallback = [{**doc, "score": 0.0} for doc in docs]
        return {
            **state,
            "scored_docs": scored_fallback,
            "validation_passed": False,
            "error": f"Validator scoring failed: {e}",
            "stage_logs": state["stage_logs"] + [log_entry],
        }
