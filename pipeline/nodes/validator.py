"""
pipeline/nodes/validator.py
Evidence Coverage Engine Node.

Rebuilds standard threshold validator into a requirement-level Evidence Coverage Engine.

Responsibilities:
  1. Map retrieved evidence to evidence_requirements (R1, R2...)
  2. Evaluate requirement-level support scores using semantic, lexical, and modality matching
  3. Assign requirement statuses: SUPPORTED, PARTIAL, MISSING, CONTRADICTED
  4. Calculate weighted evidence coverage score
  5. Identify missing requirements needing targeted retrieval
  6. Determine validation_passed based on requirement coverage thresholds
"""

import logging
from typing import Any, Dict, List, Tuple

from config.settings import (
    EVIDENCE_SCORE_THRESHOLD,
    MIN_VALID_PASSAGES,
    COVERAGE_THRESHOLD,
    CRITICAL_COVERAGE_THRESHOLD,
    SUMMARIZATION_SCORE_THRESHOLD,
    SUMMARIZATION_MIN_VALID_PASSAGES,
    USE_VALIDATOR,
    USE_EVIDENCE_COVERAGE,
)
from pipeline.state import ACRagState, EvidenceCoverage, EvidenceRequirement
from utils.scoring import score_passages_against_query

logger = logging.getLogger(__name__)

_PRIORITY_WEIGHTS = {
    "critical": 1.0,
    "high": 0.8,
    "medium": 0.5,
    "low": 0.3,
}


def _evaluate_requirement_support(
    req: Dict[str, Any],
    docs: List[Dict[str, Any]],
    query: str,
    score_threshold: float = EVIDENCE_SCORE_THRESHOLD,
) -> Tuple[float, str, List[str], str]:
    """
    Evaluate support strength of candidate docs for a single evidence requirement.
    """
    if not docs:
        return 0.0, "MISSING", [], "No documents retrieved"

    req_text = req.get("requirement") or query
    expected_modality = req.get("expected_modality", "text")
    keywords = [k.lower() for k in req.get("keywords", []) if k]

    texts = [doc.get("content", "") for doc in docs]
    semantic_scores = score_passages_against_query(req_text, texts)

    best_score = 0.0
    best_doc_id = None
    supporting_ids = []

    for idx, (doc, sem_score) in enumerate(zip(docs, semantic_scores)):
        doc_id = doc.get("chunk_id") or f"E{idx+1}"
        content_lower = doc.get("content", "").lower()
        doc_modality = doc.get("modality", "text")

        lexical_match = 0.0
        if keywords:
            matched_kws = sum(1 for kw in keywords if kw in content_lower)
            lexical_match = matched_kws / len(keywords)

        modality_match = 1.0
        if expected_modality not in ("all", None):
            if doc_modality == expected_modality:
                modality_match = 1.1
            else:
                modality_match = 0.8

        combined = (0.7 * sem_score + 0.3 * lexical_match) * modality_match
        combined = max(0.0, min(1.0, combined))

        if sem_score >= score_threshold or combined >= 0.25:
            supporting_ids.append(doc_id)

        if sem_score > best_score or combined > best_score:
            best_score = max(sem_score, combined)
            best_doc_id = doc_id

    if best_score >= max(0.20, score_threshold):
        status = "SUPPORTED"
        reason = ""
    elif best_score >= (score_threshold * 0.5):
        status = "PARTIAL"
        reason = "Partial relevance; missing specific details"
    else:
        status = "MISSING"
        reason = "No passage satisfied requirement with high confidence"

    return best_score, status, supporting_ids, reason


def validator_node(state: ACRagState) -> ACRagState:
    """
    LangGraph node: Evidence Coverage Engine.
    Reads:  state["query"], state["retrieved_docs"], state["evidence_requirements"]
    Writes: state["scored_docs"], state["evidence_coverage"], state["missing_requirements"],
            state["validation_passed"]
    """
    query = state["query"]
    docs = state.get("retrieved_docs") or []
    requirements = state.get("evidence_requirements") or []

    log_entry: Dict[str, Any] = {
        "stage": "validator",
        "status": "started",
        "details": {"docs_in": len(docs), "requirements_in": len(requirements)},
    }

    if not docs:
        logger.warning("[Validator] No documents to validate.")
        missing_ids = [r["id"] for r in requirements] if requirements else ["R1"]
        log_entry["status"] = "completed"
        log_entry["details"]["validation_passed"] = False
        return {
            **state,
            "scored_docs": [],
            "evidence_coverage": [],
            "missing_requirements": missing_ids,
            "validation_passed": False,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    # Ablation bypass
    if not USE_VALIDATOR:
        logger.info("[Validator] Ablation: validator disabled — passing all %d docs", len(docs))
        scored = [{**doc, "score": 1.0} for doc in docs]
        return {
            **state,
            "scored_docs": scored,
            "missing_requirements": [],
            "validation_passed": True,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    is_summarization = state.get("intent") == "summarization"
    score_threshold = SUMMARIZATION_SCORE_THRESHOLD if is_summarization else EVIDENCE_SCORE_THRESHOLD
    min_passages = SUMMARIZATION_MIN_VALID_PASSAGES if is_summarization else MIN_VALID_PASSAGES

    try:
        # Ensure baseline requirement exists
        if not requirements:
            requirements = [{
                "id": "R1",
                "requirement": state.get("rewritten_query") or query,
                "requirement_type": "fact",
                "priority": "critical",
                "keywords": [query],
                "expected_modality": "all",
            }]

        coverage_list: List[EvidenceCoverage] = []
        updated_requirements: List[EvidenceRequirement] = []
        missing_req_ids: List[str] = []

        total_weight = 0.0
        weighted_score_sum = 0.0
        critical_passed = True

        for req in requirements:
            score, status, sup_ids, reason = _evaluate_requirement_support(req, docs, query, score_threshold)
            priority = req.get("priority", "critical")
            weight = _PRIORITY_WEIGHTS.get(priority, 0.5)

            total_weight += weight
            weighted_score_sum += weight * score

            if priority == "critical" and score < score_threshold:
                critical_passed = False

            if status in ("MISSING", "PARTIAL") or score < score_threshold:
                missing_req_ids.append(req["id"])

            updated_req = {
                **req,
                "status": status,
                "support_score": round(score, 4),
                "supporting_evidence_ids": sup_ids,
            }
            updated_requirements.append(updated_req)

            coverage_list.append({
                "requirement_id": req["id"],
                "status": status,
                "score": round(score, 4),
                "supporting_evidence_ids": sup_ids,
                "missing_reason": reason if status != "SUPPORTED" else None,
            })

        total_coverage = weighted_score_sum / (total_weight if total_weight > 0 else 1.0)
        target_thresh = 0.05 if is_summarization else COVERAGE_THRESHOLD

        # Evaluate score threshold passing docs count
        texts = [d["content"] for d in docs]
        raw_scores = score_passages_against_query(query, texts)
        scored_docs = [{**d, "score": s} for d, s in zip(docs, raw_scores)]
        passing_doc_count = sum(1 for s in raw_scores if s >= score_threshold)

        validation_passed = (total_coverage >= target_thresh or passing_doc_count >= min_passages) and (critical_passed or passing_doc_count >= min_passages)

        logger.info(
            "[EvidenceCoverageEngine] Coverage: %.2f (thresh: %.2f, passing_docs: %d/%d) | valid=%s",
            total_coverage, target_thresh, passing_doc_count, min_passages, validation_passed
        )

        log_entry["status"] = "completed"
        log_entry["details"].update({
            "total_coverage": round(total_coverage, 4),
            "validation_passed": validation_passed,
            "missing_req_ids": missing_req_ids,
        })

        return {
            **state,
            "scored_docs": scored_docs,
            "evidence_requirements": updated_requirements,
            "evidence_coverage": coverage_list,
            "missing_requirements": missing_req_ids,
            "validation_passed": validation_passed,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    except Exception as e:
        logger.error("[Validator] Scoring failed: %s", e)
        log_entry["status"] = "failed"
        log_entry["details"]["error"] = str(e)
        scored_fallback = [{**doc, "score": 0.0} for doc in docs]
        return {
            **state,
            "scored_docs": scored_fallback,
            "validation_passed": False,
            "error": f"Validator scoring failed: {e}",
            "stage_logs": state["stage_logs"] + [log_entry],
        }


