"""
pipeline/nodes/contradiction_detector.py
Contradiction Detector Node.

Detects factual conflicts between evidence passages or generated claims.

Algorithm:
  1. Compare retrieved evidence passages and claims for direct factual contradictions
  2. Differentiate true contradictions from distinct scope/version conditions
  3. Record structured Contradiction objects in state["contradictions"]
"""

import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.settings import USE_CONTRADICTION_DETECTION
from pipeline.state import ACRagState, Contradiction
from utils.llm_factory import get_llm

logger = logging.getLogger(__name__)


class ContradictionSpec(BaseModel):
    id: str = Field(description="Contradiction ID, e.g. 'X1'")
    claim_id_1: str = Field(default="", description="ID of first claim involved, if applicable")
    claim_id_2: str = Field(default="", description="ID of second claim involved, if applicable")
    evidence_id_1: str = Field(description="ID of first evidence passage")
    evidence_id_2: str = Field(description="ID of second evidence passage")
    conflict_description: str = Field(description="Description of the conflicting factual propositions")
    resolution_status: str = Field(
        default="unresolved",
        description="Status: 'unresolved', 'resolved_by_version', or 'resolved_by_authority'"
    )


class ContradictionDetectionOutput(BaseModel):
    has_contradictions: bool = Field(description="True if genuine factual conflicts exist")
    contradictions: List[ContradictionSpec] = Field(
        default_factory=list,
        description="List of detected factual contradictions"
    )


_CONTRADICTION_SYSTEM = """You are a contradiction detection expert for a research QA system.
Analyse candidate claims and evidence passages for genuine factual conflicts.

Rules:
1. Do NOT flag two statements as contradictory merely because they describe different conditions, scopes, or document versions.
2. Flag statements as CONTRADICTORY only if they assert mutually exclusive facts for the exact same entity, parameter, or scenario.
3. Check document authority or date metadata if available to resolve conflicts when possible.
4. If unresolved, describe the conflict clearly."""

_CONTRADICTION_HUMAN = """Evidence Passages:
{evidence_block}

---

Claims:
{claims_block}

Detect any factual contradictions."""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _CONTRADICTION_SYSTEM),
    ("human", _CONTRADICTION_HUMAN),
])


def contradiction_detector_node(state: ACRagState) -> ACRagState:
    """
    LangGraph node: Contradiction Detector.
    Reads:  state["claims"], state["retrieved_evidence"], state["scored_docs"]
    Writes: state["contradictions"]
    """
    claims = state.get("claims") or []
    docs = state.get("scored_docs") or state.get("retrieved_docs") or []
    evidence_items = state.get("retrieved_evidence") or []

    log_entry: Dict[str, Any] = {
        "stage": "contradiction_detector",
        "status": "started",
        "details": {"docs_count": len(docs)},
    }

    if len(docs) < 2 or not USE_CONTRADICTION_DETECTION:
        log_entry["status"] = "completed"
        log_entry["details"]["skipped"] = True
        return {
            **state,
            "contradictions": [],
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    # Format evidence & claims
    evidence_lines = []
    if evidence_items:
        for e in evidence_items[:6]:
            evidence_lines.append(f"[{e.get('id', '?')}] source={e.get('source', '?')}\n{e.get('content', '')}")
    else:
        for idx, d in enumerate(docs[:6], start=1):
            evidence_lines.append(f"[E{idx}] source={d.get('source', '?')}\n{d.get('content', '')}")

    evidence_block = "\n\n".join(evidence_lines)
    claims_block = "\n".join([f"{c['id']}: {c['text']}" for c in claims if isinstance(c, dict)])

    try:
        chain = _prompt | get_llm().with_structured_output(ContradictionDetectionOutput)
        result: ContradictionDetectionOutput = chain.invoke({
            "evidence_block": evidence_block,
            "claims_block": claims_block or "None",
        })

        contradiction_dicts: List[Contradiction] = []
        if result.has_contradictions and result.contradictions:
            for idx, item in enumerate(result.contradictions, start=1):
                contradiction_dicts.append({
                    "id": item.id or f"X{idx}",
                    "claim_id_1": item.claim_id_1 or None,
                    "claim_id_2": item.claim_id_2 or None,
                    "evidence_id_1": item.evidence_id_1 or "E1",
                    "evidence_id_2": item.evidence_id_2 or "E2",
                    "conflict_description": item.conflict_description,
                    "resolution_status": item.resolution_status,
                })

        logger.info("[ContradictionDetector] Detected %d contradictions", len(contradiction_dicts))

        log_entry["status"] = "completed"
        log_entry["details"]["num_contradictions"] = len(contradiction_dicts)

        return {
            **state,
            "contradictions": contradiction_dicts,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    except Exception as e:
        logger.warning("[ContradictionDetector] Exception during detection: %s", e)
        log_entry["status"] = "failed"
        log_entry["details"]["error"] = str(e)
        return {
            **state,
            "contradictions": [],
            "stage_logs": state["stage_logs"] + [log_entry],
        }
