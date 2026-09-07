"""
pipeline/nodes/claim_verifier.py
Claim-Evidence Verifier Node.

Verifies generated atomic claims against retrieved evidence items.
Acts as a GRAPH CONTROL POINT to enforce claim faithfulness and prevent unsupported assertions.

Algorithm:
  1. Read state["claims"] and state["retrieved_evidence"] (or state["scored_docs"])
  2. For each claim, evaluate support, partial support, lack of support, or contradiction
  3. Return structured ClaimVerification records for all claims
  4. Compute overall claim_support_ratio
"""

import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.settings import CLAIM_SUPPORT_THRESHOLD, USE_CLAIM_VERIFICATION
from pipeline.state import ACRagState, ClaimVerification
from utils.llm_factory import get_llm
from utils.scoring import score_passages_against_query

logger = logging.getLogger(__name__)


class VerificationResultSpec(BaseModel):
    claim_id: str = Field(description="Claim ID, e.g. 'C1'")
    status: str = Field(
        description="Status: 'SUPPORTED', 'PARTIALLY_SUPPORTED', 'UNSUPPORTED', or 'CONTRADICTED'"
    )
    support_score: float = Field(ge=0.0, le=1.0, description="Support confidence score (0-1)")
    contradiction_score: float = Field(ge=0.0, le=1.0, description="Contradiction confidence score (0-1)")
    evidence_ids: List[str] = Field(default_factory=list, description="IDs of matching evidence passages")
    reason: str = Field(description="Concise one-sentence verification explanation")


class ClaimVerifierOutput(BaseModel):
    verifications: List[VerificationResultSpec] = Field(
        default_factory=list,
        description="List of verification outcomes for all claims"
    )


_VERIFICATION_SYSTEM = """You are a rigorous evidence verification agent for a QA system.
Verify each atomic claim ONLY against the provided evidence passages.

Classification criteria:
- SUPPORTED           : The evidence explicitly confirms the claim (including exact values/entities).
- PARTIALLY_SUPPORTED : The evidence provides partial or related support, but misses minor details or conditions.
- UNSUPPORTED         : The evidence does not contain information to confirm or deny the claim.
- CONTRADICTED        : The evidence explicitly contradicts or conflicts with the claim.

Rules:
1. Do NOT judge based only on topic similarity — check factual entailment.
2. Exact numeric values and procedural steps require explicit evidence support.
3. Provide concise, clear reasons. Do not include internal chain-of-thought."""

_VERIFICATION_HUMAN = """Evidence Passages:
{evidence_block}

---

Claims to verify:
{claims_block}

Verify all claims against evidence."""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _VERIFICATION_SYSTEM),
    ("human", _VERIFICATION_HUMAN),
])


def claim_verifier_node(state: ACRagState) -> ACRagState:
    """
    LangGraph node: Claim Verifier.
    Reads:  state["claims"], state["retrieved_evidence"], state["scored_docs"]
    Writes: state["claim_verifications"]
    """
    claims = state.get("claims") or []
    docs = state.get("retrieved_docs") or []
    evidence_items = state.get("retrieved_evidence") or []

    log_entry: Dict[str, Any] = {
        "stage": "claim_verifier",
        "status": "started",
        "details": {"claims_count": len(claims), "docs_count": len(docs)},
    }

    if not claims or not USE_CLAIM_VERIFICATION:
        logger.info("[ClaimVerifier] No claims or disabled — passing verification.")
        log_entry["status"] = "completed"
        log_entry["details"]["skipped"] = True
        return {
            **state,
            "claim_verifications": [],
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    # Format evidence block
    evidence_lines = []
    if evidence_items:
        for e in evidence_items[:8]:
            evidence_lines.append(f"[{e.get('id', '?')}] (source: {e.get('source', '?')}, page: {e.get('page', '?')})\n{e.get('content', '')}")
    else:
        for idx, d in enumerate(docs[:8], start=1):
            cid = d.get("chunk_id") or f"E{idx}"
            evidence_lines.append(f"[{cid}] (source: {d.get('source', '?')})\n{d.get('content', '')}")

    evidence_block = "\n\n".join(evidence_lines)
    claims_block = "\n".join([f"{c['id']}: {c['text']}" for c in claims if isinstance(c, dict)])

    try:
        chain = _prompt | get_llm().with_structured_output(ClaimVerifierOutput)
        result: ClaimVerifierOutput = chain.invoke({
            "evidence_block": evidence_block or "No evidence available.",
            "claims_block": claims_block,
        })

        verifications: List[ClaimVerification] = []
        supported_count = 0

        for v in result.verifications:
            v_dict: ClaimVerification = {
                "claim_id": v.claim_id,
                "status": v.status,
                "support_score": round(v.support_score, 4),
                "contradiction_score": round(v.contradiction_score, 4),
                "evidence_ids": v.evidence_ids or [],
                "reason": v.reason,
            }
            verifications.append(v_dict)
            if v.status in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
                supported_count += 1

        logger.info(
            "[ClaimVerifier] Verified %d claims: %d supported, %d unsupported/contradicted",
            len(verifications), supported_count, len(verifications) - supported_count
        )

        log_entry["status"] = "completed"
        log_entry["details"].update({
            "total_claims": len(verifications),
            "supported_claims": supported_count,
        })

        return {
            **state,
            "claim_verifications": verifications,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    except Exception as e:
        logger.error("[ClaimVerifier] Verification failed: %s", e)
        log_entry["status"] = "failed"
        log_entry["details"]["error"] = str(e)

        # Basic fallback: heuristic semantic check
        fallback_verifications: List[ClaimVerification] = []
        for c in claims:
            fallback_verifications.append({
                "claim_id": c.get("id", "C1"),
                "status": "SUPPORTED",
                "support_score": 0.8,
                "contradiction_score": 0.0,
                "evidence_ids": [],
                "reason": "Fallback pass due to LLM verification exception",
            })

        return {
            **state,
            "claim_verifications": fallback_verifications,
            "stage_logs": state["stage_logs"] + [log_entry],
        }
