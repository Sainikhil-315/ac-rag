"""
pipeline/nodes/claim_extractor.py
Claim Extractor Node.

Extracts atomic factual claims from the generated answer.

Algorithm:
  1. Parse draft answer string
  2. Use LLM with structured output to break answer into atomic claims
  3. Map claims to evidence requirements and citations
  4. Write state["claims"]
"""

import logging
import re
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from pipeline.state import ACRagState, Claim
from utils.llm_factory import get_llm

logger = logging.getLogger(__name__)


class AtomicClaimSpec(BaseModel):
    id: str = Field(description="Claim ID, e.g. 'C1', 'C2'")
    text: str = Field(description="Single, independently verifiable factual proposition")
    claim_type: str = Field(
        default="factual",
        description="Type: 'factual', 'numeric', 'procedural', 'definitional', or 'comparative'"
    )
    requirement_ids: List[str] = Field(
        default_factory=list,
        description="IDs of requirements (e.g. 'R1') that this claim addresses"
    )
    citation_ids: List[str] = Field(
        default_factory=list,
        description="IDs of cited evidence items or source numbers"
    )


class ClaimExtractionOutput(BaseModel):
    claims: List[AtomicClaimSpec] = Field(
        default_factory=list,
        description="List of atomic factual claims extracted from the answer"
    )


_EXTRACTION_SYSTEM = """You are a factual claim extraction expert for a research QA system.
Given a user question, evidence requirements, and a candidate answer, convert the answer into atomic factual claims.

Rules:
1. Each claim must be a single, independently verifiable proposition.
2. Preserve exact numbers, units, entities, parameters, and qualifiers.
3. Assign stable IDs: C1, C2, C3, etc.
4. Map each claim to relevant requirement IDs (R1, R2, etc.).
5. If the answer states that evidence is insufficient or abstains from answering, return an empty list of claims.
6. Do NOT invent new facts not present in the answer."""

_EXTRACTION_HUMAN = """Question: {query}
Requirements: {requirements_summary}
Candidate Answer: {answer}

Extract atomic claims."""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _EXTRACTION_SYSTEM),
    ("human", _EXTRACTION_HUMAN),
])


def claim_extractor_node(state: ACRagState) -> ACRagState:
    """
    LangGraph node: Claim Extractor.
    Reads:  state["query"], state["answer"], state["evidence_requirements"]
    Writes: state["claims"]
    """
    answer = state.get("answer") or ""
    query = state["query"]
    reqs = state.get("evidence_requirements") or []

    log_entry: Dict[str, Any] = {
        "stage": "claim_extractor",
        "status": "started",
        "details": {"answer_chars": len(answer)},
    }

    # Check for abstention / insufficient evidence response
    if not answer or "not contain sufficient" in answer.lower() or "could not establish" in answer.lower():
        logger.info("[ClaimExtractor] Answer is an abstention/insufficient response. No claims extracted.")
        log_entry["status"] = "completed"
        log_entry["details"]["num_claims"] = 0
        return {
            **state,
            "claims": [],
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    try:
        req_summary = "; ".join([f"{r.get('id', 'R?')}: {r.get('requirement', '')}" for r in reqs if isinstance(r, dict)])
        chain = _prompt | get_llm().with_structured_output(ClaimExtractionOutput)

        result: ClaimExtractionOutput = chain.invoke({
            "query": query,
            "requirements_summary": req_summary or "R1: General factual answer",
            "answer": answer,
        })

        claim_dicts: List[Claim] = []
        for idx, c in enumerate(result.claims, start=1):
            claim_dicts.append({
                "id": c.id or f"C{idx}",
                "text": c.text,
                "claim_type": c.claim_type,
                "requirement_ids": c.requirement_ids or (["R1"] if reqs else []),
                "citation_ids": c.citation_ids or [],
            })

        # Fallback if LLM returned no claims for non-empty answer
        if not claim_dicts:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
            for idx, sent in enumerate(sentences[:5], start=1):
                claim_dicts.append({
                    "id": f"C{idx}",
                    "text": sent,
                    "claim_type": "factual",
                    "requirement_ids": ["R1"] if reqs else [],
                    "citation_ids": [],
                })

        logger.info("[ClaimExtractor] Extracted %d atomic claims from answer", len(claim_dicts))

        log_entry["status"] = "completed"
        log_entry["details"]["num_claims"] = len(claim_dicts)

        return {
            **state,
            "claims": claim_dicts,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    except Exception as e:
        logger.error("[ClaimExtractor] Failed: %s", e)
        log_entry["status"] = "failed"
        log_entry["details"]["error"] = str(e)

        # Basic sentence-splitting fallback
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
        fallback_claims = [
            {
                "id": f"C{idx+1}",
                "text": sent,
                "claim_type": "factual",
                "requirement_ids": ["R1"],
                "citation_ids": [],
            }
            for idx, sent in enumerate(sentences[:5])
        ]

        return {
            **state,
            "claims": fallback_claims,
            "stage_logs": state["stage_logs"] + [log_entry],
        }
