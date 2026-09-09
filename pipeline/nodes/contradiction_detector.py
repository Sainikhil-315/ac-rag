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


_CONTRADICTION_SYSTEM = """You are a strict contradiction detector for a research QA system.

Your job is ONLY to identify genuine factual contradictions between claims
and/or evidence passages.

A contradiction exists ONLY when two statements:
1. refer to the SAME entity, parameter, metric, or scenario,
2. refer to the SAME conditions, scope, experiment, or version, and
3. assert mutually exclusive factual values or propositions.

IMPORTANT:
- Comparative claims about different systems are NEVER contradictions.
  Example: "AC-RAG supports adaptive retrieval" and "RAG does not support
  adaptive retrieval" are a valid comparison, NOT a contradiction.
  Do not compare the truth value of a capability across different systems.
  A contradiction must occur WITHIN the same system/entity.
  Example: "AC-RAG supports 8 dimensions" and "RAG supports 3 dimensions"
  are compatible comparative statements.
- Partial support is not the same as no support.
  The symbol "~" means partially supported. Therefore:
  "ReAct partially supports adaptive retrieval" and
  "ReAct does not fully support adaptive retrieval" are compatible statements,
  NOT contradictions.
- Different metrics are NOT contradictions.
  Example: "faithfulness decreased to 0.75" and "context relevance decreased
  to 0.72" are compatible because they describe different metrics.
- A general statement and a more specific statement are NOT contradictions.
  Example: "validation ensures relevance" and "validation filters
  low-quality content" are compatible.
- A statement and a subset/detail of that statement are NOT contradictions.
- Statements describing different experiments, ablations, datasets, versions,
  time periods, or conditions are NOT contradictions.
- Baseline versus improved-system results are NOT contradictions.
- A claim and evidence supporting that claim are NOT contradictions.
- Consistent or redundant statements are NOT contradictions.

Only return a contradiction when the two statements CANNOT both be true
under the same scope and conditions.

For every detected conflict:
- use resolution_status="unresolved" ONLY for a genuine unresolved conflict.
- use resolution_status="resolved_by_version" when different versions,
  experiments, datasets, or conditions explain the difference.
- use resolution_status="resolved_by_authority" when source authority
  resolves the apparent conflict.

- TABLE EVIDENCE:
  When evidence is a Markdown/structured table, interpret each symbol only
  within its row and column context.
  ✓ / ✔ means supported/yes.
  ~ means partially supported.
  ✗ / × means not supported/no.
  Never treat a symbol in one column as evidence about another column.
  A table row is not a contradiction merely because different systems have
  different symbols in the same row.
  For example, "RAG: ✗" and "AC-RAG: ✓" means the two systems differ; it does
  NOT mean AC-RAG contradicts itself.

When in doubt, DO NOT flag a contradiction.
False positives are worse than missing a weak or ambiguous contradiction."""

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

def _normalize_table_evidence(content: str) -> str:
    """
    Convert Markdown table evidence into explicit row/column statements.

    The original evidence is not modified. This representation is used
    only when sending structured table evidence to the contradiction detector.
    """
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    table_lines = [
        line for line in lines
        if line.startswith("|") and line.endswith("|")
    ]

    if len(table_lines) < 2:
        return content

    def split_row(line: str) -> List[str]:
        return [cell.strip() for cell in line.strip("|").split("|")]

    headers = split_row(table_lines[0])

    symbol_map = {
        "✓": "SUPPORTED",
        "✔": "SUPPORTED",
        "~": "PARTIALLY_SUPPORTED",
        "✗": "NOT_SUPPORTED",
        "×": "NOT_SUPPORTED",
        "x": "NOT_SUPPORTED",
        "X": "NOT_SUPPORTED",
    }

    normalized = ["STRUCTURED TABLE EVIDENCE:"]

    for line in table_lines[1:]:
        cells = split_row(line)

        # Skip Markdown separator row: | --- | --- | --- |
        if cells and all(
            cell.replace("-", "").replace(":", "").strip() == ""
            for cell in cells
        ):
            continue

        if not cells or not cells[0]:
            continue

        normalized.append(f"\nDimension: {cells[0]}")

        for i, value in enumerate(cells[1:], start=1):
            if i >= len(headers):
                break

            column = headers[i]
            semantic_value = symbol_map.get(value.strip(), value.strip())

            normalized.append(f"{column}: {semantic_value}")

    return "\n".join(normalized) if len(normalized) > 1 else content

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
            content = e.get("content", "")

            if str(e.get("modality", "")).lower() == "table":
                content = _normalize_table_evidence(content)

            evidence_lines.append(
                f"[{e.get('id', '?')}] "
                f"(source: {e.get('source', '?')}, page: {e.get('page', '?')})\n"
                f"{content}"
            )
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
        verified_status = {
            v.get("claim_id"): v.get("status")
            for v in (state.get("claim_verifications") or [])
        }
        contradiction_dicts: List[Contradiction] = []
        if result.has_contradictions and result.contradictions:
            for idx, item in enumerate(result.contradictions, start=1):
                status = (item.resolution_status or "unresolved").strip().lower()
                claim_1_status = verified_status.get(item.claim_id_1)
                claim_2_status = verified_status.get(item.claim_id_2)

                # If both claims are independently verified as supported,
                # a claim-vs-claim contradiction reported by the detector
                # is treated as a false positive and does not block the answer.
                if (
                    item.claim_id_1
                    and item.claim_id_2
                    and claim_1_status == "SUPPORTED"
                    and claim_2_status == "SUPPORTED"
                ):
                    logger.info(
                        "[ContradictionDetector] Ignoring claim-level contradiction "
                        "%s: both claims independently verified as SUPPORTED.",
                        item.id,
                    )
                    continue
                # Only unresolved conflicts should block the answer.
                # Version/authority differences are retained in the evidence trace
                # but must not trigger a retry.
                contradiction_dicts.append({
                    "id": item.id or f"X{idx}   ",
                    "claim_id_1": item.claim_id_1 or None,
                    "claim_id_2": item.claim_id_2 or None,
                    "evidence_id_1": item.evidence_id_1 or "E1",
                    "evidence_id_2": item.evidence_id_2 or "E2",
                    "conflict_description": item.conflict_description,
                    "resolution_status": status,
                })
                
        logger.info("[ContradictionDetector] Detected %d contradictions", len(contradiction_dicts))

        log_entry["status"] = "completed"
        log_entry["details"]["num_contradictions"] = len(contradiction_dicts)
        log_entry["details"]["contradictions"] = contradiction_dicts

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
