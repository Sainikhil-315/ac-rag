"""
pipeline/nodes/query_analyzer.py
Query Analyzer Node — first stage of the AC-RAG pipeline.

Responsibilities:
  1. Intent detection    : classify query as factual / analytical / comparative / summarization
  2. Complexity scoring  : 0.0 (simple lookup) → 1.0 (multi-hop reasoning)
  3. Query decomposition : if complexity ≥ 0.6, split into focused sub-queries

Design:
  Uses structured LLM output (Pydantic model) to guarantee parseable JSON.
  Pydantic validation acts as a contract — downstream nodes can trust field types.

  Complexity thresholds:
    0.0 – 0.39 : simple   → k=4,  no decomposition
    0.4 – 0.69 : moderate → k=6,  optional decomposition
    0.7 – 1.0  : complex  → k=10–12, decomposition enabled
"""

import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from pipeline.state import ACRagState
from utils.llm_factory import get_llm

logger = logging.getLogger(__name__)

# ── Pydantic output schema ─────────────────────────────────────────────────────

class EvidenceRequirementSpec(BaseModel):
    id: str = Field(description="Requirement ID, e.g., 'R1', 'R2'")
    requirement: str = Field(description="Short description of what specific information must be established")
    requirement_type: str = Field(
        default="fact",
        description="Type: 'fact', 'value', 'definition', 'procedure', 'comparison', 'relationship', 'summary', 'numeric', 'table', or 'figure'"
    )
    priority: str = Field(
        default="critical",
        description="Priority: 'critical', 'high', 'medium', or 'low'"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Key search terms for retrieval"
    )
    expected_modality: str = Field(
        default="text",
        description="Expected modality: 'text', 'table', 'figure', or 'all'"
    )


class QueryAnalysis(BaseModel):
    rewritten_query: str = Field(
        description=(
            "A clarified, retrieval-optimised version of the original query. "
            "Fix grammar, resolve vague pronouns, expand abbreviations, and make the "
            "question self-contained and specific. If the original is already clear, "
            "return it unchanged."
        )
    )
    intent: str = Field(
        description="Query intent: 'factual', 'analytical', 'comparative', or 'summarization'"
    )
    complexity_score: float = Field(
        ge=0.0, le=1.0,
        description="Query complexity from 0.0 (simple) to 1.0 (complex multi-hop)"
    )
    complexity_reason: str = Field(
        description="One-sentence explanation of the complexity score"
    )
    decomposed_queries: List[str] = Field(
        default_factory=list,
        description="Sub-queries if complex (complexity >= 0.6), else empty list"
    )
    evidence_requirements: List[EvidenceRequirementSpec] = Field(
        default_factory=list,
        description="List of 1 to 6 independently verifiable evidence requirements needed to answer the question"
    )


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an evidence requirement planner and query analysis expert for a document QA system.
Analyse the user's query and return a structured JSON response.

Query rewriting rules (rewritten_query field):
  - Fix spelling mistakes, grammar errors, and incomplete sentences
  - Resolve vague pronouns ("it", "that", "they") using context clues in the query,
    OR using the conversation history if provided below.
  - Expand abbreviations if their meaning is inferable
  - Make the question self-contained and specific enough for vector/lexical search
  - If original query is already clear, return it unchanged

Intent categories:
  - factual       : asking for a specific fact, number, date, or definition
  - analytical    : asking to explain, analyse, or reason about something
  - comparative   : asking to compare two or more things
  - summarization : asking for a summary or overview

Evidence Requirement Planning rules (evidence_requirements field):
  - Identify the minimum independently verifiable information needed to answer the query correctly.
  - Return short requirement descriptions, NOT answers.
  - Assign stable IDs: R1, R2, R3, etc.
  - For factual/value/numeric questions, preserve exact entity names and target parameter names in keywords.
  - For comparative questions, create corresponding requirements for each side/aspect.
  - For table/figure questions, set expected_modality to 'table' or 'figure'.
  - Maximum 6 requirements.

Complexity scoring guide:
  0.0–0.3 : Single fact lookup, one document needed
  0.4–0.6 : Requires synthesising a few passages, some reasoning
  0.7–0.9 : Multi-hop reasoning, cross-document synthesis
  1.0     : Extremely complex, requires full document understanding

Query decomposition rules:
  - Only decompose if complexity_score >= 0.6
  - Base sub-queries on the rewritten_query, not the original
  - Each sub-query must be self-contained and answerable independently
  - Maximum 4 sub-queries
  - If simple, return an empty list for decomposed_queries"""

_HUMAN_PROMPT = "{history_block}Query: {query}"

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", _HUMAN_PROMPT),
])

_MAX_HISTORY_TURNS = 3  # most recent N turns only


def _format_history(history) -> str:
    if not history:
        return ""
    recent = history[-_MAX_HISTORY_TURNS:]
    lines = ["Conversation history (most recent last):"]
    for turn in recent:
        role = turn.get("role", "user")
        content = (turn.get("content") or "")[:500]
        lines.append(f"  {role}: {content}")
    lines.append("")
    return "\n".join(lines) + "\n"


# ── LLM with structured output ────────────────────────────────────────────────

def _get_analyzer_chain():
    return _prompt | get_llm().with_structured_output(QueryAnalysis)


# ── Node function ─────────────────────────────────────────────────────────────

def query_analyzer_node(state: ACRagState) -> ACRagState:
    """
    LangGraph node: Query Analyzer & Evidence Requirement Planner.
    Reads:  state["query"], state["conversation_history"]
    Writes: state["rewritten_query"], state["intent"], state["complexity_score"],
            state["decomposed_queries"], state["evidence_requirements"]
    """
    query = state["query"]
    history_block = _format_history(state.get("conversation_history"))
    log_entry = {"stage": "query_analyzer", "status": "started", "details": {"query": query}}
    logger.info("[QueryAnalyzer] Analysing query: '%s'", query)

    try:
        chain = _get_analyzer_chain()
        result: QueryAnalysis = chain.invoke({"query": query, "history_block": history_block})

        # If not complex enough, clear any decomposed queries
        if result.complexity_score < 0.6:
            result.decomposed_queries = []

        # Convert Pydantic EvidenceRequirementSpec items into state EvidenceRequirement dicts
        req_dicts = []
        if result.evidence_requirements:
            for r in result.evidence_requirements:
                req_dicts.append({
                    "id": r.id or f"R{len(req_dicts)+1}",
                    "requirement": r.requirement,
                    "requirement_type": r.requirement_type,
                    "priority": r.priority,
                    "keywords": r.keywords,
                    "expected_modality": r.expected_modality,
                    "status": "MISSING",
                    "support_score": 0.0,
                    "supporting_evidence_ids": [],
                })
        else:
            # Default requirement if none provided
            req_dicts.append({
                "id": "R1",
                "requirement": result.rewritten_query or query,
                "requirement_type": "fact",
                "priority": "critical",
                "keywords": [query],
                "expected_modality": "all",
                "status": "MISSING",
                "support_score": 0.0,
                "supporting_evidence_ids": [],
            })

        # Log if the query was actually changed
        if result.rewritten_query != query:
            logger.info("[QueryAnalyzer] Query rewritten: '%s' → '%s'", query, result.rewritten_query)
        else:
            logger.info("[QueryAnalyzer] Query unchanged after rewrite")

        logger.info(
            "[QueryAnalyzer] intent=%s | complexity=%.2f | sub-queries=%d | requirements=%d",
            result.intent, result.complexity_score, len(result.decomposed_queries), len(req_dicts)
        )

        log_entry["status"] = "completed"
        log_entry["details"].update({
            "rewritten_query": result.rewritten_query,
            "intent": result.intent,
            "complexity_score": result.complexity_score,
            "complexity_reason": result.complexity_reason,
            "num_sub_queries": len(result.decomposed_queries),
            "num_requirements": len(req_dicts),
        })

        return {
            **state,
            "rewritten_query": result.rewritten_query,
            "intent": result.intent,
            "complexity_score": result.complexity_score,
            "decomposed_queries": result.decomposed_queries,
            "evidence_requirements": req_dicts,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    except Exception as e:
        logger.error("[QueryAnalyzer] Failed: %s", e)
        log_entry["status"] = "failed"
        log_entry["details"]["error"] = str(e)

        # Build fallback single requirement
        fallback_reqs = [{
            "id": "R1",
            "requirement": query,
            "requirement_type": "fact",
            "priority": "critical",
            "keywords": [query],
            "expected_modality": "all",
            "status": "MISSING",
            "support_score": 0.0,
            "supporting_evidence_ids": [],
        }]

        return {
            **state,
            "rewritten_query": query,
            "intent": "factual",
            "complexity_score": 0.3,
            "decomposed_queries": [],
            "evidence_requirements": fallback_reqs,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

