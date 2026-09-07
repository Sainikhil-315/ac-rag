"""
pipeline/state.py
LangGraph shared state schema for the AC-RAG pipeline.

ALL nodes read from and write to this TypedDict.
Using TypedDict (not dataclass) because LangGraph requires it for state management
and supports partial updates — each node only writes the fields it owns.

Field ownership (which node writes what):
  query_analyzer   → rewritten_query, intent, complexity_score, decomposed_queries
  retrieval_planner→ retrieval_plan
  retriever        → retrieved_docs
  validator        → scored_docs, validation_passed
  context_refiner  → refined_context
  generator        → answer, answer_with_attribution
  critic           → critic_scores, critic_passed, critic_feedback

Shared lifecycle fields:
  retry_count      → incremented by the graph router on each retry loop
  stage_logs       → append-only list; every node appends its trace entry
  error            → set if a node fails; triggers graceful exit
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class RetrievalPlan(TypedDict):
    """
    Output of the Retrieval Planner node.
    Drives how the Retriever node executes its search.
    """
    k: int                      # number of final passages to retrieve (4–12)
    fetch_k: int                # MMR candidate pool size
    lambda_mult: float          # MMR diversity parameter (0=diverse, 1=relevant)
    modality_filter: str        # "text" | "table" | "figure" | "all"
    use_multi_query: bool       # decompose query into sub-queries before retrieval
    retrieval_depth: str        # "shallow" | "standard" | "deep"
    strategy: Optional[str]     # "dense" | "bm25" | "hybrid"
    lexical_weight: Optional[float]
    dense_weight: Optional[float]


class EvidenceRequirement(TypedDict):
    """
    Structured information requirement identified from user query.
    """
    id: str                               # e.g. "R1", "R2"
    requirement: str                      # description of what must be established
    requirement_type: str                 # "fact" | "value" | "definition" | "procedure" | "comparison" | "relationship" | "summary" | "numeric" | "table" | "figure"
    priority: str                         # "critical" | "high" | "medium" | "low"
    keywords: List[str]                   # key terms for retrieval
    expected_modality: str                # "text" | "table" | "figure" | "all"
    status: str                           # "SUPPORTED" | "PARTIAL" | "MISSING" | "CONTRADICTED" | "NOT_APPLICABLE"
    support_score: float                  # 0.0 to 1.0
    supporting_evidence_ids: List[str]    # IDs of EvidenceItems satisfying this


class EvidenceItem(TypedDict):
    """
    Canonical evidence item retrieved and scored by the system.
    """
    id: str                               # e.g. "E1", "E2"
    content: str                          # passage content
    source: str                           # file name / path
    page: Optional[int]                   # page number
    section: Optional[str]                # section header
    chunk_id: str                         # unique chunk identifier
    modality: str                         # "text" | "table" | "figure"
    retrieval_method: str                 # "dense" | "bm25" | "hybrid" | "targeted" | "table" | "figure"
    retrieval_score: float                # raw score
    rerank_score: float                   # fused/reranked score


class EvidenceCoverage(TypedDict):
    """
    Requirement-level coverage evaluation.
    """
    requirement_id: str                   # "R1", "R2"
    status: str                           # "SUPPORTED" | "PARTIAL" | "MISSING" | "CONTRADICTED" | "NOT_APPLICABLE"
    score: float                          # 0.0 to 1.0
    supporting_evidence_ids: List[str]
    missing_reason: Optional[str]


class Claim(TypedDict):
    """
    Atomic factual claim extracted from candidate answer.
    """
    id: str                               # e.g. "C1", "C2"
    text: str                             # atomic statement
    claim_type: str                       # "factual" | "numeric" | "procedural" | "definitional" | "comparative"
    requirement_ids: List[str]            # requirements this claim addresses
    citation_ids: List[str]               # evidence IDs cited


class ClaimVerification(TypedDict):
    """
    Verification outcome of an atomic claim against evidence.
    """
    claim_id: str                         # "C1"
    status: str                           # "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED"
    support_score: float                  # 0.0 to 1.0
    contradiction_score: float            # 0.0 to 1.0
    evidence_ids: List[str]               # matching evidence IDs
    reason: str                           # concise verification explanation


class Contradiction(TypedDict):
    """
    Detected factual conflict between evidence passages or claims.
    """
    id: str                               # e.g. "X1"
    claim_id_1: Optional[str]
    claim_id_2: Optional[str]
    evidence_id_1: str
    evidence_id_2: str
    conflict_description: str
    resolution_status: str                # "unresolved" | "resolved_by_version" | "resolved_by_authority"


class EvidenceTrace(TypedDict):
    """
    Auditable provenance trace mapping requirements → evidence → claims → verification.
    """
    requirements: List[Dict[str, Any]]
    evidence_items: List[Dict[str, Any]]
    claims: List[Dict[str, Any]]
    verifications: List[Dict[str, Any]]
    contradictions: List[Dict[str, Any]]
    confidence_summary: Dict[str, Any]


class CriticScores(TypedDict):
    """
    Output of the Critic (self-reflection) node.
    All dimensions scored 1–5. Accept only if ALL ≥ CRITIC_MIN_SCORE.
    """
    faithfulness: int
    completeness: int
    table_accuracy: int
    figure_accuracy: int
    conciseness: int
    overall: float              # mean of all dimensions
    feedback: str               # critic's textual feedback for retry


class StageLogEntry(TypedDict):
    stage: str
    status: str                 # "started" | "completed" | "failed" | "skipped"
    details: Dict[str, Any]


class ACRagState(TypedDict):
    """
    Full shared state for the AC-RAG LangGraph pipeline.
    """
    # ── Input ─────────────────────────────────────────────────────────────────
    query: str
    conversation_history: Optional[List[Dict[str, str]]]

    # ── Entry Router output ───────────────────────────────────────────────────
    route: Optional[str]                 # "rag" | "unknown"

    # ── Query Analyzer outputs ────────────────────────────────────────────────
    rewritten_query: Optional[str]      # clarified query optimised for vector retrieval
    intent: Optional[str]               # "factual" | "analytical" | "comparative" | "summarization"
    complexity_score: Optional[float]   # 0.0 (simple) → 1.0 (complex)
    decomposed_queries: Optional[List[str]]  # sub-queries if complex
    evidence_requirements: Optional[List[EvidenceRequirement]] # structured information requirements

    # ── Retrieval Planner outputs ─────────────────────────────────────────────
    retrieval_plan: Optional[RetrievalPlan]

    # ── Retriever & Hybrid Search outputs ─────────────────────────────────────
    retrieved_docs: Optional[List[Dict[str, Any]]]  # list of {content, metadata}
    retrieved_evidence: Optional[List[EvidenceItem]] # structured EvidenceItem objects

    # ── Validator / Coverage Engine outputs ───────────────────────────────────
    scored_docs: Optional[List[Dict[str, Any]]]     # docs with "score" field added
    validation_passed: Optional[bool]
    evidence_coverage: Optional[List[EvidenceCoverage]]
    missing_requirements: Optional[List[str]]        # IDs of unresolved requirements
    targeted_queries: Optional[List[Dict[str, Any]]]  # queries generated for missing requirements

    # ── Context Refiner outputs ───────────────────────────────────────────────
    refined_context: Optional[str]      # deduplicated, re-ranked, compressed context

    # ── Generator outputs ─────────────────────────────────────────────────────
    answer: Optional[str]
    answer_with_attribution: Optional[List[Dict[str, Any]]]  # [{sentence, source_chunk_id}]

    # ── Claim Extraction & Verification outputs ───────────────────────────────
    claims: Optional[List[Claim]]
    claim_verifications: Optional[List[ClaimVerification]]
    contradictions: Optional[List[Contradiction]]
    is_answerable: Optional[bool]
    abstained: Optional[bool]
    evidence_confidence: Optional[float]
    evidence_trace: Optional[EvidenceTrace]

    # ── Critic outputs ────────────────────────────────────────────────────────
    critic_scores: Optional[CriticScores]
    critic_passed: Optional[bool]
    critic_feedback: Optional[str]

    # ── Pipeline control & budgets ────────────────────────────────────────────
    retry_count: int
    retry_reason: Optional[str]         # "content" | "format" | "coverage" | "unsupported_claim" | "contradiction"
    retrieval_attempts: int
    reflection_attempts: int
    repair_attempts: int
    stage_logs: List[StageLogEntry]     # append-only trace for research analysis
    error: Optional[str]                # non-None triggers graceful pipeline exit

    # ── Cost / usage tracking ─────────────────────────────────────────────────
    token_usage: Optional[Dict[str, Any]]


def initial_state(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> ACRagState:
    """
    Create a clean initial state for a new query.
    All optional fields start as None; pipeline fills them in order.
    """
    return ACRagState(
        query=query,
        conversation_history=conversation_history or None,
        route=None,
        rewritten_query=None,
        intent=None,
        complexity_score=None,
        decomposed_queries=None,
        evidence_requirements=None,
        retrieval_plan=None,
        retrieved_docs=None,
        retrieved_evidence=None,
        scored_docs=None,
        validation_passed=None,
        evidence_coverage=None,
        missing_requirements=None,
        targeted_queries=None,
        refined_context=None,
        answer=None,
        answer_with_attribution=None,
        claims=None,
        claim_verifications=None,
        contradictions=None,
        is_answerable=None,
        abstained=None,
        evidence_confidence=None,
        evidence_trace=None,
        critic_scores=None,
        critic_passed=None,
        critic_feedback=None,
        retry_count=0,
        retry_reason=None,
        retrieval_attempts=0,
        reflection_attempts=0,
        repair_attempts=0,
        stage_logs=[],
        error=None,
        token_usage=None,
    )

