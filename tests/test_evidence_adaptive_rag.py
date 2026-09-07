"""
tests/test_evidence_adaptive_rag.py
Comprehensive Unit & Integration Tests for Evidence-Driven Adaptive AC-RAG.
"""

from unittest.mock import MagicMock
import pytest
from pipeline.retrieval.bm25_retriever import BM25Retriever
from pipeline.retrieval.hybrid import hybrid_fuse_results, reciprocal_rank_fusion
from pipeline.state import initial_state
from pipeline.nodes.query_analyzer import query_analyzer_node
from pipeline.nodes.validator import validator_node
from pipeline.nodes.claim_extractor import claim_extractor_node
from pipeline.nodes.claim_verifier import claim_verifier_node
from pipeline.nodes.contradiction_detector import contradiction_detector_node
from pipeline.nodes.critic import critic_node
from pipeline.nodes.targeted_retrieval import make_targeted_retrieval_node
from pipeline.nodes.entry_router import make_entry_router_node
from pipeline.router import (
    route_after_validator,
    route_after_critic,
    route_after_retriever,
)
from pipeline.graph import _route_after_retry_increment, _end_max_retries_node
from config.settings import MAX_RETRIEVAL_ROUNDS, MAX_GENERATION_REPAIRS, MAX_TOTAL_CONTROL_STEPS
from tests.conftest import make_doc


def test_bm25_retriever_indexing_and_search():
    bm25 = BM25Retriever()
    docs = [
        {"content": "MAX_RETRIES configuration parameter is set to 3 in settings.", "metadata": {"chunk_id": "c1", "source": "s1"}},
        {"content": "The vector database uses FAISS backend for fast similarity search.", "metadata": {"chunk_id": "c2", "source": "s2"}},
    ]
    bm25.index_documents(docs)
    results = bm25.search("MAX_RETRIES", top_k=1)
    assert len(results) == 1
    assert results[0]["metadata"]["chunk_id"] == "c1"
    assert results[0]["score"] > 0.0


def test_hybrid_fusion_rrf():
    dense = [{"content": "passage A", "metadata": {"chunk_id": "cA"}, "score": 0.9}]
    bm25 = [{"content": "passage B", "metadata": {"chunk_id": "cB"}, "score": 0.8}]

    fused = hybrid_fuse_results(dense_results=dense, bm25_results=bm25, top_k=2)
    assert len(fused) == 2
    chunk_ids = {f["metadata"]["chunk_id"] for f in fused}
    assert chunk_ids == {"cA", "cB"}


def test_evidence_coverage_engine_requirements():
    state = initial_state("What embedding model and chunk size are used?")
    state["evidence_requirements"] = [
        {"id": "R1", "requirement": "embedding model", "priority": "critical", "keywords": ["embedding", "model"]},
        {"id": "R2", "requirement": "chunk size", "priority": "critical", "keywords": ["chunk", "size"]},
    ]
    state["retrieved_docs"] = [
        make_doc("The system uses text-embedding-3-large model.", "c1"),
        make_doc("The chunk size is configured to 512 tokens.", "c2"),
    ]

    result = validator_node(state)
    assert "evidence_coverage" in result
    assert len(result["evidence_coverage"]) == 2
    assert result["validation_passed"] is True


def test_claim_extractor_fallbacks():
    state = initial_state("What is the chunk overlap?")
    state["answer"] = "The system uses a chunk overlap of 64 tokens."
    state["evidence_requirements"] = [
        {"id": "R1", "requirement": "chunk overlap", "priority": "critical"}
    ]

    result = claim_extractor_node(state)
    assert "claims" in result
    assert len(result["claims"]) >= 1
    assert "64 tokens" in result["claims"][0]["text"]


def test_claim_verifier_logic():
    state = initial_state("q")
    state["claims"] = [
        {"id": "C1", "text": "The chunk overlap is 64 tokens.", "claim_type": "numeric", "requirement_ids": ["R1"], "citation_ids": []}
    ]
    state["retrieved_evidence"] = [
        {
            "id": "E1",
            "content": "The system uses CHUNK_OVERLAP = 64 tokens for chunking.",
            "source": "settings.py",
            "chunk_id": "c1",
            "modality": "text",
            "retrieval_method": "bm25",
            "retrieval_score": 0.9,
            "rerank_score": 0.9,
        }
    ]

    result = claim_verifier_node(state)
    assert "claim_verifications" in result
    assert len(result["claim_verifications"]) == 1
    assert result["claim_verifications"][0]["status"] in ("SUPPORTED", "PARTIALLY_SUPPORTED")


def test_contradiction_detector_empty():
    state = initial_state("q")
    state["retrieved_docs"] = [make_doc("a", "c1")]
    result = contradiction_detector_node(state)
    assert result["contradictions"] == []


def test_critic_confidence_and_trace_calculation():
    state = initial_state("What is the embedding model?")
    state["answer"] = "The system uses text-embedding-3-large."
    state["evidence_requirements"] = [{"id": "R1", "requirement": "embedding model", "priority": "critical"}]
    state["evidence_coverage"] = [{"requirement_id": "R1", "status": "SUPPORTED", "score": 0.9, "supporting_evidence_ids": ["E1"]}]
    state["claim_verifications"] = [{"claim_id": "C1", "status": "SUPPORTED", "support_score": 0.9, "contradiction_score": 0.0, "evidence_ids": ["E1"], "reason": "Match"}]
    state["contradictions"] = []

    result = critic_node(state)
    assert "evidence_confidence" in result
    assert result["evidence_confidence"] > 0.5
    assert "evidence_trace" in result
    assert "requirements" in result["evidence_trace"]


# ── Regression Tests for Issues 1-6 & 8 ───────────────────────────────────────

def test_regression_issue1_claim_verifier_exception_fails_safe(monkeypatch):
    """Scenario A: Claim verification exception MUST produce UNSUPPORTED, score 0.0."""
    state = initial_state("What is X?")
    state["claims"] = [{"id": "C1", "text": "X is 10."}]
    state["retrieved_docs"] = [make_doc("X is 10", "c1")]

    # Force get_llm to throw an exception
    monkeypatch.setattr("pipeline.nodes.claim_verifier.get_llm", MagicMock(side_effect=RuntimeError("LLM call failed")))

    result = claim_verifier_node(state)
    assert "claim_verifications" in result
    verifs = result["claim_verifications"]
    assert len(verifs) == 1
    assert verifs[0]["status"] == "UNSUPPORTED"
    assert verifs[0]["support_score"] == 0.0
    assert verifs[0]["reason"] == "Claim verification could not be established."


def test_regression_issue2_retrieval_budget_enforcement():
    """Scenario B: Retrieval budget enforcement (retrieval_attempts >= MAX_RETRIEVAL_ROUNDS)."""
    state = initial_state("What is X?")
    state["validation_passed"] = False
    state["missing_requirements"] = ["R1"]
    state["retrieval_attempts"] = MAX_RETRIEVAL_ROUNDS

    route = route_after_validator(state)
    assert route == "end_max_retries"


def test_regression_issue3_targeted_retrieval_budget_increment():
    """Scenario C: Targeted retrieval increments retrieval_attempts and router enforces cap."""
    mock_vsm = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "MAX_RETRIES = 3"
    mock_doc.metadata = {"chunk_id": "c1", "source": "settings.py", "modality": "text", "score": 0.9}
    mock_vsm.bm25_search.return_value = [mock_doc]

    targeted_node = make_targeted_retrieval_node(mock_vsm)
    state = initial_state("What is MAX_RETRIES?")
    state["missing_requirements"] = ["R1"]
    state["evidence_requirements"] = [{"id": "R1", "requirement": "MAX_RETRIES", "requirement_type": "numeric"}]
    state["retrieval_attempts"] = 2

    res_state = targeted_node(state)
    assert res_state["retrieval_attempts"] == 3
    assert res_state["total_control_steps"] == 1

    # Router should block further targeted retrieval when retrieval_attempts >= MAX_RETRIEVAL_ROUNDS
    res_state["validation_passed"] = False
    next_route = route_after_validator(res_state)
    assert next_route == "end_max_retries"


def test_regression_issue2_generation_repair_budget_enforcement():
    """Scenario D: Generation repair budget enforcement."""
    state = initial_state("What is X?")
    state["critic_passed"] = False
    state["retry_reason"] = "format"
    state["repair_attempts"] = MAX_GENERATION_REPAIRS

    route = route_after_critic(state)
    assert route == "end_max_retries"


def test_regression_issue4_total_control_budget_termination():
    """Scenario E: Total control budget termination (total_control_steps >= MAX_TOTAL_CONTROL_STEPS)."""
    state = initial_state("What is X?")
    state["total_control_steps"] = MAX_TOTAL_CONTROL_STEPS
    state["validation_passed"] = False

    assert route_after_validator(state) == "end_max_retries"
    assert route_after_retriever(state) == "end_max_retries"
    assert route_after_critic(state) == "end_max_retries"
    assert _route_after_retry_increment(state) == "end_max_retries"


def test_regression_issue5_entry_router_lexical_bypass():
    """Scenario F: Low dense similarity + strong lexical match routes to RAG."""
    mock_vsm = MagicMock()
    mock_vsm.similarity_search_with_score.return_value = [(make_doc("unrelated text", "c1"), 0.15)]
    mock_bm25_doc = MagicMock()
    mock_bm25_doc.metadata = {"score": 5.4}
    mock_vsm.bm25_search.return_value = [mock_bm25_doc]

    router_node = make_entry_router_node(mock_vsm)
    state = initial_state("What is MAX_RETRIES?")

    res_state = router_node(state)
    assert res_state["route"] == "rag"


def test_regression_issue6_numeric_config_evidence_coverage():
    """Scenario G: Numeric / configuration evidence coverage receives high score for exact match."""
    state = initial_state("What is MAX_RETRIES?")
    state["evidence_requirements"] = [
        {
            "id": "R1",
            "requirement": "MAX_RETRIES value",
            "requirement_type": "numeric",
            "priority": "critical",
            "keywords": ["MAX_RETRIES", "3"],
        }
    ]
    state["retrieved_docs"] = [
        make_doc("The system configures MAX_RETRIES = 3.", "c1")
    ]

    result = validator_node(state)
    assert result["validation_passed"] is True
    cov = result["evidence_coverage"][0]
    assert cov["status"] == "SUPPORTED"
    assert cov["score"] >= 0.75


def test_regression_issue6_table_modality_evidence_coverage():
    """Scenario H: Table modality requirement strongly prefers table evidence."""
    state = initial_state("Show comparison table")
    state["evidence_requirements"] = [
        {
            "id": "R1",
            "requirement": "comparison matrix",
            "requirement_type": "table",
            "expected_modality": "table",
            "priority": "high",
            "keywords": ["matrix", "table"],
        }
    ]
    state["retrieved_docs"] = [
        {
            "content": "| Model | Score |\n| A | 90 |",
            "modality": "table",
            "chunk_id": "c_table",
            "metadata": {"source": "eval.pdf", "modality": "table"},
        }
    ]

    result = validator_node(state)
    assert result["validation_passed"] is True
    cov = result["evidence_coverage"][0]
    assert cov["status"] == "SUPPORTED"


def test_regression_issue4_repeated_failure_safe_termination():
    """Scenario I: Repeated failure leads to safe termination / abstention."""
    from pipeline.nodes.direct_responder import _UNKNOWN_RESPONSE
    state = initial_state("Unanswerable question?")
    state["answer"] = None
    state["claim_verifications"] = [{"claim_id": "C1", "status": "UNSUPPORTED", "support_score": 0.0, "reason": "No evidence"}]

    term_state = _end_max_retries_node(state)
    assert term_state["answer"] == _UNKNOWN_RESPONSE
    assert "Max retries exhausted" in term_state["error"]
