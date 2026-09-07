"""
tests/test_evidence_adaptive_rag.py
Comprehensive Unit & Integration Tests for Evidence-Driven Adaptive AC-RAG.
"""

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
