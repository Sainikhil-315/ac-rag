"""
tests/test_retrieval_metrics.py
Unit tests for Recall@k, MRR, Precision@k in evaluation/metrics.py.

These are pure-Python, no LLM/embedding calls — fast and CI-friendly.
"""

import pytest

from evaluation.metrics import (
    recall_at_k,
    mrr,
    precision_at_k,
    retrieval_metrics,
)


# ── recall_at_k ──────────────────────────────────────────────────────────────

class TestRecallAtK:
    def test_all_relevant_in_top_k(self):
        # Top 3 retrieved, all 3 are relevant → 1.0
        assert recall_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_partial_relevant_in_top_k(self):
        # 2 relevant chunks total, 1 in top-3 → 0.5
        assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == 0.5

    def test_none_relevant_in_top_k(self):
        assert recall_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0

    def test_k_smaller_than_retrieved(self):
        # Only the first 2 are inspected; b is not in top-2
        assert recall_at_k(["a", "x", "b"], ["a", "b"], k=2) == 0.5

    def test_no_relevant_chunks_returns_one(self):
        # Vacuously satisfied — caller treats as un-evaluable, not scored 0
        assert recall_at_k(["a", "b"], [], k=2) == 1.0

    def test_no_retrieved_returns_zero(self):
        assert recall_at_k([], ["a", "b"], k=4) == 0.0

    def test_k_larger_than_retrieved(self):
        # Looking past what was retrieved; should not crash
        assert recall_at_k(["a"], ["a", "b"], k=10) == 0.5


# ── MRR ──────────────────────────────────────────────────────────────────────

class TestMRR:
    def test_first_position(self):
        assert mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_second_position(self):
        assert mrr(["x", "a", "b"], ["a"]) == 0.5

    def test_third_position(self):
        assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(0.3333, abs=1e-3)

    def test_none_relevant(self):
        assert mrr(["x", "y", "z"], ["a"]) == 0.0

    def test_empty_retrieved(self):
        assert mrr([], ["a"]) == 0.0

    def test_no_relevant_chunks_returns_one(self):
        assert mrr(["a", "b"], []) == 1.0


# ── precision_at_k ───────────────────────────────────────────────────────────

class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], ["a", "b"], k=2) == 1.0

    def test_half_relevant(self):
        assert precision_at_k(["a", "x"], ["a", "b"], k=2) == 0.5

    def test_k_zero_returns_zero(self):
        assert precision_at_k(["a"], ["a"], k=0) == 0.0

    def test_no_retrieved(self):
        assert precision_at_k([], ["a"], k=5) == 0.0


# ── retrieval_metrics (composite) ────────────────────────────────────────────

class TestRetrievalMetrics:
    def test_returns_expected_keys(self):
        docs = [
            {"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"},
            {"chunk_id": "d"}, {"chunk_id": "e"},
        ]
        out = retrieval_metrics(docs, relevant_chunk_ids=["b", "d"])
        assert "mrr" in out
        assert "recall@4" in out
        assert "recall@6" in out
        assert "recall@12" in out
        assert "precision@4" in out

    def test_first_relevant_at_rank_2(self):
        # b at index 1 (rank 2) → mrr = 0.5
        docs = [{"chunk_id": "x"}, {"chunk_id": "b"}]
        out = retrieval_metrics(docs, relevant_chunk_ids=["b"])
        assert out["mrr"] == 0.5

    def test_recall_at_default_ks(self):
        # 1 relevant in top-4 out of 2 total → 0.5; 1 in top-12 → 0.5
        docs = [{"chunk_id": "a"}] + [{"chunk_id": f"x{i}"} for i in range(20)]
        out = retrieval_metrics(docs, relevant_chunk_ids=["a", "z"])
        assert out["recall@4"]  == 0.5
        assert out["recall@12"] == 0.5

    def test_empty_relevant_returns_empty_dict(self):
        out = retrieval_metrics([{"chunk_id": "a"}], relevant_chunk_ids=None)
        assert out == {}
        out = retrieval_metrics([{"chunk_id": "a"}], relevant_chunk_ids=[])
        assert out == {}

    def test_doc_without_chunk_id_is_ignored(self):
        docs = [{"chunk_id": "a"}, {"no_id": True}, {"chunk_id": "b"}]
        # a is at rank 1
        out = retrieval_metrics(docs, relevant_chunk_ids=["a"])
        assert out["mrr"] == 1.0

    def test_custom_k_values(self):
        docs = [{"chunk_id": str(i)} for i in range(5)]
        out = retrieval_metrics(docs, relevant_chunk_ids=["1", "3"], k_values=[1, 3, 5])
        assert "recall@1" in out
        assert "recall@3" in out
        assert "recall@5" in out
        # 0 in top-1, 1 in top-3 (just "1"), 2 in top-5
        assert out["recall@1"] == 0.0
        assert out["recall@3"] == 0.5
        assert out["recall@5"] == 1.0
