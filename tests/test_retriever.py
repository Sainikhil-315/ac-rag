"""
tests/test_retriever.py
Unit tests for pipeline/nodes/retriever.py pure helper functions.

_escalate_plan_for_retry is the regression guard for the bug documented in
GAPS_AND_IMPROVEMENTS.md item A.1: retries used to repeat the exact same
k/fetch_k/lambda_mult, so a validator-rejected retrieval was guaranteed to
fail identically on every retry. These tests pin the widening behaviour down.
"""

from pipeline.nodes.retriever import (
    _escalate_plan_for_retry,
    _apply_modality_filter,
    _deduplicate,
)
from tests.conftest import make_doc

BASE_PLAN = {
    "k": 4,
    "fetch_k": 12,
    "lambda_mult": 0.5,
    "modality_filter": "all",
    "use_multi_query": False,
    "retrieval_depth": "shallow",
}


def test_no_escalation_on_first_attempt():
    plan = _escalate_plan_for_retry(BASE_PLAN, retry_count=0)
    assert plan == BASE_PLAN


def test_escalation_widens_k_and_fetch_k_progressively():
    plan1 = _escalate_plan_for_retry(BASE_PLAN, retry_count=1)
    plan2 = _escalate_plan_for_retry(BASE_PLAN, retry_count=2)

    assert plan1["k"] > BASE_PLAN["k"]
    assert plan2["k"] > plan1["k"], "each retry must widen further than the last"
    assert plan1["fetch_k"] > BASE_PLAN["fetch_k"]
    assert plan2["fetch_k"] > plan1["fetch_k"]


def test_escalation_lowers_lambda_for_more_diversity():
    plan1 = _escalate_plan_for_retry(BASE_PLAN, retry_count=1)
    plan2 = _escalate_plan_for_retry(BASE_PLAN, retry_count=2)
    assert plan1["lambda_mult"] < BASE_PLAN["lambda_mult"]
    assert plan2["lambda_mult"] <= plan1["lambda_mult"]


def test_escalation_never_produces_lower_lambda_than_floor():
    plan = _escalate_plan_for_retry(BASE_PLAN, retry_count=10)
    assert plan["lambda_mult"] >= 0.1


def test_escalation_caps_k_at_retrieval_k_max():
    from config.settings import RETRIEVAL_K_MAX
    plan = _escalate_plan_for_retry(BASE_PLAN, retry_count=10)
    assert plan["k"] <= RETRIEVAL_K_MAX


def test_escalation_preserves_other_plan_fields():
    plan = _escalate_plan_for_retry(BASE_PLAN, retry_count=1)
    assert plan["modality_filter"] == BASE_PLAN["modality_filter"]
    assert plan["use_multi_query"] == BASE_PLAN["use_multi_query"]


def test_modality_filter_all_returns_everything():
    docs = [make_doc("a", modality="text"), make_doc("b", modality="figure")]
    assert _apply_modality_filter(docs, "all") == docs


def test_modality_filter_keeps_only_matching():
    docs = [make_doc("a", modality="text"), make_doc("b", modality="figure")]
    filtered = _apply_modality_filter(docs, "figure")
    assert len(filtered) == 1
    assert filtered[0]["content"] == "b"


def test_modality_filter_falls_back_to_all_if_empty_result():
    docs = [make_doc("a", modality="text"), make_doc("b", modality="text")]
    filtered = _apply_modality_filter(docs, "figure")
    assert filtered == docs, "should fall back to unfiltered rather than return empty"


def test_deduplicate_removes_repeated_chunk_ids():
    docs = [
        make_doc("a", chunk_id="c1"),
        make_doc("a-dup", chunk_id="c1"),
        make_doc("b", chunk_id="c2"),
    ]
    unique = _deduplicate(docs)
    assert len(unique) == 2
    assert [d["chunk_id"] for d in unique] == ["c1", "c2"]


def test_deduplicate_preserves_first_occurrence_order():
    docs = [make_doc("first", chunk_id="c1"), make_doc("second", chunk_id="c1")]
    unique = _deduplicate(docs)
    assert unique[0]["content"] == "first"
