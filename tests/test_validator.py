"""
tests/test_validator.py
Unit tests for pipeline/nodes/validator.py, with the embedding call mocked out.

No network/API calls happen here — score_passages_against_query is monkeypatched
to return fixed scores, so these tests run instantly and require no API keys.
"""

import pipeline.nodes.validator as validator_mod
from pipeline.state import initial_state
from tests.conftest import make_doc


def test_validator_passes_when_enough_docs_clear_threshold(monkeypatch):
    monkeypatch.setattr(validator_mod, "USE_VALIDATOR", True)
    monkeypatch.setattr(validator_mod, "USE_EVIDENCE_COVERAGE", True)
    monkeypatch.setattr(validator_mod, "EVIDENCE_SCORE_THRESHOLD", 0.20)
    monkeypatch.setattr(
        validator_mod, "score_passages_against_query",
        lambda query, texts: [0.5, 0.3, 0.1],
    )

    state = {
        **initial_state("q"),
        "retrieved_docs": [make_doc("a"), make_doc("b", "c2"), make_doc("c", "c3")],
    }
    result = validator_mod.validator_node(state)

    assert result["validation_passed"] is True
    assert len(result["scored_docs"]) == 3


def test_validator_fails_when_too_few_docs_clear_threshold(monkeypatch):
    monkeypatch.setattr(validator_mod, "USE_VALIDATOR", True)
    monkeypatch.setattr(validator_mod, "USE_EVIDENCE_COVERAGE", True)
    monkeypatch.setattr(validator_mod, "COVERAGE_THRESHOLD", 0.60)
    monkeypatch.setattr(
        validator_mod, "score_passages_against_query",
        lambda query, texts: [0.15, 0.10],
    )

    state = {
        **initial_state("q"),
        "retrieved_docs": [make_doc("a"), make_doc("b", "c2")],
    }
    result = validator_mod.validator_node(state)

    assert result["validation_passed"] is False
    assert len(result["scored_docs"]) == 2



def test_validator_ablation_bypass_passes_everything_unscored(monkeypatch):
    monkeypatch.setattr(validator_mod, "USE_VALIDATOR", False)

    docs = [make_doc("a"), make_doc("b", "c2"), make_doc("c", "c3")]
    state = {**initial_state("q"), "retrieved_docs": docs}
    result = validator_mod.validator_node(state)

    assert result["validation_passed"] is True
    assert len(result["scored_docs"]) == len(docs)
    assert all(d["score"] == 1.0 for d in result["scored_docs"])


def test_validator_handles_empty_retrieved_docs(monkeypatch):
    monkeypatch.setattr(validator_mod, "USE_VALIDATOR", True)
    state = {**initial_state("q"), "retrieved_docs": []}
    result = validator_mod.validator_node(state)

    assert result["validation_passed"] is False
    assert result["scored_docs"] == []


def test_summarization_intent_uses_relaxed_threshold_and_min_passages(monkeypatch):
    """
    Regression test for a real bug report (2026-08-20): a 3KB single-chunk document
    (passport acknowledgement slip) scored 0.07 against "what is this document about?"
    - below the normal 0.20 threshold - and MIN_VALID_PASSAGES=2 could never be
    satisfied by a document that only chunks into 1 piece. Both are structural
    mismatches for summarization queries, not real relevance failures.
    """
    monkeypatch.setattr(validator_mod, "USE_VALIDATOR", True)
    monkeypatch.setattr(validator_mod, "SUMMARIZATION_SCORE_THRESHOLD", 0.05)
    monkeypatch.setattr(validator_mod, "SUMMARIZATION_MIN_VALID_PASSAGES", 1)
    monkeypatch.setattr(validator_mod, "EVIDENCE_SCORE_THRESHOLD", 0.20)
    monkeypatch.setattr(validator_mod, "MIN_VALID_PASSAGES", 2)
    monkeypatch.setattr(
        validator_mod, "score_passages_against_query",
        lambda query, texts: [0.07],  # single low-scoring passage, like the real bug
    )

    state = {
        **initial_state("what is this document about?"),
        "intent": "summarization",
        "retrieved_docs": [make_doc("the entire short document")],
    }
    result = validator_mod.validator_node(state)

    assert result["validation_passed"] is True
    assert len(result["scored_docs"]) == 1


def test_non_summarization_intent_still_uses_normal_threshold(monkeypatch):
    """Same low score, but a factual query must NOT get the relaxed bar."""
    monkeypatch.setattr(validator_mod, "USE_VALIDATOR", True)
    monkeypatch.setattr(validator_mod, "SUMMARIZATION_SCORE_THRESHOLD", 0.05)
    monkeypatch.setattr(validator_mod, "SUMMARIZATION_MIN_VALID_PASSAGES", 1)
    monkeypatch.setattr(validator_mod, "EVIDENCE_SCORE_THRESHOLD", 0.20)
    monkeypatch.setattr(validator_mod, "MIN_VALID_PASSAGES", 2)
    monkeypatch.setattr(
        validator_mod, "score_passages_against_query",
        lambda query, texts: [0.07],
    )

    state = {
        **initial_state("what is the exact fee amount?"),
        "intent": "factual",
        "retrieved_docs": [make_doc("the entire short document")],
    }
    result = validator_mod.validator_node(state)

    assert result["validation_passed"] is False


def test_validator_survives_scoring_exception(monkeypatch):
    """A scoring failure must not crash the pipeline — it should fail validation
    gracefully so the router can retry, per the node's own docstring contract."""
    monkeypatch.setattr(validator_mod, "USE_VALIDATOR", True)

    def _raise(*a, **kw):
        raise RuntimeError("embedding API down")

    monkeypatch.setattr(validator_mod, "score_passages_against_query", _raise)

    state = {**initial_state("q"), "retrieved_docs": [make_doc("a")]}
    result = validator_mod.validator_node(state)

    assert result["validation_passed"] is False
    assert "error" in result and result["error"]
