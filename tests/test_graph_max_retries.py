"""
tests/test_graph_max_retries.py
Regression test for a real bug report (2026-08-20): a query about content genuinely
absent from the document went through the full RAG path (Entry Router's coarse check
passed it as "rag"), the Validator correctly rejected every retry as irrelevant, and
the pipeline hard-failed with "Max retries exhausted" and NO answer at all — instead
of the same graceful "I don't know" the Entry Router's "unknown" route gives for the
exact same underlying situation when it catches it earlier.

Known tradeoff (documented, not "fixed away"): this also means a genuine retrieval
false-negative (relevant content exists but scoring missed it — see
FAILURE_ANALYSIS.md) now looks identical to a true out-of-scope query from the user's
perspective. state["error"] is still set in both cases so evaluation/runner.py's error
counting is unaffected — only the user-facing answer text changed.
"""

from pipeline.graph import _end_max_retries_node
from pipeline.state import initial_state
from pipeline.nodes.direct_responder import _UNKNOWN_RESPONSE


def test_max_retries_with_no_answer_gets_graceful_fallback():
    """Validator-exhausted case: generator never ran, answer stayed None."""
    state = {**initial_state("who is hanumanthu rao"), "retry_count": 3}
    result = _end_max_retries_node(state)

    assert result["answer"] == _UNKNOWN_RESPONSE
    assert result["error"] == "Max retries exhausted without a passing answer."


def test_max_retries_with_existing_answer_is_preserved():
    """Critic-exhausted case: generator DID produce something, critic kept rejecting
    it. The generated answer is a better best-effort result than a generic fallback —
    must not be overwritten."""
    state = {
        **initial_state("what is the fee?"),
        "retry_count": 3,
        "answer": "The fee is Rs. 1500, based on the available context.",
    }
    result = _end_max_retries_node(state)

    assert result["answer"] == "The fee is Rs. 1500, based on the available context."
    assert result["error"] == "Max retries exhausted without a passing answer."


def test_max_retries_preserves_existing_error_message():
    """If some earlier node already set a more specific error, don't clobber it."""
    state = {
        **initial_state("q"),
        "retry_count": 3,
        "error": "Retriever failed: connection timeout",
    }
    result = _end_max_retries_node(state)
    assert result["error"] == "Retriever failed: connection timeout"
