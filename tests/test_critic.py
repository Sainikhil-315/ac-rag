"""
tests/test_critic.py
Unit tests for pipeline/nodes/critic.py pure helper functions.

_sanitize_retry_reason exists specifically to guard against garbage LLM output
(observed from Groq/Llama and older GPT models) confusing the router into an
infinite retry loop — these tests cover the documented decision logic.
"""

from pipeline.nodes.critic import (
    _sanitize_retry_reason,
    _all_pass,
    _compute_overall,
)
from pipeline.state import CriticScores


class _FakeEval:
    """Minimal stand-in for CriticEvaluation, avoids constructing a real Pydantic model."""
    def __init__(self, faithfulness=5, completeness=5, table_accuracy=5,
                 figure_accuracy=5, conciseness=5):
        self.faithfulness = faithfulness
        self.completeness = completeness
        self.table_accuracy = table_accuracy
        self.figure_accuracy = figure_accuracy
        self.conciseness = conciseness


# ── _sanitize_retry_reason ───────────────────────────────────────────────────

def test_sanitize_exact_valid_values_pass_through():
    assert _sanitize_retry_reason("content") == "content"
    assert _sanitize_retry_reason("format") == "format"
    assert _sanitize_retry_reason("none") == "none"


def test_sanitize_is_case_and_whitespace_insensitive():
    assert _sanitize_retry_reason("  Content  ") == "content"
    assert _sanitize_retry_reason("FORMAT") == "format"


def test_sanitize_empty_or_none_defaults_to_content():
    assert _sanitize_retry_reason("") == "content"
    assert _sanitize_retry_reason(None) == "content"


def test_sanitize_format_without_content_keyword():
    assert _sanitize_retry_reason("formatting issue") == "format"


def test_sanitize_prefers_content_when_both_keywords_present():
    assert _sanitize_retry_reason("content and format both wrong") == "content"


def test_sanitize_recognises_content_related_keywords():
    for phrase in ("faithfulness issue", "incomplete answer", "table error", "figure wrong"):
        assert _sanitize_retry_reason(phrase) == "content"


def test_sanitize_garbage_llm_output_defaults_to_content():
    """Regression guard: malformed structured output must never crash the router
    or produce an unrecognised route — safest fallback is 'content' (full retry)."""
    for junk in ("}", "><|python_tag|>", ", and "):
        assert _sanitize_retry_reason(junk) == "content"


# ── _all_pass / _compute_overall ─────────────────────────────────────────────

def test_all_pass_true_when_every_dimension_meets_minimum():
    eval_result = _FakeEval(faithfulness=4, completeness=4, table_accuracy=4,
                             figure_accuracy=4, conciseness=4)
    assert _all_pass(eval_result) is True


def test_all_pass_false_if_any_dimension_below_minimum():
    eval_result = _FakeEval(faithfulness=3, completeness=5, table_accuracy=5,
                             figure_accuracy=5, conciseness=5)
    assert _all_pass(eval_result) is False


def test_compute_overall_is_mean_of_all_dimensions():
    eval_result = _FakeEval(faithfulness=5, completeness=4, table_accuracy=3,
                             figure_accuracy=4, conciseness=4)
    assert _compute_overall(eval_result) == 4.0
