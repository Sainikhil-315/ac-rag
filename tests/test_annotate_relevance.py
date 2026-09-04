"""
tests/test_annotate_relevance.py
Unit tests for the two-tier relevance decision in evaluation/annotate_relevance.py.

These tests cover the relevance heuristic only — no vector store is loaded.
The integration test (annotator end-to-end against the real index) is run
manually via `python -m evaluation.annotate_relevance`.
"""

import pytest

from evaluation.annotate_relevance import _is_relevant, _content_tokens, _overlap_fraction


# ── _content_tokens ──────────────────────────────────────────────────────────

class TestContentTokens:
    def test_lowercases_and_drops_short(self):
        # Hyphen splits via [a-z0-9]+ regex → "ac" and "rag"; "ac" is 2 chars, dropped
        assert _content_tokens("The AC-RAG system") == ["rag", "system"]

    def test_drops_stopwords(self):
        out = _content_tokens("the cat is on the table")
        # "the", "is", "on" dropped; cat (3 chars) kept; table kept
        assert "the" not in out
        assert "cat" in out
        assert "table" in out

    def test_preserves_alphanumerics(self):
        out = _content_tokens("GPT-4o and MMR are 2024 systems")
        # "gpt" kept; "4o" is 2 chars so dropped; "mmr" kept; "2024" kept; "systems" kept
        assert "gpt" in out
        assert "4o" not in out  # too short
        assert "mmr" in out
        assert "2024" in out
        assert "systems" in out

    def test_empty_string(self):
        assert _content_tokens("") == []


# ── _is_relevant: long-answer (token-overlap) path ────────────────────────────

class TestIsRelevantLong:
    def test_exact_match_high_overlap(self):
        # Long reference, same text in chunk → relevant
        ref = "Adaptive retrieval combined with validation and self-reflection improves reliability"
        chunk = "Adaptive retrieval combined with validation and self-reflection improves reliability in document question answering."
        assert _is_relevant(ref, chunk, overlap=0.3) is True

    def test_partial_overlap_above_threshold(self):
        ref = "Adaptive retrieval validation self-reflection reliability document question answering"
        chunk = "Adaptive retrieval combined with validation and self-reflection improves reliability"
        # Most tokens present, overlap > 0.3
        assert _is_relevant(ref, chunk, overlap=0.3) is True

    def test_low_overlap_below_threshold(self):
        ref = "Adaptive retrieval validation self-reflection reliability document question answering"
        chunk = "The cat sat on the mat and looked at the dog"
        # None of the content tokens present
        assert _is_relevant(ref, chunk, overlap=0.3) is False

    def test_higher_threshold_more_strict(self):
        ref = "adaptive retrieval validation self-reflection reliability document question answering"
        chunk = "Adaptive retrieval combined with validation"
        # 0.5 threshold makes this fail when 0.3 would pass
        assert _is_relevant(ref, chunk, overlap=0.3) is True
        assert _is_relevant(ref, chunk, overlap=0.9) is False


# ── _is_relevant: short-answer (substring) path ───────────────────────────────

class TestIsRelevantShort:
    def test_short_answer_substring_match(self):
        # Reference has < MIN_CONTENT_TOKENS (2) content tokens → falls back to substring
        # "MMR" → tokens: ["mmr"] (1 token, below threshold of 2)
        ref = "MMR."
        chunk = "We use Maximum Marginal Relevance (MMR) for diverse retrieval."
        assert _is_relevant(ref, chunk, overlap=0.3) is True

    def test_short_answer_no_substring_match(self):
        ref = "MMR."
        chunk = "The system uses cosine similarity for retrieval."
        assert _is_relevant(ref, chunk, overlap=0.3) is False

    def test_short_answer_case_insensitive(self):
        # Substring match is case-insensitive (chunk lowered before check)
        ref = "GPT-4o"
        chunk = "The model gpt-4o is used for generation."
        assert _is_relevant(ref, chunk, overlap=0.3) is True

    def test_empty_reference_never_relevant(self):
        # Empty ref short-circuits to False
        assert _is_relevant("", "anything here", overlap=0.3) is False


# ── _overlap_fraction ────────────────────────────────────────────────────────

class TestOverlapFraction:
    def test_all_in(self):
        assert _overlap_fraction(["a", "b", "c"], "a b c d e") == 1.0

    def test_none_in(self):
        assert _overlap_fraction(["a", "b"], "x y z") == 0.0

    def test_half_in(self):
        assert _overlap_fraction(["a", "b"], "a x y z") == 0.5

    def test_empty_reference(self):
        assert _overlap_fraction([], "anything") == 0.0
